#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:09:25 2024

@author: arthurlecoz

03_06_compute_fooof_params.py
"""
# %% Paths
import mne 
import os 

import numpy as np
import pandas as pd
import SLHIP_config_ALC as config

from statsmodels.nonparametric.smoothers_lowess import lowess
from fooof import FOOOF
from fooof.bands import Bands
from glob import glob

cleanDataPath = config.cleanDataPath
powerPath = config.powerPath

fooofparamsPath = os.path.join(powerPath, 'fooof_params')
reports_path = os.path.join(fooofparamsPath, "reports")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

channels = np.array(config.eeg_channels)

subtypes = ["C1", "HI", "N1"]
psd_palette = ["#8d99ae", "#d00000", "#ffb703"]

freqs = np.linspace(0.5, 40, 159)

method = "welch"
fmin = 0.5
fmax = 40
n_fft = 1024
n_per_seg = n_fft
n_overlap = int(n_per_seg/2)
window = "hamming"

bands = Bands({
    "delta" : (.5, 4),
    "theta" : (4, 8),
    "alpha" : (8, 12),
    "sigma" : (12, 16),
    "iota" : (25, 35)
    })

files = glob(os.path.join(cleanDataPath, "epochs_probes", "*.fif"))

coi = ["sub_id", "subtype", "mindstate", "channel",
       "offset", "exponent", "r_squared", "error"]

# %% Loop

mindstates = ['ON', 'MW', 'HALLU', 'MB']

    
this_df_savepath = os.path.join(
    fooofparamsPath, "fooof_params.csv"
    )

big_dic = {f : [] for f in coi}
    
for i_file, file in enumerate(files) :
    sub_id = file.split('probes/')[-1].split('_epo')[0]
    subtype = sub_id[:2]
        
    print(f"...processing {sub_id}... [{i_file+1}/{len(files)}]")
    
    epochs = mne.read_epochs(file, preload = True)
    metadata = epochs.metadata
 
    for ms in mindstates:
        print(f'processing {ms}')
        if ms not in metadata.mindstate.unique() : continue
        
        temp_power = epochs[epochs.metadata.mindstate == ms].compute_psd(
                method = method,
                fmin = fmin, 
                fmax = fmax,
                n_fft = n_fft,
                n_overlap = n_overlap,
                n_per_seg = n_per_seg,
                window = window,
                picks = channels
                )
        for i_ch, channel in enumerate(channels) :
            print(f'processing channel {channel}')
            for i_epoch in range(
                    len(epochs[epochs.metadata.mindstate == ms])
                    ) :
                this_power = temp_power[i_epoch]                   
                
                psd = lowess(np.squeeze(
                    this_power.copy().pick(channel).get_data()), 
                    freqs, 0.075)[:, 1]
                
                if np.any(psd < 0) :
                    for id_0 in np.where(psd<0)[0] :
                        psd[id_0] = abs(psd).min()
                        
                fm = FOOOF(peak_width_limits = [.5, 4], aperiodic_mode="fixed")
                fm.add_data(freqs, psd)
                fm.fit()
                
                big_dic['sub_id'].append(sub_id)
                big_dic['subtype'].append(subtype)
                big_dic['mindstate'].append(ms)
                big_dic['channel'].append(channel)
                
                big_dic['offset'].append(
                    fm.get_results().aperiodic_params[0]
                    )
                big_dic['exponent'].append(
                    fm.get_results().aperiodic_params[1]
                    )
                big_dic['r_squared'].append(fm.r_squared_)
                big_dic['error'].append(fm.error_)

df = pd.DataFrame.from_dict(big_dic)
df.to_csv(this_df_savepath)
