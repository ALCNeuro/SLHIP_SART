#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:09:25 2024

@author: arthurlecoz

03_03_compute_periodic_power.py
"""
# %% Paths
import mne 
import os 

import numpy as np
import pandas as pd
import SLHIP_config_ALC as config

from statsmodels.nonparametric.smoothers_lowess import lowess
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from glob import glob

cleanDataPath = config.cleanDataPath
powerPath = config.powerPath

periodicPath = os.path.join(powerPath, 'periodic')
reports_path = os.path.join(periodicPath, "reports")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

channels = np.array(config.eeg_channels)

subtypes = ["C1", "HI", "N1"]
psd_palette = ["#8d99ae", "#d00000", "#ffb703"]

threshold = dict(eeg = 600e-6)

freqs = np.linspace(0.5, 40, 159)

method = "welch"
fmin = 0.5
fmax = 40
n_fft = 1024
n_per_seg = n_fft
n_overlap = int(n_per_seg/2)
window = "hamming"

coi = ["sub_id", "subtype", "channel", "mindstate", "sleepiness", "n_probe",
       "n_block", "voluntary", "freq_bin", "power_value"]

files = glob(os.path.join(cleanDataPath, "epochs_probes", "*.fif"))

mindstates = ['ON', 'MW', 'HALLU', 'MB', 'FORGOT']

# %% Single Process

redo = 0

all_csv = []

for i_f, file in enumerate(files) :
    
    sub_id = file.split('probes/')[-1].split('_epo')[0]
    daytime = sub_id[-2:]
    subtype = sub_id[:2]
    sub_id = sub_id[:-3]
    
    this_subject_savepath = os.path.join(
        periodicPath, f"{sub_id}_{daytime}_periodic_psd.csv"
        )
    
    if os.path.exists(this_subject_savepath) and not redo : 
        print(f"[{i_f+1}/{len(files)}] Loading and adding {sub_id}, {daytime} session.")
        subdf = pd.read_csv(this_subject_savepath)
        del subdf['Unnamed: 0']
        all_csv.append(subdf)
    
    else : 
        sub_dic = {f:[] for f in coi}
        print(f"[{i_f+1}/{len(files)}] Loading and processing {sub_id}, {daytime} session.")
        
        epochs = mne.read_epochs(file, preload = True)
        epochs.drop_bad(threshold)
        
        metadata = epochs.metadata
 
        for ms in mindstates:
            print(f"[{i_f+1}/{len(files)}] Processing {ms}")
            if ms not in metadata.mindstate.unique() : continue
            temp_list = []
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
            this_metadata = metadata.loc[metadata.mindstate == ms]
            
            for i_epoch in range(len(this_metadata)) :
                this_power = temp_power[i_epoch]   
                this_sleepi = this_metadata.sleepiness.iloc[i_epoch]           
                this_vol = this_metadata.voluntary.iloc[i_epoch]     
                this_block = this_metadata.nblock.iloc[i_epoch]     
                this_probe = this_metadata.nprobe.iloc[i_epoch]     
                
                for channel in channels:
                        
                    psd = lowess(np.squeeze(
                        this_power.copy().pick(channel).get_data()), 
                        freqs, 0.075)[:, 1]
                    
                    if not psd.shape:
                        input('inspect')
                    
                    if np.any(psd < 0) :
                        for id_0 in np.where(psd<0)[0] :
                            psd[id_0] = abs(psd).min()
                            
                    fm = FOOOF(peak_width_limits = [.5, 4], aperiodic_mode="fixed")
                    fm.add_data(freqs, psd)
                    fm.fit()
                    
                    if fm.r_squared_ < .95 : continue
                    if fm.error_ > .1 : continue
                    
                    init_ap_fit = gen_aperiodic(
                        fm.freqs, 
                        fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
                        )
                    
                    init_flat_spec = fm.power_spectrum - init_ap_fit
    
                    for i, f in enumerate(freqs):
                        sub_dic['sub_id'].append(sub_id)
                        sub_dic['subtype'].append(subtype)
                        sub_dic['mindstate'].append(ms)
                        sub_dic['channel'].append(channel)
                        sub_dic['sleepiness'].append(this_sleepi)
                        sub_dic['voluntary'].append(this_vol)
                        sub_dic['n_block'].append(this_block)
                        sub_dic['n_probe'].append(this_probe)
                        
                        sub_dic["freq_bin"].append(f)
                        sub_dic["power_value"].append(init_flat_spec[i])
    
        subdf = pd.DataFrame.from_dict(sub_dic)
        subdf.to_csv(this_subject_savepath)
        all_csv.append(subdf)
        del subdf
    
df = pd.concat(all_csv)
df.to_csv(os.path.join(
    periodicPath, "all_periodic_psd.csv"
    ))
