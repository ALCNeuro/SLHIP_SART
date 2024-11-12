#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:09:25 2024

@author: arthurlecoz

03_01_compute_periodic_power.py
"""
# %% Paths
import mne 
import os 
import multiprocessing
import pickle

import numpy as np
import SLHIP_config_ALC as config

from statsmodels.nonparametric.smoothers_lowess import lowess
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.bands import Bands
from glob import glob

cleanDataPath = config.cleanDataPath
powerPath = config.powerPath

periodicPath = os.path.join(powerPath, 'periodic')
reports_path = os.path.join(periodicPath, "reports")
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

# %% Loop

# big_dic_psd_savepath = os.path.join(
#     fig_dir, "fooof_psd_flat_spectra_2.pickle"
#     )

mindstates = ['ON', 'MW', 'HALLU', 'MB']

# big_dic = {subtype : {stage : {chan : [] for chan in channels}
#                       for stage in stages} for subtype in subtypes}

    
def compute_periodic_psd(file) :
    
    # if subtype.startswith('N') : continue
    sub_id = file.split('probes/')[-1].split('_epo')[0]
    # session = sub_id[-2:]
    
    this_subject_savepath = os.path.join(
        periodicPath, f"{sub_id}_periodic_psd.pickle"
        )
    
    if not os.path.exists(this_subject_savepath) : 
    
        temp_dic = {ms : {chan : [] for chan in channels}
                              for ms in mindstates}
        
        print(f"...processing {sub_id}")
        
        epochs = mne.read_epochs(file, preload = True)
        metadata = epochs.metadata
 
        for ms in mindstates:
            print(f'processing {ms}')
            if ms not in metadata.mindstate.unique() : 
                for channel in channels :
                    temp_dic[ms][channel].append(np.nan*np.empty(freqs.shape[0]))
            else : 
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
                        
                        init_ap_fit = gen_aperiodic(
                            fm.freqs, 
                            fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
                            )
                        
                        init_flat_spec = fm.power_spectrum - init_ap_fit
                        temp_list.append(init_flat_spec)
                    temp_dic[ms][channel].append(
                        np.nanmean(temp_list, axis = 0)
                        )
        with open (this_subject_savepath, 'wb') as handle:
            pickle.dump(temp_dic, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    from glob import glob
    # Get the list of EEG files
    eeg_files = files
    
    # Set up a pool of worker processes
    pool = multiprocessing.Pool(processes = 4)
    
    # Process the EEG files in parallel
    pool.map(compute_periodic_psd, eeg_files)
    
    # Clean up the pool of worker processes
    pool.close()
    pool.join()
    



