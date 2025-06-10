#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:26:16 2025

@author: arthurlecoz

06_04_yasa_gethypno.py
"""
# %% Paths & Packages

import numpy as np
import mne 
import yasa
import os
from glob import glob

#### Paths
if 'arthur' in os.getcwd():
    path_root='/Volumes/DDE_ALC/PhD/SLHIP'
else:
    path_root='your_path'

path_data=os.path.join(path_root, '00_Raw')
path_edfusleep=os.path.join(path_root, "01_Preproc", "edfs_usleep")
path_usleep=os.path.join(path_root, '06_USleep')

# Paths to the EEG files, here brainvision files
files = glob(os.path.join(path_edfusleep, '*.edf'))

# Get channel names
raw = mne.io.read_raw_brainvision(
    glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))[0]
    )
channels = raw.ch_names
eeg_channels = channels[:-4]
eog_channel = "HEOG"

channel_oi = ['C3','TP10']
njobs=4

# %% Script

str_to_int = {"N1" : 1, "N2" : 2, "N3" : 3, "W" : 0, "R" : 5}

def map_values(x):
    return str_to_int[x]

# file = files[0]
for i, edf_file in enumerate(files) :
    sub_id = edf_file.split('usleep/')[-1][:-4]
    this_savepath = os.path.join(path_usleep, 'yasa', f"{sub_id}_hypnodensity.csv")
    
    if not os.path.exists(this_savepath) :
        print(f"\n...Processing {sub_id}: n°{i+1} out of {len(files)}")
        
        raw = mne.io.read_raw_edf(
            edf_file, include = channel_oi, preload = True
            )
        mne.set_eeg_reference(raw, ["TP10"], copy = False)
        raw.drop_channels(['TP10'])
        
        sfreq = raw.info["sfreq"] 
        raw.resample(100, npad="auto", n_jobs=njobs)
        
        sls = yasa.SleepStaging(
            raw, 
            eeg_name="C3",
            )
        y_pred = sls.predict()
        hypnodensity = sls.predict_proba()
        confidence = sls.predict_proba().max(1)
        
        hypnodensity.insert(0, "confidence", confidence)
        
        vectorized_map_values = np.vectorize(map_values)
        y_pred_mapped = vectorized_map_values(y_pred)
        
        hypnodensity.insert(0, "scorred_stage", y_pred)
        hypnodensity.insert(0, "int_stage", y_pred_mapped)
        hypnodensity.reset_index(inplace = True)
        
        hypnodensity.to_csv(this_savepath)        
    
    print(f"\n{sub_id} was just processed")

