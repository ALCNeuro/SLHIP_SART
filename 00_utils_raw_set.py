#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 11:16:59 2025

@author: arthurlecoz

00_utils_raw_set.py
"""
# %% Paths & Packages

from glob import glob
import os
import mne

#### Paths
if 'arthur' in os.getcwd():
    path_root='/Volumes/DDE_ALC/PhD/SLHIP'
else:
    path_root='your_path'

path_preproc=os.path.join(path_root, '01_Preproc')

files = glob(os.path.join(path_preproc, "raw_icaed", "*raw.fif"))

# %% Script

for i_f, file in enumerate(files) :
    sub_id = file.split("icaed/")[1].split("_raw")[0]
    
    this_savepath = os.path.join(
        path_preproc, "raw_dot_set", f"{sub_id}_raw.set"
        )
    
    if os.path.exists(this_savepath):continue
    
    print(f"...Processing {sub_id} | n°{i_f+1} / {len(files)}")
    
    raw = mne.io.read_raw(file)
    
    mne.export.export_raw(this_savepath, raw)
    