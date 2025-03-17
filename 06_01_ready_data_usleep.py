#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:55:42 2025

@author: arthurlecoz

06_01_ready_data_usleep.py

"""
# %% Paths & Packages

#### Packages
import os
import mne
from glob import glob
import multiprocessing

#### Paths
if 'arthur' in os.getcwd():
    path_root='/Volumes/DDE_ALC/PhD/SLHIP'
else:
    path_root='your_path'

path_data=os.path.join(path_root, '00_Raw')
path_preproc=os.path.join(path_root, '01_Preproc')
path_usleep=os.path.join(path_root, "01_Preproc", "edfs_usleep")

# Paths to the EEG files, here brainvision files
files = glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))

# amount of cpu used for functions with multiprocessing :
n_jobs = 4


# %% Script

file = files[0]

def export_to_edf(file):

    sub_id = f"{file.split('/sub_')[1][:6]}{file.split('SART')[1][:3]}"
    this_savepath = os.path.join(path_usleep, f"{sub_id}.edf")
    raw = mne.io.read_raw_brainvision(vhdr_fname=file)
    
    raw.export(this_savepath, fmt="edf", overwrite=True)
    
if __name__ == '__main__':
    from glob import glob
    # Get the list of EEG files
    files = glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))
    
    # Set up a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = 1)
    
    # Process the EEG files in parallel
    pool.map(export_to_edf, files)
    
    # Clean up the pool of worker processes
    pool.close()
    pool.join()
    