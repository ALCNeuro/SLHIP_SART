#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:26:16 2025

@author: arthurlecoz

06_02_usleep_gethypno.py
"""
# %% Paths & Packages

import logging
import mne 
from usleep_api import USleepAPI
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

# amount of cpu used for functions with multiprocessing :
n_jobs = 4

# Key copy-pasted from the link above
key='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NDIyNDY3NjYsImlhdCI6MTc0MjIwMzU2NiwibmJmIjoxNzQyMjAzNTY2LCJpZGVudGl0eSI6IjhmYWQyMzgxMmFiNyJ9.rcH1L1Ih-GplMyw38ScMFODQH8gZrMCV7TGIGxit-Ic'

# Get channel names
raw = mne.io.read_raw_brainvision(
    glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))[0]
    )
channels = raw.ch_names
eeg_channels = channels[:-4]
eog_channel = "HEOG"

# channel_groups = [[str(ch), eog_channel] for ch in eeg_channels]

channel_groups =[['F3', 'HEOG'],
                ['F4', 'HEOG'],
                ['C3', 'HEOG'],
                ['P4', 'HEOG'],
                ['O1', 'HEOG'],
                ['O2', 'HEOG'],
                ['Fz', 'HEOG'],
                ['FC1', 'HEOG'],
                ['FC2', 'HEOG'],
                ['CP1', 'HEOG'],
                ['CP2', 'HEOG'],
                ['F1', 'HEOG'],
                ['F2', 'HEOG'],
                ['P1', 'HEOG'],
                ['P2', 'HEOG'],
                ['AF3', 'HEOG'],
                ['AF4', 'HEOG'],
                ['PO3', 'HEOG'],
                ['PO4', 'HEOG'],
                ['CPz', 'HEOG'],
                ['POz', 'HEOG'],
                ['Oz', 'HEOG']
                ]

# %% Script

# file = files[0]
for i, edf_file in enumerate(files) :
    sub_id = edf_file.split('usleep/')[-1][:-4]
    this_savepath = os.path.join(path_usleep, f"{sub_id}_hypnodensity.npy")
    
    if not os.path.exists(this_savepath) :
        print(f"\n...Processing {sub_id}: n°{i+1} out of {len(files)}")
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("api_example")
        
        # Create an API object and (optionally) a new session.
        api = USleepAPI(api_token=key)
        session = api.new_session(session_name="my_session")
        
        # Specify which model you wanna use
        # logger.info(f"Available models: {session.get_model_names()}")
        session.set_model("U-Sleep v2.0") # EEG + EOG # else, use U-Sleep-EEG v2.0 for EEG-only
            
        # Upload an edf file 
        session.upload_file(edf_file, anonymize_before_upload=False)
        
        # Start the prediction:
        win_prediction = 1 # in second
        session.predict(data_per_prediction=128*win_prediction,
                        channel_groups=channel_groups
                        )
        
        success = session.wait_for_completion()
        
        if success:
            # Fetch hypnogram
            hyp = session.get_hypnogram()
            logger.info(hyp["hypnogram"])
        
            # Export hypnogram file
            session.download_hypnogram(
                out_path=this_savepath, file_type="npy",with_confidence_scores=True
                )
        else:
            logger.error("Prediction failed.")
        
        # Delete session 
        session.delete_session()
    
    print(f"\n{sub_id} was just processed")

