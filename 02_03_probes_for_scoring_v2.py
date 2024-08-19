#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:51:27 2024

@author: arthurlecoz

02_03_probes_for_scoring.py
******************************************************************************

The goal of this script is to extract a scorable 30s window around the probe:
    > With only the channel needed to score : F3, C3, O1 reffed to TP10
    > And HEOG, VEOG to get the eye movements
    > The 10s window used in the next analysis will be the center, 
    10s before and after that window will be taken too
    => Scoring will occur according to the AASM criteria, 
    
    
BRAINSTO :
    
    I SHOULD USE MATPLOTLIB TO DISPLAY ALL EPOCHS PER SUBJECTS
    AND DO A SCRIPT THAT WILL ALLOW ME TO SCORE ONLINE
    
    IF THE HYPNOGRAM EXISTS THEN I SKIP THE SUBJECT
    
    AND I LINK THE HYPNOGRAM TO THE EPOCHS_PROBES.PY SCRIPT AND ADD IT 
    TO THE METADATA!
    
    I would need to find a way to display in matplotlib : 
        3 channels
        VEOG & HEOG 
        at the correct scales
        
        with an input everytime I display a window that would write down
        each time I press a key
        -> Ending up with a 40 (ish) array/list of corresponding to the hypnogram
        
******************************************************************************
"""
# %% Packages, Paths, Variables
#### Packages
import os
import numpy as np
import pandas as pd
import mne
from glob import glob
from scipy.io import loadmat
import SLHIP_config_ALC as cfg 
import matplotlib.pyplot as plt
import warnings

import locale
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')

import matplotlib

#### Paths
if 'arthur' in os.getcwd():
    path_root='/Volumes/DDE_ALC/PhD/SLHIP'
else:
    path_root='your_path'

path_data=os.path.join(path_root, '00_Raw')
path_preproc=os.path.join(path_root, '01_Preproc')

if os.path.exists(os.path.join(path_data, "reports"))==False:
    os.makedirs(os.path.join(path_data, "reports"))
if os.path.exists(os.path.join(path_data, "intermediary"))==False:
    os.makedirs(os.path.join(path_data,"intermediary"))
    
#### Variables    

# Paths to the EEG files, here brainvision files
files = glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))
# amount of cpu used for functions with multiprocessing :
n_jobs = 4

# Columns of interest for probes
# probe_col = [
#     "nprobe","t_probe_th","t_probe_act","nblock","block_cond","ntrial",
#     "PQ1_respkey","PQ2_respkey","PQ3_respkey",
#     "PQ1_resptime","PQ2_resptime","PQ3_resptime",
#     "PQ1_questime","PQ2_questime","PQ3_questime",
#     "PQ1_respval","PQ2_respval","PQ3_respval"
#     ]
# ms_dic = {
#     0 : "MISS",
#     1 : 'ON',
#     2 : 'MW',
#     3 : 'DISTRACTED',
#     4 : 'HALLU',
#     5 : 'MB',
#     6 : 'FORGOT'
#     }

# Epochs threshold to have epochs with : 300µV > PTP amplitude > 1µV
flat_criteria = dict(eeg=1e-6)
# threshold = 300e-6

sub_ids = np.unique(np.array(
    [file.split('experiment/')[1].split('/')[0] for file 
     in glob(os.path.join(path_data, "experiment", "**", "*.mat"))]
    ))

# %% fun

def visu_scoring(
        data, 
        events, 
        scoring_savepath,
        channels_names = ['F3', 'C3', 'O1', 'VEOG', 'HEOG'],
        ):
    
    sleep_scores = []
    for idx, event in enumerate(events[:, 0]):
        fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(16, 16))
        for i in range(data.shape[0]) :
            if i < 3 :
                ax[i].set_ylim([-200, 200])
                ax[i].plot(
                    data[i, event-20*256:event+10*256], 
                    linewidth = .5,
                    color = 'k'
                    )
            else :
                ax[i].set_ylim([-500, 500])
                ax[i].plot(
                    data[i, event-20*256:event+10*256], 
                    linewidth = .5, 
                    color = 'royalblue'
                    )
            ax[i].text(
                -0.05, 
                0.5, 
                channels_names[i], 
                transform=ax[i].transAxes,
                va='center', 
                ha='right', 
                fontsize=12, 
                fontweight='bold'
                )
            ax[i].vlines(
                x=event - (event-20*256), ymin=-500, ymax=500, color = 'r'
                )
        ax[0].set_xlim([0, 30*256])
        ax[0].set_xticks(np.linspace(0, 7680, 7), np.arange(-20, 15, 5))
        ax[data.shape[0]-1].set_xlabel(
            'Time before probe onset (s)', 
            fontsize=12, 
            fontweight='bold'
            )
        fig.tight_layout(pad = 1.5)
        plt.show()
        plt.pause(.1)
        
        score = input(
            f"""Enter sleep score for event {idx+1}/40: 
                Use your keyboard : 
                    0 = Wake
                    1 = N1 
                    2 = N2
                    3 = N3
                    4 = REM
            """
            )
        sleep_scores.append(score)
        plt.close()
        
    with open(scoring_savepath, "w") as file:
        for score in sleep_scores:
            file.write(f"{score}\n")
    
    print(f"Sleep scores have been saved to '{scoring_savepath}'.")
    

# %% Script

for i, file_path in enumerate(files) :
    #### [1] Import Data and Minimally Process it
    sub_id = f"{file_path.split('/sub_')[1][:6]}{file_path.split('SART')[1][:3]}"
    
    if ("HS_007" in sub_id 
        or 'HS_008' in sub_id 
        or 'N1_001_PM' in sub_id):
        continue
    
    print(f"...Processing {sub_id}, file {i+1} / {len(files)}...")
    
    subtype = sub_id[:2]
    session = sub_id[-2:]
    
    et_filepath = glob(os.path.join(
        path_data, 'experiment', f'sub_{sub_id[:-3]}', "*.asc")
        )
    
    scoring_savepath = os.path.join(
        path_preproc, "epochs_scoring", f"{sub_id}_hypno.txt"
        )
    
    if session == 'AM' :
        et_filepath = et_filepath[0]
    else : 
        et_filepath = et_filepath[1]
    
    if os.path.exists(scoring_savepath):
        print(f"...{sub_id}, file {i+1} / {len(files)} Already scored, skipping...")
        continue
    
    raw = cfg.load_and_preprocess_data(file_path)
    raw.pick(['F3', 'C3', 'O1', 'TP10', 'VEOG', 'HEOG'])
    raw.set_eeg_reference(ref_channels = ['TP10'])
    raw.drop_channels('TP10')
    raw.filter(.5, 30)
    
    sf = raw.info['sfreq']
    
    events, event_id = mne.events_from_annotations(raw)
    ms_probes =  np.stack(
        [event for i, event in enumerate(events[events[:, 2] == 128]) 
         if not i%3])
    
    data = raw.get_data(units = {'eeg' : 'uV', 'eog' : 'uV'})
    
    
    
    # visu_scoring(
    #         data, 
    #         ms_probes, 
    #         scoring_savepath,
    #         channels_names = ['F3', 'C3', 'O1', 'VEOG', 'HEOG']
    #         )
  
