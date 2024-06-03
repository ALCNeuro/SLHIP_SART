#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:51:27 2024

@author: arthurlecoz

02_02_epochs_probes.py
******************************************************************************

At the end of the script, will be saved:
    For each subject : 
    * Probes epochs cleaned (Threshold 300µV + ICAed) with metadata :
        in the metadata of the probe epochs will be found :
    [sub_id, subtype, nblock, nprobe, mindstate, voluntary, sleepiness]
        
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

import matplotlib
matplotlib.use('Agg')

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
probe_col = [
    "nprobe","t_probe_th","t_probe_act","nblock","block_cond","ntrial",
    "PQ1_respkey","PQ2_respkey","PQ3_respkey",
    "PQ1_resptime","PQ2_resptime","PQ3_resptime",
    "PQ1_questime","PQ2_questime","PQ3_questime",
    "PQ1_respval","PQ2_respval","PQ3_respval"
    ]
ms_dic = {
    0 : "MISS",
    1 : 'ON',
    2 : 'MW',
    3 : 'DISTRACTED',
    4 : 'HALLU',
    5 : 'MB',
    6 : 'FORGOT'
    }

# Epochs threshold to have epochs with : 300µV > PTP amplitude > 1µV
flat_criteria = dict(eeg=1e-6)
threshold = 300e-6

sub_ids = np.unique(np.array(
    [file.split('experiment/')[1].split('/')[0] for file 
     in glob(os.path.join(path_data, "experiment", "**", "*.mat"))]
    ))

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
    
    this_probes_savename = os.path.join(
        path_preproc, "epochs_probes", f"{sub_id}_epo.fif"
        )
    
    if os.path.exists(this_probes_savename):
        print(f"...{sub_id}, file {i+1} / {len(files)} Already processed, skipping...")
        continue
    
    raw = cfg.load_and_preprocess_data(file_path)
    sf = raw.info['sfreq']
    
    events, event_id = mne.events_from_annotations(raw)
    ms_probes =  np.stack(
        [event for i, event in enumerate(events[events[:, 2] == 128]) 
         if not i%3])

    behav_paths = glob(os.path.join(
        path_data, "experiment", f"sub_{sub_id[:-3]}", "*.mat"
        ))
    
    if len(behav_paths) < 1 :
        print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
        continue
    if session == "AM" :
        behav_path = behav_paths[0]
    else :
        behav_path = behav_paths[1]
    
    mat = loadmat(behav_path)
    df_probe = pd.DataFrame(
        mat['probe_res'], 
        columns = probe_col)
    if any(df_probe.PQ1_respval.isna()) :
        df_probe.PQ1_respval.replace(np.nan, 0, inplace = True)
        
    ms_answers = np.array(
        [ms_dic[value] for value in df_probe.PQ1_respval.values]
        )
    vol_answers = df_probe.PQ2_respval.values
    sleepi_answers = df_probe.PQ3_respval.values
    
    print(f"""\nIn {sub_id} file, were found :
        * {ms_probes.shape[0]} Probes (first question)
        -> {ms_answers.shape[0]} MS Answers
        -> {vol_answers.shape[0]} Voluntary Answers
        -> {sleepi_answers.shape[0]} Sleepiness answers""")
    
    this_icapath = glob(os.path.join(
        path_preproc, "ica_files", f"*{sub_id}*ica.fif"
        )) 
    
    if len(this_icapath) < 1 :
        print(f"No ICA file found for {sub_id}, skipping...")
        continue
    else : 
        this_icapath = this_icapath[0]
    
    ica = mne.preprocessing.read_ica(this_icapath)
    
        
    #### [7] SART Probes

    ms_metadatadic = {
        "sub_id" : [sub_id for i in range(ms_probes.shape[0])], 
        "subtype" : [subtype for i in range(ms_probes.shape[0])], 
        "nblock" : list(np.repeat([0, 1, 2, 3], 10)), 
        "nprobe" : [i%10 for i in range(ms_probes.shape[0])], 
        "mindstate" : list(ms_answers), 
        "voluntary" : list(vol_answers), 
        "sleepiness" : list(sleepi_answers)
        }
                
    probe_metadata = pd.DataFrame.from_dict(ms_metadatadic)
    
    raw_ica = ica.apply(raw.copy())
    
    epochs_probes = mne.Epochs(
        raw_ica, 
        ms_probes, 
        tmin = -10,
        tmax = 0,
        baseline = (None, None),
        preload = True,
        flat = flat_criteria,
        reject=dict(eeg=threshold),
        event_repeated = 'merge'
        )
    good_epochs = [True if not log else False 
                  for i, log in enumerate(epochs_probes.drop_log)]
    epochs_probes.metadata = probe_metadata[good_epochs]
    
    epochs_probes.set_eeg_reference(ref_channels = ['TP9', 'TP10'])
    
    epochs_probes.save(this_probes_savename, overwrite = True)
    
    # plt.close('all')    
