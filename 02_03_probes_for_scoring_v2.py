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
from scipy.stats import zscore
import SLHIP_config_ALC as cfg 
import matplotlib.pyplot as plt
import warnings
import uuid

# This will allow to no be disturbed by the script running later on
import matplotlib
matplotlib.use('Agg')
matplotlib.use('QtAgg')

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
path_scoring_coded = os.path.join(path_root, "07_Scoring", "Coded_figures")

if os.path.exists(os.path.join(path_data, "reports"))==False:
    os.makedirs(os.path.join(path_data, "reports"))
if os.path.exists(os.path.join(path_data, "intermediary"))==False:
    os.makedirs(os.path.join(path_data,"intermediary"))
    
#### Variables    

# Paths to the EEG files, here brainvision files
files = glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))
# amount of cpu used for functions with multiprocessing :
n_jobs = 4

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
        fig, ax = plt.subplots(nrows=data.shape[0], ncols=1, figsize=(16, 16))
        for i in range(data.shape[0]) :
            if i < data.shape[0]-2 :
                ax[i].set_ylim([-150, 150])
                ax[i].plot(
                    data[i, event-20*256:event+10*256], 
                    linewidth = .5,
                    color = 'k'
                    )
                ax[i].set_yticks(np.linspace(-150, 150, 3), np.linspace(-150, 150, 3))
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
            # Remove top, right, and bottom spines
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].set_xlim([0, 30*256])
        
        # Remove xticks for all axes except the last one
        for i in range(len(ax)-1):
            ax[i].set_xticks([])
            
        # Configure the shared x-axis ticks and label on the last subplot
       
        ax[-1].set_xticks(np.linspace(0, 7680, 7))
        ax[-1].set_xticklabels(np.arange(-20, 15, 5))
        ax[-1].set_xlabel(
            'Time before probe onset (s)', 
            fontsize=12, 
            fontweight='bold'
            )
        
        fig.tight_layout(pad = 1)
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
    

# %% Script Score All

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


# %% Narco Hallu Script - Adaptative Threshold

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

channels_oi =  ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'TP9',
  'TP10', 'VEOG', 'HEOG', 'ECG','RESP']

col_oi = ["sub_id", "group", "session", "nprobe", "mindstate", "code"]
big_dic = {col:[ ]for col in col_oi}

for i, file_path in enumerate(files) :
    #### [1] Import Data and Minimally Process it
    sub_id = f"{file_path.split('/sub_')[1][:6]}{file_path.split('SART')[1][:3]}"
    
    if (sub_id.startswith('HS')
        or sub_id.startswith('HI')
        or 'N1_001_PM' in sub_id):
        continue
    
    print(f"...Processing {sub_id}, file {i+1} / {len(files)}...")
    
    subtype = sub_id[:2]
    session = sub_id[-2:]
    
    et_filepath = glob(os.path.join(
        path_data, 'experiment', f'sub_{sub_id[:-3]}', "*.asc")
        )
    
    scoring_savepath = os.path.join(
        path_preproc, "epochs_scoring", f"{sub_id}_hypno_hallu.txt"
        )
    
    behav_paths = glob(os.path.join(
        path_data, "experiment", f"sub_{sub_id[:-3]}", "*.mat"
        ))
    
    #### Extract Behav Infos
    if len(behav_paths) < 1 :
        print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
        # continue
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
    
    raw = cfg.load_and_preprocess_data(file_path)
    raw.pick(channels_oi)
    raw.set_eeg_reference(ref_channels = ['TP9', 'TP10'])
    raw.drop_channels(['TP9', 'TP10'])
    raw.filter(.5, 40)
    raw.filter(1,10, picks=['VEOG', 'HEOG'])
    raw.filter(1,40, picks=['ECG'])
    raw.filter(1,40, picks=['RESP'])
    
    channels_names = ['F3', 'F4', 'C3', 'C4', 'O1', 
                     'O2', 'VEOG', 'HEOG', 'ECG','RESP']
    
    sf = raw.info['sfreq']
    
    events, event_id = mne.events_from_annotations(raw)
    ms_probes =  np.stack(
        [event for i, event in enumerate(events[events[:, 2] == 128]) 
         if not i%3])
    
    data = raw.get_data(units = {'eeg' : 'uV', 'eog' : 'uV', 'ecg' : 'uV'})
    
    if not len(ms_answers) == len(ms_probes) :
        print(f"!!!\n{sub_id} : Careful, inconsistencies found between EEG and Behav\n!!!")
        continue
    
    # hallu_pos = np.where(ms_answers == 'HALLU')[0]
    # hallu_events = ms_probes[hallu_pos]
        
    # Define EEG channels
    eeg_channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    
    # Compute a common 95th percentile for all EEG channels combined.
    # eeg_data_list = [np.abs(data[i, :]) for i, ch in enumerate(channels_names) if ch in eeg_channels]
    # if eeg_data_list:
    #     combined_eeg = np.concatenate(eeg_data_list)
    #     common_eeg_limit = np.percentile(combined_eeg, 98)
    #     # Avoid a zero limit in edge cases.
    #     if common_eeg_limit == 0:
    #         common_eeg_limit = 100
    # else:
    #     common_eeg_limit = None

    # Precompute individual limits for non-EEG channels.
    # channel_limits = []
    # for i, ch in enumerate(channels_names):
    #     if ch in eeg_channels and common_eeg_limit is not None:
    #         ylim_val = common_eeg_limit
    #     else:
    #         p95 = np.percentile(np.abs(data[i, :]), 95)
    #         ylim_val = p95 if p95 != 0 else 100
    #     channel_limits.append(ylim_val)
    
    # Loop through each event to plot a segment.
    for idx, event in enumerate(ms_probes[:, 0]):
        # Define the segment window: 20 sec before to 10 sec after the event
        tmin_idx = event - 20 * int(sf)
        tmax_idx = event + 10 * int(sf)
        
        # Compute the common threshold for EEG channels from the event segment.
        eeg_segments = []
        for j, ch in enumerate(channels_names):
            if ch in eeg_channels:
                seg = data[j, tmin_idx:tmax_idx]
                eeg_segments.append(np.abs(seg))
        if eeg_segments:
            combined_eeg_event = np.concatenate(eeg_segments)
            common_eeg_limit_event = np.percentile(combined_eeg_event, 95)
            if common_eeg_limit_event == 0:
                common_eeg_limit_event = 100
        else:
            common_eeg_limit_event = 100

        # Create figure and axes.
        fig, ax = plt.subplots(nrows=data.shape[0], ncols=1, figsize=(16, 16))
        for j, ch in enumerate(channels_names):
            # Extract the segment for the current channel.
            seg = data[j, tmin_idx:tmax_idx]
            # Use the common threshold for EEG channels; for others, compute individually.
            if ch in eeg_channels:
                ylim_val = common_eeg_limit_event
            else:
                ylim_val = np.percentile(np.abs(seg), 95)
                if ylim_val == 0:
                    ylim_val = 100

            # Set the color for ECG and RESP channels.
            plot_color = 'royalblue' if ch in ['ECG', 'RESP'] else 'k'
            ax[j].plot(seg, linewidth=0.5, color=plot_color)
            ax[j].set_ylim([-ylim_val, ylim_val])
            ax[j].set_yticks(np.linspace(-ylim_val, ylim_val, 3))
            ax[j].text(-0.05, 0.5, ch, transform=ax[j].transAxes,
                       va='center', ha='right', fontsize=12, fontweight='bold')
            ax[j].vlines(x=20*int(sf), ymin=-ylim_val, ymax=ylim_val, color='r')
            ax[j].spines['top'].set_visible(False)
            ax[j].spines['right'].set_visible(False)
            ax[j].spines['bottom'].set_visible(False)
            ax[j].set_xlim([0, (tmax_idx - tmin_idx)])
        
        # Remove x-tick labels for all subplots except the bottom one.
        for j in range(len(ax) - 1):
            ax[j].set_xticks([])
        ax[-1].set_xticks(np.linspace(0, (tmax_idx - tmin_idx), 7))
        ax[-1].set_xticklabels(np.arange(-20, 15, 5))
        ax[-1].set_xlabel('Time before probe onset (s)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=1)
        
        # Generate a random unique code (using uuid4 for full randomness)
        this_code = uuid.uuid4().hex
        
        # Overlay the random code on the image (at the bottom right corner)
        fig.text(0.95, 0.05, f"Code: {this_code}", ha="right", va="bottom",
                 fontsize=14, color="red", alpha=0.8)
        
        # Define the save path including the random code in the filename and save the figure.
        this_savepath = os.path.join(path_scoring_coded, f"{this_code}.png")
        plt.savefig(this_savepath)
        plt.close(fig)
        
        big_dic["sub_id"].append(sub_id)
        big_dic["group"].append(subtype)
        big_dic["session"].append(session)
        big_dic["nprobe"].append(idx)
        big_dic["mindstate"].append(ms_answers[idx])
        big_dic["code"].append(this_code)
    
    
    
# %% Narco Hallu Script - Homogeneous Threshold

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

# channels_oi =  ['F3', 'C3', 'O1', 
#   'TP10', 'VEOG', 'HEOG', 'ECG','RESP']
channels_oi =  ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'TP9',
  'TP10', 'VEOG', 'HEOG', 'ECG','RESP']

col_oi = ["sub_id", "group", "session", "nprobe", "mindstate", "code"]
big_dic = {col:[ ] for col in col_oi}

# Define standard thresholds for each channel.
standard_thresholds = {
    'F3': 150,
    'F4': 150,
    'C3': 150,
    'C4': 150,
    'O1': 150,
    'O2': 150,
    'VEOG': 200,
    'HEOG': 100,
    'ECG': 500,
    'RESP': 100
    }

merge_dict = {
    101 : [65,66,68,69,70,71,72,73],
    100 : [67]
    }

for i, file_path in enumerate(files):
    #### [1] Import Data and Minimally Process it
    sub_id = f"{file_path.split('/sub_')[1][:6]}{file_path.split('SART')[1][:3]}"
    
    if (sub_id.startswith('HS')
        or sub_id.startswith('HI')
        or 'N1_001_PM' in sub_id):
        continue
    
    print(f"...Processing {sub_id}, file {i+1} / {len(files)}...")
    
    subtype = sub_id[:2]
    session = sub_id[-2:]
    
    et_filepath = glob(os.path.join(
        path_data, 'experiment', f'sub_{sub_id[:-3]}', "*.asc")
        )
    
    scoring_savepath = os.path.join(
        path_preproc, "epochs_scoring", f"{sub_id}_hypno_hallu.txt"
        )
    
    behav_paths = glob(os.path.join(
        path_data, "experiment", f"sub_{sub_id[:-3]}", "*.mat")
        )   
    
    #### Extract Behav Infos
    if len(behav_paths) < 1:
        print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
        # continue
    if session == "AM":
        behav_path = behav_paths[0]
        et_filepath = et_filepath[0]
    else:
        behav_path = behav_paths[1]
        et_filepath = et_filepath[1]
    mat = loadmat(behav_path)
    df_probe = pd.DataFrame(
        mat['probe_res'], 
        columns = probe_col)
    if any(df_probe.PQ1_respval.isna()):
        df_probe.PQ1_respval.replace(np.nan, 0, inplace=True)
        
    ms_answers = np.array(
        [ms_dic[value] for value in df_probe.PQ1_respval.values]
        )
    
    #### Handle EEG
    raw = cfg.load_and_preprocess_data(file_path)    
    raw.pick(channels_oi)
    raw.set_eeg_reference(ref_channels=['TP9', 'TP10'])
    raw.drop_channels(['TP9', 'TP10'])
    # raw.set_eeg_reference(ref_channels=['TP10'])
    # raw.drop_channels(['TP10'])
    raw.filter(.5, 40)
    raw.filter(1,10, picks=['VEOG', 'HEOG'])
    raw.filter(1,30, picks=['ECG'])
    raw.filter(1,30, picks=['RESP'])
    
    # channels_names = ['F3', 'C3', 'O1', 
    #                  'VEOG', 'HEOG', 'ECG','RESP']
    # channels_names = ['F3', 'F4', 'C3', 'C4', 'O1', 
    #                  'O2', 'VEOG', 'HEOG', 'ECG','RESP']
    channels_names = ['F3', 'F4', 'C3', 'C4', 'O1', 
                     'O2', 'ECG','RESP']
    
    sf = raw.info['sfreq']
    
    events, event_id = mne.events_from_annotations(raw)
    
    events_go, _ = cfg.handle_events(raw, merge_dict)
    events_trials = events_go[np.isin(events_go[:, 2], [100, 101])]
    
    answers_trials = events_go[events_go[:, 2] == 5]
    
    ms_probes =  np.stack(
        [event for i, event in enumerate(events[events[:, 2] == 128]) 
         if not i % 3])
    
    data = raw.get_data(units={'eeg': 'uV', 'eog': 'uV', 'ecg': 'uV'})
    
    #### Handle Eye Tracker
    et = mne.io.read_raw_eyelink(et_filepath, create_annotations=True)
    et_sf = et.info['sfreq']
    et.drop_channels('DIN')
    
    et_data = et.get_data()
    xpos = zscore(et_data[0, :] - np.nanmean(et_data[0, :], axis = 0), nan_policy='omit')
    ypos = zscore(et_data[1, :] - np.nanmean(et_data[1, :]), nan_policy='omit')
    pupil = zscore(et_data[2, :], nan_policy='omit')
    
    et_events, et_event_id = mne.events_from_annotations(et)
    probes_et_id = {k: v for k, v in et_event_id.items() if "_Q1" in k}
    probes_list_id = [v for v in probes_et_id.values()]
    temp_et_probes = []
    for probe_id in probes_list_id : 
        temp_et_probes.append(
            et_events[et_events[:, 2] == probe_id]
            )
    good_et_events = np.vstack(temp_et_probes)
    
    when_probes_et = np.sort(good_et_events[:, 0])
    
    if not len(ms_answers) == len(ms_probes):
        print(f"!!!\n{sub_id} : Careful, inconsistencies found between EEG and Behav\n!!!")
        continue
        
    # Define EEG channels for reference.
    eeg_channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    # eeg_channels = ['F3', 'C3', 'O1']
    
    # Loop through each event to plot a segment.
    for idx, event in enumerate(ms_probes[:, 0]):
        # Define the segment window: 20 sec before to 10 sec after the event
        tmin_idx = event - 20 * int(sf)
        tmax_idx = event + 10 * int(sf)
        
        et_tmin_idx = when_probes_et[idx] - 20 * int(et_sf)
        et_tmax_idx = when_probes_et[idx] + 10 * int(et_sf)
        
        # Create figure and axes.
        fig, ax = plt.subplots(nrows=data.shape[0], ncols=1, figsize=(20, 12))
        for j, ch in enumerate(channels_names):
            # Extract the segment for the current channel.
            seg = data[j, tmin_idx:tmax_idx]
            # Set the y-axis limits using the fixed threshold for each channel.
            ylim_val = standard_thresholds[ch]
            # Set the color for ECG and RESP channels.
            plot_color = 'royalblue' if ch in ['ECG', 'RESP'] else 'k'
            ax[j].plot(seg, linewidth=0.5, color=plot_color)
            ax[j].set_ylim([-ylim_val, ylim_val])
            ax[j].set_yticks(np.linspace(-ylim_val, ylim_val, 3))
            ax[j].text(-0.05, 0.5, ch, transform=ax[j].transAxes,
                       va='center', ha='right', fontsize=12, fontweight='bold')
            # Draw a vertical red line at probe onset (20 sec into the window).
            ax[j].vlines(x=20*int(sf), ymin=-ylim_val, ymax=ylim_val, color='r')
            ax[j].spines['top'].set_visible(False)
            ax[j].spines['right'].set_visible(False)
            ax[j].spines['bottom'].set_visible(False)
            ax[j].set_xlim([0, (tmax_idx - tmin_idx)])
        
        # Remove x-tick labels for all subplots except the bottom one.
        for j in range(len(ax) - 1):
            ax[j].set_xticks([])
        ax[-1].set_xticks(np.linspace(0, (tmax_idx - tmin_idx), 7))
        ax[-1].set_xticklabels(np.arange(-20, 15, 5))
        ax[-1].set_xlabel('Time before probe onset (s)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=1)
        
        digit_events = events_trials[
            (events_trials[:, 0] >= tmin_idx) 
            & (events_trials[:, 0] < tmax_idx)]
        answer_events = answers_trials[
            (answers_trials[:, 0] >= tmin_idx) 
            & (answers_trials[:, 0] < tmax_idx)]
    
        
        # Generate a random unique code (using uuid4 for full randomness)
        this_code = uuid.uuid4().hex
        
        # Overlay the random code on the image (at the bottom right corner)
        fig.text(0.98, 0.01, f"Code: {this_code}", ha="right", va="bottom",
                 fontsize=8, color="red", alpha=0.8)
        
        # Define the save path including the random code in the filename and save the figure.
        this_savepath = os.path.join(path_scoring_coded, f"{this_code}.png")
        plt.savefig(this_savepath)
        plt.close(fig)
        
        big_dic["sub_id"].append(sub_id)
        big_dic["group"].append(subtype)
        big_dic["session"].append(session)
        big_dic["nprobe"].append(idx)
        big_dic["mindstate"].append(ms_answers[idx])
        big_dic["code"].append(this_code)
    
df = pd.DataFrame.from_dict(big_dic)    
df.to_csv(os.path.join(path_scoring_coded, "code_correspondance.csv"))

# %% VF Generate Images : Only NT1

group_oi = "HS"

files = glob(os.path.join(path_data, 'experiment', f'*{group_oi}*' , '*SART*.vhdr'))[:10]

"""
Create figure with extra axes:
  - Top axis for event markers
  - Middle axes for each EEG channel (channels_names)
  - Subtle highlight when participant close their eyes
"""

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

# channels_oi includes some channels that will be dropped later (TP9 and TP10)
channels_oi =  ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'TP9',
                'TP10', 'VEOG', 'HEOG', 'ECG','RESP']

col_oi = ["sub_id", "group", "session", "nprobe", "mindstate", "code"]
big_dic = {col:[ ] for col in col_oi}

# Define standard thresholds for each channel.
standard_thresholds = {
    'F3': 100,
    'F4': 100,
    'C3': 100,
    'C4': 100,
    'O1': 100,
    'O2': 100,
    'EMG': 100,
    'VEOG': 200,
    'HEOG': 100,
    'ECG': 500,
    'RESP': 100
    }

merge_dict = {
    101 : [65,66,68,69,70,71,72,73],
    100 : [67]
    }

for i, file_path in enumerate(files):
    #### [1] Import Data and Minimally Process it
    sub_id = f"{file_path.split('/sub_')[1][:6]}{file_path.split('SART')[1][:3]}"
    
    if group_oi == "N1" :
        if (sub_id.startswith('HS')
            or sub_id.startswith('HI')
            or 'N1_001_PM' in sub_id):
            continue
    
    print(f"...Processing {sub_id}, file {i+1} / {len(files)}...")
    
    subtype = sub_id[:2]
    session = sub_id[-2:]
    
    et_filepath = glob(os.path.join(
        path_data, 'experiment', f'sub_{sub_id[:-3]}', "*.asc")
        )
    
    if len(et_filepath) < 1 :
        print(f"\nNo Eye Tracking Recordings found {sub_id}... Skipping for now...\n")
        continue
    
    behav_paths = glob(os.path.join(
        path_data, "experiment", f"sub_{sub_id[:-3]}", "*.mat")
        )   
    
    #### Extract Behav Infos
    if len(behav_paths) < 1:
        print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
        continue
    if session == "AM":
        behav_path = behav_paths[0]
        et_filepath = et_filepath[0]
    else:
        behav_path = behav_paths[1]
        et_filepath = et_filepath[1]
    mat = loadmat(behav_path)
    df_probe = pd.DataFrame(
        mat['probe_res'], 
        columns = probe_col)
    if any(df_probe.PQ1_respval.isna()):
        df_probe.PQ1_respval.replace(np.nan, 0, inplace=True)
        
    ms_answers = np.array(
        [ms_dic[value] for value in df_probe.PQ1_respval.values]
        )
    
    #### Handle EEG
    raw = cfg.load_and_preprocess_data(file_path)    
    raw.pick(channels_oi)
    mne.set_bipolar_reference(raw,
        "Fp1",
        "Fp2",
        ch_name="EMG",
        ch_info=None,
        drop_refs=True,
        copy=False,
        on_bad='warn',
        verbose=None
        )
    raw.set_channel_types({'EMG':'emg'})
    raw.set_eeg_reference(ref_channels=['TP9', 'TP10'])
    raw.drop_channels(['TP9', 'TP10'])
    raw.filter(.5, 40, picks = 'eeg')
    raw.filter(1,10, picks=['VEOG', 'HEOG'])
    raw.filter(1,30, picks=['ECG'])
    raw.filter(1,30, picks=['RESP'])
    raw.filter(10,100, picks=['EMG'])
    channels_names = [
        'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 
        'EMG', 'VEOG', 'HEOG', 'ECG','RESP'
        ]
    raw.pick(channels_names)
    sf = raw.info['sfreq']
    
    # Get the events, and modify them a bit
    events, event_id = mne.events_from_annotations(raw)
    events_go, _ = cfg.handle_events(raw, merge_dict)
    events_trials = events_go[np.isin(events_go[:, 2], [100, 101])]
    answers_trials = events_go[events_go[:, 2] == 5]
    
    ms_probes = np.stack(
        [event for i, event in enumerate(events[events[:, 2] == 128]) 
         if not i % 3])
    
    # In order to align ET and EEG
    start_blocks = events[events[:, 2] == 2][-4:]
    eeg_event_ref = start_blocks[0,0]
    
    # Get data for plotting
    data = raw.get_data(units={'eeg':'uV', 'eog':'uV', 'ecg':'uV', 'emg':'uV'})
    
    #### Handle Eye Tracker
    et = mne.io.read_raw_eyelink(et_filepath, create_annotations=True)
    et_sf = et.info['sfreq']
    et.drop_channels('DIN')
    
    et_data = et.get_data()
    pupil = et_data[2, :]
    del et_data
    pupil_z = zscore(pupil, nan_policy='omit')
    
    # Get the events
    et_events, et_event_id = mne.events_from_annotations(et)
    probes_et_id = {k: v for k, v in et_event_id.items() if "_Q1" in k}
    start_blocks_et = {k: v for k, v in et_event_id.items() 
                       if k in ['B1', 'B2', 'B3', 'B4']}
    sample_start_blocks_et = [et_events[:, 0][et_events[:, 2] == v][0] 
                              for k, v in start_blocks_et.items()]
    # In order to align ET and EEG
    et_event_ref = sample_start_blocks_et[0]
    
    diff_eeg_et_sec = eeg_event_ref/sf - et_event_ref/et_sf
    
    if diff_eeg_et_sec > 0: 
        diff_sample_eeg = int(np.round(diff_eeg_et_sec * sf))
        
        # Add 0 to pupil_recording to have same time than EEG
        empty_diff = np.zeros((int(np.round(diff_eeg_et_sec * et_sf))))
        long_pupil_z = np.concat((empty_diff, pupil_z))
    
    else : 
        amount_cut = int(np.abs(np.round(diff_eeg_et_sec * et_sf)))
        long_pupil_z = pupil_z[amount_cut:]        
    
    onset_blinks = et.annotations.onset[
        et.annotations.description == 'BAD_blink'
        ]
    onset_blinks_sample_eeg = np.round((onset_blinks + diff_eeg_et_sec) * sf)
    
    duration_blinks = et.annotations.duration[
        et.annotations.description == 'BAD_blink'
        ]
    duration_blinks_sample_eeg = np.round(duration_blinks * sf)    
    
    probes_list_id = [v for v in probes_et_id.values()]
    temp_et_probes = []
    for probe_id in probes_list_id : 
        temp_et_probes.append(
            et_events[et_events[:, 2] == probe_id]
            )
    good_et_events = np.vstack(temp_et_probes)
    
    when_probes_et = np.sort(good_et_events[:, 0])
    
    if not len(ms_answers) == len(ms_probes):
        print(f"!!!\n{sub_id} : Careful, inconsistencies found between EEG and Behav\n!!!")
        continue
        
    # Define EEG channels for reference.
    eeg_channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    
    # Loop through each event to plot a segment.
    for idx, event in enumerate(ms_probes[:, 0]):
        # Define the segment window: 20 sec before to 10 sec after the event
        tmin_idx = event - 20 * int(sf)
        tmax_idx = event + 10 * int(sf)
        n_eeg = tmax_idx - tmin_idx
        time_eeg = np.arange(n_eeg) / sf 
        
        # et_tmin_idx = when_probes_et[idx] - 20 * int(et_sf) 
        # et_tmax_idx = when_probes_et[idx] + 10 * int(et_sf)
        
        n_eeg_channels = len(channels_names)
        n_rows = n_eeg_channels + 2 # + Events + Pupil
        fig, ax = plt.subplots(nrows=n_rows, ncols=1, figsize=(20, 12), sharex=True)
        
        # -----------------------------------------
        # Top axis: Event Markers and Blink Highlight
        # -----------------------------------------
        ax_events = ax[0]
        ax_events.set_ylim([-1, 1])
        ax_events.set_yticks([])
        ax_events.text(-0.05, 0.5, "Events", transform=ax_events.transAxes,
                        va='center', ha='right', fontsize=12, fontweight='bold')
        ax_events.spines['top'].set_visible(False)
        ax_events.spines['right'].set_visible(False)
        ax_events.spines['bottom'].set_visible(False)
        ax_events.spines['left'].set_visible(False)
        # Mark the probe onset at 20 seconds (since the x-axis is now in seconds)
        ax_events.axvline(x=20, color='red', linewidth=2)
        
        # Highlight blink periods in the events axis (convert to seconds)
        onset_overlapping_blinks = onset_blinks_sample_eeg[
            (onset_blinks_sample_eeg >= tmin_idx) & (onset_blinks_sample_eeg < tmax_idx)
        ]
        idx_overlapping_blinks = np.where(np.isin(onset_blinks_sample_eeg, onset_overlapping_blinks))[0]
        duration_overlapping_blinks = duration_blinks_sample_eeg[idx_overlapping_blinks]
        for onset, duration in zip(onset_overlapping_blinks, duration_overlapping_blinks):
            rel_start = (onset - tmin_idx) / sf
            rel_end = (onset + duration - tmin_idx) / sf
            ax_events.axvspan(rel_start, rel_end, color='red', alpha=0.2)
        
        # Plot digit events and responses, converting to seconds:
        digit_events = events_trials[(events_trials[:, 0] >= tmin_idx) & (events_trials[:, 0] < tmax_idx)]
        for de in digit_events:
            event_time = (de[0] - tmin_idx) / sf  # Convert from samples to seconds
            if de[2] == 101:
                ax_events.scatter(event_time, 0.25, color='black', marker='o', s=50)
            elif de[2] == 100:
                ax_events.scatter(event_time, 0.25, color='red', marker='o', s=50)
        answer_events = answers_trials[(answers_trials[:, 0] >= tmin_idx) & (answers_trials[:, 0] < tmax_idx)]
        for ae in answer_events:
            event_time = (ae[0] - tmin_idx) / sf  # Convert from samples to seconds
            ax_events.scatter(event_time, -0.25, color='purple', marker='o', s=50)
        
        # -----------------------------------------
        # Top axis 2: Pupil
        # -----------------------------------------
        # Convert the EEG window (tmin_idx, tmax_idx in EEG samples) to pupil samples
        pupil_tmin = int((tmin_idx / sf) * et_sf)
        pupil_tmax = int((tmax_idx / sf) * et_sf)
        pupil_seg = long_pupil_z[pupil_tmin:pupil_tmax]
        
        n_pupil = pupil_tmax - pupil_tmin
        time_pupil = np.arange(n_pupil) / et_sf
        
        ax_pupil = ax[1]
        ax_pupil.plot(time_pupil, pupil_seg, color='purple', linewidth=1)
        ax_pupil.text(-0.05, 0.5, "Pupil", transform=ax_pupil.transAxes,
                        va='center', ha='right', fontsize=12, fontweight='bold')
        ax_pupil.set_xlim([0, 30])
        ax_pupil.axvline(x=20, color='red')
        ax_pupil.spines['top'].set_visible(False)
        ax_pupil.spines['right'].set_visible(False)
        ax_pupil.spines['bottom'].set_visible(False)
        # Optionally, adjust y-axis as needed
        ax_pupil.set_ylim(-1, 6)
        
        # -----------------------------------------
        # Middle Axes: EEG Channels
        # -----------------------------------------
        for j, ch in enumerate(channels_names):
            seg = data[j, tmin_idx:tmax_idx]
            ylim_val = standard_thresholds[ch]
            if ch == 'ECG':
                plot_color = 'red'
            elif ch in ['RESP']:
                plot_color = 'royalblue'
            elif ch in ['VEOG', 'HEOG']:
                plot_color = '#9c6644'
            elif ch in ['EMG']:
                plot_color = '#344e41'
            else:
                plot_color = 'k'            
            ax_eeg = ax[j+2]  # rows 1 to n_eeg_channels
            ax_eeg.plot(time_eeg, seg, linewidth=1, color=plot_color)
            ax_eeg.set_ylim([-ylim_val, ylim_val])
            ax_eeg.set_yticks(np.linspace(-ylim_val, ylim_val, 3))
            ax_eeg.text(-0.05, 0.5, ch, transform=ax_eeg.transAxes,
                       va='center', ha='right', fontsize=12, fontweight='bold') 
            ax_eeg.axvline(x=20, color='red')
            ax_eeg.spines['top'].set_visible(False)
            ax_eeg.spines['right'].set_visible(False)
            ax_eeg.spines['bottom'].set_visible(False)
            ax_eeg.set_xlim([0, 30])
            # Remove x-tick labels for all EEG axes.
            ax_eeg.set_xticks([])
        
        # -----------------------------------------
        # Finish Up: Save Figure
        # -----------------------------------------
        fig.tight_layout(pad=1)
        this_code = uuid.uuid4().hex
        fig.text(0.98, 0.01, f"Code: {this_code}", ha="right", va="bottom",
                 fontsize=8, color="red", alpha=0.8)
        this_savepath = os.path.join(path_scoring_coded, f"{this_code}.png")
        plt.savefig(this_savepath)
        plt.close(fig)
        
        big_dic["sub_id"].append(sub_id)
        big_dic["group"].append(subtype)
        big_dic["session"].append(session)
        big_dic["nprobe"].append(idx)
        big_dic["mindstate"].append(ms_answers[idx])
        big_dic["code"].append(this_code)
    
df = pd.DataFrame.from_dict(big_dic)    
df.to_csv(os.path.join(path_scoring_coded, f"code_correspondance_{group_oi}.csv"))
