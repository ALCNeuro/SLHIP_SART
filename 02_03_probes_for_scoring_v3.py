#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:51:27 2024

@author: arthurlecoz

02_03_probes_for_scoring.py
******************************************************************************

Visu : 
- EEG 
- EMG 
- EOG
- ECG
- Axe de temps
- Emphase sur les 5 dernières secondes [faire apparaître les délimitations par 20s?]

Avoir un PPT par participant :
D’abord le resting state (homogénéiser à 5’ par sujet)
	—> Avec 2 slides par probes : 
		—> 30s - 20s before probe - 10s après probe
        
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
# matplotlib.use('QtAgg')

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

# %% Let's generate random IDS first

this_identifiers = os.path.join(path_scoring_coded, 'identifiers.csv')

if os.path.exists(this_identifiers) :
    df_id = pd.read_csv(this_identifiers)
    print("Identifers loaded successfully.")
else : 
    import random
    import string
    these_ids = np.array([sub_id[4:] for sub_id in sub_ids])
    # Function to generate a random alphanumeric string of a given length
    def random_string(length=6):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=length))
    
    df_id = pd.DataFrame({'sub_id': these_ids})
    df_id['random_id'] = [random_string(6) for _ in range(len(sub_ids))]
    df_id.to_csv(this_identifiers, index=False)
    print("CSV file 'identifiers.csv' generated and saved successfully!")
 
# %% VF Generate Images per Subject : Probes
matplotlib.use('Agg')
group_oi = "N1"

files = glob(os.path.join(
    path_data, 'experiment', f'*{group_oi}*' , '*SART*.vhdr')
    )

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
                'TP10', 'VEOG', 'HEOG', 'ECG']

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
    }

dic_when = {0 : "before", 1 : "during"}

for i, file_path in enumerate(files):
    #### [1] Import Data and Minimally Process it
    sub_id = f"{file_path.split('/sub_')[1][:6]}{file_path.split('SART')[1][:3]}"
    this_subid_code = df_id.random_id.loc[
        df_id.sub_id == sub_id[:-3]
        ].iloc[0]
    
    if group_oi == "N1" :
        if (sub_id.startswith('HS')
            or sub_id.startswith('HI')
            or 'N1_001_PM' in sub_id):
            continue
    
    print(f"...Processing {sub_id}, file {i+1} / {len(files)}...")
    
    subtype = sub_id[:2]
    session = sub_id[-2:]
    
    behav_paths = glob(os.path.join(
        path_data, "experiment", f"sub_{sub_id[:-3]}", "*.mat")
        )   
    
    #### Extract Behav Infos
    if len(behav_paths) < 1:
        print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
        continue
    if session == "AM":
        behav_path = behav_paths[0]
    else:
        behav_path = behav_paths[1]
        
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
    raw.filter(10,100, picks=['EMG'])
    channels_names = [
        'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 
        'EMG', 'VEOG', 'HEOG', 'ECG'
        ]
    raw.pick(channels_names)
    sf = raw.info['sfreq']
    
    # Get the events
    events, event_id = mne.events_from_annotations(raw)
    ms_probes = events[events[:, 2] == 3]
    
    # Get data for plotting
    data = raw.get_data(units={'eeg':'uV', 'eog':'uV', 'ecg':'uV', 'emg':'uV'})
    
    if not len(ms_answers) == len(ms_probes):
        print(f"!!!\n{sub_id} : Careful, inconsistencies found between EEG and Behav\n!!!")
        continue
        
    # Define EEG channels for reference.
    eeg_channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    
    # Loop through each event to plot a segment.
    for idx, event in enumerate(ms_probes[:, 0]):
        for t_idx, [tmin, tmax] in enumerate([[50, -20], [20, 10]]) : 
            
            # Define the segment window: [-50, -20 or -20, +10]
            tmin_idx = event - tmin * int(sf)
            tmax_idx = event + tmax * int(sf)
            n_eeg = tmax_idx - tmin_idx
            time_eeg = np.arange(n_eeg) / sf 
            n_eeg_channels = len(channels_names)
            n_rows = n_eeg_channels 
            fig, ax = plt.subplots(nrows=n_rows, ncols=1, figsize=(20, 12), sharex=True)
            
            for j, ch in enumerate(channels_names):
                seg = data[j, tmin_idx:tmax_idx]
                ylim_val = standard_thresholds[ch]
                if ch == 'ECG':
                    plot_color = 'royalblue'
                elif ch in ['VEOG', 'HEOG']:
                    plot_color = '#9c6644'
                elif ch in ['EMG']:
                    plot_color = '#344e41'
                else:
                    plot_color = 'k'            
                ax_eeg = ax[j]
                ax_eeg.plot(time_eeg, seg, linewidth=1, color=plot_color)
                ax_eeg.set_ylim([-ylim_val, ylim_val])
                ax_eeg.set_yticks(np.linspace(-ylim_val, ylim_val, 3))
                ax_eeg.text(
                    -0.05, 0.5, 
                    ch, 
                    transform=ax_eeg.transAxes,
                    va='center', 
                    ha='right', 
                    fontsize=12, 
                    fontweight='bold'
                    ) 
                if t_idx :
                    ax_eeg.axvline(x=20, color='red')
                    
                ax_eeg.spines['top'].set_visible(False)
                ax_eeg.spines['right'].set_visible(False)
                ax_eeg.spines['bottom'].set_visible(False)
                
                ax_eeg.set_xticks(np.arange(0, 31, 5))
                ax_eeg.grid(
                    axis='x', 
                    linestyle='--', 
                    linewidth=0.5, 
                    color='gray', 
                    alpha=0.7
                    )
                ax_eeg.set_xlim([0, 30])
            ax[-1].set_xlabel("Time (s)")
            
            # -----------------------------------------
            # Finish Up: Save Figure
            # -----------------------------------------
            fig.tight_layout(pad=1)
            fig.text(
                0.98, 
                0.01, 
                f"Code: {this_subid_code}_session_{session}_{dic_when[t_idx]}_probe_{idx}", 
                ha="right", 
                va="bottom",
                fontsize=8, 
                color="red", 
                alpha=0.8
                )
            this_savepath = os.path.join(
                path_scoring_coded, 
                f"{this_subid_code}_session_{session}_{dic_when[t_idx]}_probe_{idx}.png"
                )
            plt.savefig(this_savepath)
            plt.close(fig)
    
# df = pd.DataFrame.from_dict(big_dic)    
# df.to_csv(os.path.join(path_scoring_coded, f"code_correspondance_{group_oi}.csv"))

# %% Generate Images per subject : Resting States

group_oi = "N1"
sub_ids = np.unique(np.array(
    [file.split('experiment/')[1].split('/')[0] for file 
     in glob(os.path.join(path_data, "experiment", f"*{group_oi}*", "*.mat"))]
    ))
sub_ids = np.array([sub_id[4:] for sub_id in sub_ids])

for i, sub_id in enumerate(sub_ids):
    this_subid_code = df_id.random_id.loc[df_id.sub_id == sub_id].iloc[0]
    for session in ["AM", "PM"] :
        for rs in [1, 2] :
            
            rs_filepath = glob(
                os.path.join(
                    path_data, 
                    'experiment', 
                    f'sub_{sub_id}', 
                    f'*RS{rs}*{session}.vhdr'
                    )
                )
        
            if len(rs_filepath) < 1 :
                print(f"For {sub_id} - {session} - RS n°{rs} is missing.. Please inspect manually.")
                continue
        
            rs_filepath = rs_filepath[0]
            
            raw = cfg.load_and_preprocess_data(rs_filepath)
            raw.crop(tmin=0, tmax=300, include_tmax=True)
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
            raw.filter(10,100, picks=['EMG'])
            channels_names = [
                'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 
                'EMG', 'VEOG', 'HEOG', 'ECG'
                ]
            raw.pick(channels_names)
            sf = raw.info['sfreq']
            
            windows = np.arange(0, 300, 30)
            
            data = raw.get_data(units={'eeg':'uV', 'eog':'uV', 'ecg':'uV', 'emg':'uV'})
            
            for i, start in enumerate(windows):
                tmin = int(start * sf)
                tmax = int((start + 30) * sf)
                
                n_eeg = tmax - tmin
                time_eeg = np.arange(n_eeg) / sf 
                n_eeg_channels = len(channels_names)
                n_rows = n_eeg_channels 
                fig, ax = plt.subplots(nrows=n_rows, ncols=1, figsize=(20, 12), sharex=True)
                
                # -----------------------------------------
                # Middle Axes: EEG Channels
                # -----------------------------------------
                for j, ch in enumerate(channels_names):
                    seg = data[j, tmin:tmax]
                    ylim_val = standard_thresholds[ch]
                    if ch == 'ECG':
                        plot_color = 'royalblue'
                    elif ch in ['VEOG', 'HEOG']:
                        plot_color = '#9c6644'
                    elif ch in ['EMG']:
                        plot_color = '#344e41'
                    else:
                        plot_color = 'k'            
                    ax_eeg = ax[j]
                    ax_eeg.plot(time_eeg, seg, linewidth=1, color=plot_color)
                    ax_eeg.set_ylim([-ylim_val, ylim_val])
                    ax_eeg.set_yticks(np.linspace(-ylim_val, ylim_val, 3))
                    ax_eeg.text(-0.05, 0.5, ch, transform=ax_eeg.transAxes,
                               va='center', ha='right', fontsize=12, fontweight='bold') 

                    ax_eeg.spines['top'].set_visible(False)
                    ax_eeg.spines['right'].set_visible(False)
                    ax_eeg.spines['bottom'].set_visible(False)
                    
                    ax_eeg.set_xticks(np.arange(0, 31, 5))
                    ax_eeg.grid(axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
                    ax_eeg.set_xlim([0, 30])
                ax[-1].set_xlabel("Time (s)")
                
                # -----------------------------------------
                # Finish Up: Save Figure
                # -----------------------------------------
                fig.tight_layout(pad=1)
                fig.text(0.98, 0.01, f"Code: {this_subid_code}_session_{session}_RS_{rs}_w_{i}", ha="right", va="bottom",
                         fontsize=8, color="red", alpha=0.8)
                this_savepath = os.path.join(path_scoring_coded, f"{this_subid_code}_session_{session}_RS_{rs}_w_{i}.png")
                plt.savefig(this_savepath)
                plt.close(fig)
                
