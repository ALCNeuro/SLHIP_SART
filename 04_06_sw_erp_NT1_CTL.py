#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/09/2024

author: Arthur_LC

04_06_sw_erp.py

I'm working with epoched data
So I can't use MNE Structure to add anotations and plot from there

"""

# %%% Paths & Packages

import SLHIP_config_ALC as config
import mne
from scipy.stats import sem
from glob import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

from matplotlib.font_manager import FontProperties
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

cleanDataPath = config.cleanDataPath
wavesPath = config.wavesPath

slowwaves_path = os.path.join(wavesPath, "slow_waves")
slowwaves_files = glob(os.path.join(slowwaves_path, "slow_waves*.csv"))
reports_path = os.path.join(wavesPath, "reports")
figs_path = os.path.join(wavesPath, "figs")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

channels = np.array(config.eeg_channels)

files = glob(os.path.join(cleanDataPath, "epochs_probes", "*epo.fif"))

# %%% Script

# Parameters
winERP = [-1, 2]           # ERP window (in seconds)
winBaseline = [-1, -.5]    # Baseline window (in seconds)
t_zero = "start"    # "start", "neg_peak_pos", "pos_peak_pos"

slope_range = [0.125, 2] # in uV/ms
positive_amp = 75 # in uV
amplitude_max = 150
subtypes = ['HS', 'N1']
thisErpFile = os.path.join(wavesPath, "figs", "NT1_CTL", "dic_erp_sw.pkl")

redo = 1

if os.path.exists(thisErpFile) and not redo :
    with open(thisErpFile, 'rb') as f:
        big_dic = pickle.load(f)
else : 
    big_dic = {
        "HS" : {
            "sub_id" : [],
            "erp" : [],
             },
        "N1" : {
            "sub_id" : [],
            "erp" : [],
             }
        }
    
    for i_f, file in enumerate(files):
        sub_id = file.split('es/')[-1].split('_epo')[0]
        subtype = sub_id[:2]
        if subtype == "HI" : continue
        
        thisSubSWPath = os.path.join(
            wavesPath, "all_waves", f"{sub_id}.csv"
            )
        
        if not os.path.exists(thisSubSWPath) :
            print(f"[!!!] Error on {sub_id} - SW are missing")
            continue
        df = pd.read_csv(thisSubSWPath)
        del df['Unnamed: 0']
    
        df = df.loc[
            (df['PTP'] < amplitude_max) 
            & (df['PTP'] > 20)
            & (df['pos_amp_peak'] < positive_amp)]
        df = df.loc[
            (df["pos_halfway_period"] <= slope_range[1])
            & (df["pos_halfway_period"] >= slope_range[0])
            ]
        
        epochs = mne.read_epochs(file)
        epochs.pick('eeg')
         
        sf = epochs.info["sfreq"]
        data = epochs.copy().get_data(units = dict(eeg = 'µV'))
        nepochs, nchans, nsamples = data.shape
        winERP_samples = int((winERP[1] - winERP[0]) * sf)
        this_erp = np.empty((nchans, winERP_samples))
    
        for ch in range(nchans) :
            print(f"...Processing {epochs.ch_names[ch]}")
            temp_erp = []
            for n_epoch in range(nepochs) :
                thisChan = data[n_epoch, ch, :]
                epoch_chan_sw = df.loc[
                    (df['n_epoch'] == n_epoch)
                    & (df['channel'] == epochs.ch_names[ch])]
                
                for _, row in epoch_chan_sw.iterrows():
                    start = row[t_zero]
                    start_idx = int(start + winERP[0]*sf)
                    end_idx = int(start + winERP[1]*sf)

                    # Create a NaN-filled array of ERP length
                    thisSW = np.full((winERP_samples,), np.nan)

                    # Determine valid signal indices
                    valid_start = max(0, start_idx)
                    valid_end = min(nsamples, end_idx)
                    if valid_end <= valid_start:
                        continue  # skip if invalid slice

                    # Extract slice from signal
                    valid_slice = thisChan[valid_start:valid_end]

                    # Calculate where to insert into padded ERP array
                    insert_start = valid_start - start_idx
                    insert_end = insert_start + len(valid_slice)

                    # Attempt baseline correction
                    baseline_start = int(start + winBaseline[0]*sf)
                    baseline_end = int(start + winBaseline[1]*sf)
                    if baseline_start >= 0 and baseline_end <= nsamples:
                        baseline = np.mean(thisChan[baseline_start:baseline_end])
                        valid_slice = valid_slice - baseline  # baseline correction

                    thisSW[insert_start:insert_end] = valid_slice
                    temp_erp.append(thisSW)
            this_erp[ch, :] = np.nanmean(temp_erp, axis=0) if temp_erp else np.full((winERP_samples,), np.nan)
        
        big_dic[subtype]['erp'].append(this_erp)
        big_dic[subtype]["sub_id"].append(sub_id)
    
    with open(thisErpFile, 'wb') as f:
        pickle.dump(big_dic, f)                
                
# %% Clean Plot

ch_name = "Cz"
palette = ["#8d99ae", "#d00000"]

times = np.linspace(0, 768, 768)
fig, ax = plt.subplots(
    nrows=1, 
    ncols=1,
    sharex=True,
    sharey=True,
    layout='tight',
    figsize=(4, 4)
    )
# Loop over conditions
for i_st, subtype in enumerate(['HS', 'N1']):
    erp_list = big_dic[subtype]["erp"]

    # Collect ERPs from the specified channel
    subject_erp_ch = []
    for subject_erp in erp_list:
        try:
            ch_index = config.eeg_channels.index(ch_name)
            subject_erp_ch.append(subject_erp[ch_index, :])
        except ValueError:
            print(f"⚠️ {ch_name} not found in config.eeg_channels.")
            continue

    if len(subject_erp_ch) == 0:
        continue

    subject_erp_ch = np.stack(subject_erp_ch, axis=0)  # shape: (n_subjects, n_times)
    group_mean = np.nanmean(subject_erp_ch, axis=0)
    group_sem = sem(subject_erp_ch, axis=0, nan_policy='omit')

    # Plot
    ax.plot(
        times,
        group_mean,
        color=palette[i_st],
        linewidth=2,
        alpha=0.7
        )
    ax.fill_between(
        times,
        group_mean - group_sem,
        group_mean + group_sem,
        color=palette[i_st],
        alpha=0.3
        )   

    # Set ticks, limits and labels
    ax.set_xticks(
        np.linspace(0, 768, 5),
        np.round(np.linspace(-1, 1, 5), 2),
        fontsize=11
        )
    ax.set_yticks(
        np.linspace(-15, 5, 5),
        np.linspace(-15, 5, 5),
        fontsize=11
        )
    ax.set_ylim(-15, 7.5)
    # ax.set_xlim(125, 500)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Time from start of SW (s)", fontsize=15)
    ax.set_ylabel("Evoked Voltage (µV)", fontsize=15)
    fig.tight_layout(pad=2)

# %% ERP - average - sessions
# thisErpFile = os.path.join(swDataPath, "dic_erp_sw.pkl")
# big_dic = pickle.load(thisErpFile)

subtypes = ["HS", "N1"]

times = np.linspace(0, 512, 512)
palette = ["#8d99ae", "#d00000", "#ffb703"]

fig, ax = plt.subplots(
    nrows = 1, 
    ncols = 1,
    sharex = True,
    sharey = True,
    layout = 'tight',
    figsize = (4, 4)
    )
for i_s, subtype in enumerate(subtypes):
    ax.vlines(
        250, 
        ymin = -15, ymax = 5, 
        color = 'lightgrey', ls = 'dotted',
        lw = 3, alpha = .4
        )
    ax.hlines(
        0, 
        xmin = -.25, xmax = 525, 
        color = 'lightgrey', ls = 'dotted',
        lw = 3, alpha = .4
        )
    ax.plot(
        times, 
        lowess(np.nanmean(np.nanmean(
            big_dic[subtype]['erp'], axis = 0
            ), axis = 0), times, .075)[:, 1],
        label = subtype,
        color = palette[i_s],
        linewidth = 3, 
        alpha = .95
        )
    ax.set_xticks(
        np.linspace(0, 500, 5),
        np.round(np.linspace(-1, 1, 5), 2),
        font = font,
        fontsize = 16
        )
    ax.set_yticks(
        np.linspace(-15, 5, 5),
        np.round(np.linspace(-15, 5, 5), 2),
        font = font,
        fontsize = 16
        )
    ax.set_ylim(-13, 5)
    ax.set_xlim(0, 512)
    sns.despine()
    ax.set_xlabel("Time (s)", font = bold_font, fontsize = 24)        
    ax.set_ylabel("ERP (µV)", font = bold_font, fontsize = 24)        

plt.savefig(os.path.join(
    wavesPath, "figs", "ERP_SW_AverageChan.png"
    ), dpi = 300)
            
# %% 

    