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
from glob import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

from highlight_text import fig_text
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

# Each Session's list will contain an np.ndarray of the shape :
#   nch, nsample  
#   64, 2*256
#   containing the mean sw erp for each subject at this session
slope_range = [0.125, 2] # in uV/ms
positive_amp = 75 # in uV
amplitude_max = 150

winERP = [-1, 1]
winBaseline = [-1, -.5]
subtypes = ['HS', 'N1', 'HI']


thisErpFile = os.path.join(wavesPath, "figs", "dic_erp_sw.pkl")

if os.path.exists(thisErpFile) :
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
             },
        "HI" : {
            "sub_id" : [],
            "erp" : [],
             },
        }
    
    for i_f, file in enumerate(files):
        sub_id = file.split('es/')[-1].split('_epo')[0]
        subtype = sub_id[:2]
        
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
        this_erp = np.empty((len(channels), int(2*sf)))
    
        for ch in range(nchans) :
            print(f"...Processing {epochs.ch_names[ch]}")
            temp_erp = []
            for n_epoch in range(nepochs) :
                thisChan = data[n_epoch, ch, :]
                epoch_chan_sw = df.loc[
                    (df['n_epoch'] == n_epoch)
                    & (df['channel'] == epochs.ch_names[ch])]
                
                n_sw, _ = epoch_chan_sw.shape
                if n_sw == 0 :
                    continue
                
                for i_start, start in enumerate(epoch_chan_sw.start):
                    if (start + winERP[0] * sf < 0 
                        or start + winERP[1] * sf > nsamples) : 
                        print("SW don't fit the window, skipping")
                        continue
                    thisSW = thisChan[
                        int(start + winERP[0]*sf)
                        : int(start + winERP[1]*sf)
                        ] - np.mean(thisChan[
                            int(start + winBaseline[0]*sf)
                            : int(start + winBaseline[1]*sf)
                            ])
                    temp_erp.append(thisSW)
            this_erp[ch, :] = np.nanmean(temp_erp, axis = 0)
        
        big_dic[subtype]['erp'].append(this_erp)
        big_dic[subtype]["sub_id"].append(sub_id)
    
    with open(thisErpFile, 'wb') as f:
        pickle.dump(big_dic, f)                
                
# %% ERP - average - sessions
# thisErpFile = os.path.join(swDataPath, "dic_erp_sw.pkl")
# big_dic = pickle.load(thisErpFile)

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

# %% B - ERP 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# Load data
thisErpFile = os.path.join(swDataPath, "dic_erp_sw_new.pkl")
big_dic = pd.read_pickle(thisErpFile)

# Define parameters
times = np.linspace(0, 500, 500)
sessions = [1, 2, 3, 4, 5]
conditions = ['EASY', 'HARD']

fig, axes = plt.subplots(
    nrows=1, 
    ncols=5, 
    figsize=(20, 4), 
    sharey=True, 
    layout='tight'
)

for i_sess, sess in enumerate(sessions):
    ax = axes[i_sess]
    
    for i_cond, condition in enumerate(conditions):
        # Compute ERP for each subject (average across channels)
        erp_subject = np.nanmean(big_dic[condition][sess], axis=1)  # shape: (n_subjects, n_times)
        
        # Group-level mean and SEM
        group_mean = np.nanmean(erp_subject, axis=0)
        group_sem = sem(erp_subject, axis=0, nan_policy='omit')

        # Plot ERP and SEM
        ax.plot(
            times,
            group_mean,
            color=palette_hardeasy[i_cond],
            linewidth=2,
            alpha=0.7,
            label=condition if i_sess == 0 else ""
        )
        ax.fill_between(
            times,
            group_mean - group_sem,
            group_mean + group_sem,
            color=palette_hardeasy[i_cond],
            alpha=0.3
        )

    # Plot guidelines
    ax.vlines(250, ymin=-12.5, ymax=7.5, color='k', ls='dashed', lw=1, alpha=.1)
    ax.hlines(0, xmin=0, xmax=500, color='k', ls='dashed', lw=1, alpha=.1)
    
    # Format subplot
    ax.set_xlim(0, 500)
    ax.set_xticks(
        np.linspace(0, 500, 5),
        np.round(np.linspace(-1, 1, 5), 2),
        fontsize=11
        )
    ax.set_ylim(-15, 7.5)
    ax.set_title(f"Session {sess}", fontsize=12)
    ax.set_xlabel("Time (ms)", fontsize=11)
    if i_sess == 0:
        ax.set_ylabel("ERP (µV)", fontsize=11)

# Add legend only to first subplot
axes[0].legend()

# Save and show
plt.savefig(os.path.join(swDataPath, "Figs", "ERP_SW_per_session.png"), dpi=300)
plt.show()

# %% B - ERP Midline by Channel and Session

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

t_zero = "neg_peak_pos"    # "start", "neg_peak_pos", "pos_peak_pos"

# Load ERP dictionary
thisErpFile = os.path.join(swDataPath, f"dic_erp_sw_longer_{t_zero}.pkl")
big_dic = pd.read_pickle(thisErpFile)

# Parameters
times = np.linspace(0, 750, 750)
sessions = [1, 2, 3, 4, 5]
conditions = ['EASY', 'HARD']
midline_channels = ["Fz", "Cz", "Pz", "Oz"]
n_rows = len(midline_channels)
n_cols = len(sessions)

# Set up plot grid
fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(n_cols * 4, n_rows * 3),
    sharex=True,
    sharey=True,
    layout='tight'
)

for i_row, ch_name in enumerate(midline_channels):
    for i_col, sess in enumerate(sessions):
        ax = axes[i_row, i_col] if n_rows > 1 else axes[i_col]

        for i_cond, condition in enumerate(conditions):
            erp_list = big_dic[condition][sess]

            # Collect ERPs from the specified channel
            subject_erp_ch = []
            for subject_erp in erp_list:
                try:
                    ch_index = config.channels.index(ch_name)
                    subject_erp_ch.append(subject_erp[ch_index, :])
                except ValueError:
                    print(f"⚠️ {ch_name} not found in config.channels.")
                    continue
                except IndexError:
                    print(f"⚠️ ERP shape mismatch for subject in session {sess}.")
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
                color=palette_hardeasy[i_cond],
                linewidth=2,
                alpha=0.7,
                label=condition if (i_row == 0 and i_col == 0) else ""
            )
            ax.fill_between(
                times,
                group_mean - group_sem,
                group_mean + group_sem,
                color=palette_hardeasy[i_cond],
                alpha=0.3
            )

        # Axes formatting
        # ax.vlines(250, ymin=-20, ymax=10, color='k', ls='dashed', lw=1, alpha=0.1)
        # ax.hlines(0, xmin=0, xmax=750, color='k', ls='dashed', lw=1, alpha=0.1)
        ax.set_xlim(0, 750)
        ax.set_ylim(-35, 15)
        ax.set_xticks(
            np.linspace(0, 750, 7),
            np.round(np.linspace(-1, 2, 7), 2),
            fontsize=11
            )

        if i_row == n_rows - 1:
            ax.set_xlabel("Time (ms)", fontsize=11)
        if i_col == 0:
            ax.set_ylabel(f"{ch_name}\nERP (µV)", fontsize=11)
        if i_row == 0:
            ax.set_title(f"Session {sess}", fontsize=12)

# Add legend only to top-left subplot
axes[0, 0].legend()

# Save and display
plt.savefig(os.path.join(swDataPath, "Figs", f"ERP_SW_long_midline_grid_{t_zero}.png"), dpi=300)
plt.show()
