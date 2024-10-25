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

thisErpFile = os.path.join(wavesPath, "figs", "dic_erp_sw.pkl")
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
            
# %% ERP [Fz, Cz, Pz, Oz]
 
times = np.linspace(0, 500, 500)
palette = ["#FFBA08", "#F48C06", "#DC2F02", "#9D0208", "#370617"]

fig, ax = plt.subplots(
    nrows = 4, 
    ncols = 2,
    sharex = True,
    sharey = True,
    layout = 'tight',
    figsize = (10, 20)
    )

for i_cond, condition in enumerate(['EASY', 'HARD']):
    for i_chan, channel in enumerate([1, 23, 12, 16]) :
        for i_sess, n_sess in enumerate([1, 2, 3, 4, 5]):
            ax[i_chan][i_cond].plot(
                times, 
                np.nanmean(
                    big_dic[condition][n_sess], axis = 0
                    )[channel],
                label = f"Session {n_sess}",
                color = palette[i_sess]
                )
            ax[i_chan][i_cond].set_xticks(
                np.linspace(0, 500, 6),
                np.round(np.linspace(-1, 1, 6), 2)
                )
            ax[i_chan][i_cond].spines['right'].set_visible(False)
            ax[i_chan][i_cond].spines['top'].set_visible(False)    
            ax[3][i_cond].set_xlabel("Time (s)")        
            ax[i_chan][0].set_ylabel("Voltage (µV)")        
            ax[i_chan][i_cond].set_title(
                f"{epochs.ch_names[channel]} - {condition}"
                )
            
    fig.suptitle("ERP of average SW, across channel for each sessions")
    ax[0][1].legend(bbox_to_anchor=(1.1, 1.05))

plt.savefig(os.path.join(
    swDataPath, "Figs", "ERP_SW_Midline.png"
    ), dpi = 300)
 
# %% 