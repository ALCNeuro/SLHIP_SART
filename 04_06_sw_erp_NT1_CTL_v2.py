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
font_path = personal_path + '/aptos.ttf'
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
subtype_palette = ["#8d99ae", "#d00000"]


# %%% Script

# Parameters
winERP = [0, 3]           # ERP window (in seconds)
winBaseline = [0, 5]    # Baseline window (in seconds)
t_zero = "start"    # "start", "neg_peak_pos", "pos_peak_pos"

cols = [
    "sub_id", "subtype", "channel", "daytime", "nblock", "nprobe",
    "mindstate", "voluntary", "sleepiness", 
    "time", "value"
    ]
bigdic = {c:[] for c in cols}

slope_range = [0.125, 2] # in uV/ms
positive_amp = 75 # in uV
amplitude_max = 150
subtypes = ['HS', 'N1']
this_df_savepath = os.path.join(wavesPath, "figs", "NT1_CTL", "erp_sw.csv")

redo = 1
times = np.linspace(0, 768, 768)

if os.path.exists(this_df_savepath) and not redo :
    bigdf = pd.read_csv(this_df_savepath)
else : 
    
    for i_f, file in enumerate(files):
        sub_id = file.split('es/')[-1].split('_epo')[0]
        subtype = sub_id[:2]
        if subtype == "HI" : continue
        
        daytime = sub_id[-2:]
        sub_id = sub_id[:-3]
        
        thisSubSWPath = os.path.join(
            wavesPath, "all_waves", f"{sub_id}_{daytime}.csv"
            )
        
        if not os.path.exists(thisSubSWPath) :
            print(f"[!!!] Error on {sub_id} - SW are missing")
            continue
        df = pd.read_csv(thisSubSWPath)
        del df['Unnamed: 0']
    
        df = df.loc[
            (df['PTP'] < amplitude_max) 
            & (df['pos_amp_peak'] < positive_amp)]
        df = df.loc[
            (df["pos_halfway_period"] <= slope_range[1])
            & (df["pos_halfway_period"] >= slope_range[0])
            ]
        
        thresholds_90 = {}
        for c in channels:
            try :
                thresholds_90[c] = np.percentile(df.PTP.loc[df.channel==c], 90)
            except :
                print(f"No SW in {c}")
                thresholds_90[c] = np.nan
        
        df_sw = pd.concat([
            df.loc[(df.channel == c) 
                   & (df.PTP >= thresholds_90[c])] for c in channels
            ])
        
        epochs = mne.read_epochs(file)
        epochs.pick('eeg')
        epochs.filter(0.5, 30)
         
        sf = epochs.info["sfreq"]
        data = epochs.copy().get_data(units = dict(eeg = 'µV'))
        nepochs, nchans, nsamples = data.shape
        winERP_samples = int((winERP[1] - winERP[0]) * sf)
        # this_erp = np.nan * np.empty((nchans, winERP_samples))
                
        for ch, channel in enumerate(channels) :
            print(f"...Processing {channel}")
            temp_erp = []
            for n_epoch in range(nepochs) :
                thisChan = data[n_epoch, ch, :]
                epoch_chan_sw = df_sw.loc[
                    (df_sw['n_epoch'] == n_epoch)
                    & (df_sw['channel'] == epochs.ch_names[ch])]
                
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
                    erp_times = np.linspace(winERP[0], winERP[1], winERP_samples)
                    
                    
                    this_mindstate = epoch_chan_sw.mindstate.iloc[0]
                    this_sleepiness = epoch_chan_sw.sleepiness.iloc[0]
                    this_voluntary = epoch_chan_sw.voluntary.iloc[0]
                    this_block = epoch_chan_sw.nblock.iloc[0]

                    for t, value in enumerate(thisSW):
                        bigdic["sub_id"].append(sub_id)
                        bigdic["subtype"].append(subtype)
                        bigdic["channel"].append(channel)
                        bigdic["daytime"].append(daytime)
                        bigdic["nblock"].append(this_block)
                        bigdic["nprobe"].append(n_epoch)
                        bigdic["mindstate"].append(this_mindstate)
                        bigdic["voluntary"].append(this_voluntary)
                        bigdic["sleepiness"].append(this_sleepiness)
                        bigdic["time"].append(erp_times[t])
                        bigdic["value"].append(value)
                                  
df = pd.DataFrame.from_dict(bigdic)            
                
# %% DF Manip

this_df = df.copy().loc[df.channel == "Pz"].drop(columns = ["channel","daytime", "nblock", "nprobe", "mindstate"]).groupby(
    ["sub_id", "subtype", "time"], as_index=False
    ).mean()

# %% Plot ERP

fig, ax = plt.subplots(figsize=(2.5, 2.5))

sns.lineplot(
    data = this_df,
    x = "time",
    y = "value",
    hue = "subtype",
    ax=ax,
    palette=subtype_palette,
    legend=None
    )
plt.axvline(x=1, color='k', linestyle='--', alpha = .4, linewidth = 1)
plt.axhline(y=0, color='k', linestyle='--', alpha = .4, linewidth = 1)
ax.set_ylim(-20, 10)
ax.set_xlim(0, 3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Time from start of SW (s)", fontsize=10, font = bold_font)
ax.set_xticks(
    ticks = np.arange(0, 3.5, .5), 
    labels = np.round(np.arange(-1, 2.5, .5),1), 
    fontsize=5, 
    font = font
    )
ax.set_ylabel("Evoked Voltage (µV)", fontsize=10, font = bold_font)
ax.set_yticks(
    ticks = np.arange(-20, 15, 5), 
    labels = np.round(np.arange(-20, 15, 5)), 
    fontsize=5, 
    font = font
    )
sns.despine()
fig.tight_layout()
plt.savefig(
    os.path.join(wavesPath, "figs", "ERP_SW_Pz.png"), dpi = 300
    )

# %% 

    