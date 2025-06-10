#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:26:16 2025

@author: arthurlecoz

06_04_yasa_gethypno.py
"""
# %% Paths & Packages

import mne 
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from scipy.io import loadmat

from matplotlib.font_manager import FontProperties

# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

#### Paths
if 'arthur' in os.getcwd():
    path_root='/Volumes/DDE_ALC/PhD/SLHIP'
else:
    path_root='your_path'

path_data=os.path.join(path_root, '00_Raw')
path_edfusleep=os.path.join(path_root, "01_Preproc", "edfs_usleep")
path_usleep=os.path.join(path_root, '06_USleep')

files = glob(os.path.join(path_usleep, 'yasa', "*hypnodensity.csv"))
eeg_files = glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))

ms_dic = {
    0 : "MISS",
    1 : 'ON',
    2 : 'MW',
    3 : 'DISTRACTED',
    4 : 'HALLU',
    5 : 'MB',
    6 : 'FORGOT'
    }

probe_col = [
    "nprobe","t_probe_th","t_probe_act","nblock","block_cond","ntrial",
    "PQ1_respkey","PQ2_respkey","PQ3_respkey",
    "PQ1_resptime","PQ2_resptime","PQ3_resptime",
    "PQ1_questime","PQ2_questime","PQ3_questime",
    "PQ1_respval","PQ2_respval","PQ3_respval"
    ]

cols_oi = ["sub_id", "subtype", "daytime", "nprobe", 
           "mindstate", "voluntary", "sleepiness",
           "scorred_stage", "P_WAKE", "P_N1", "P_N2", "P_N3", "P_REM"]

subtype_palette = ["#8d99ae", "#d00000", "#ffb703"]

# %% Script

# big_dict = {col : [] for col in cols_oi}

if os.path.exists(os.path.join(
    path_usleep, "yasa", "features", "all_hypnodensities.csv"
    )) : 
    df = pd.read_csv(os.path.join(
        path_usleep, "yasa", "features", "all_hypnodensities.csv"
        ))
    del df['Unnamed: 0']
else :

    for i, file in enumerate(files) :
    
        sub_id = file.split('yasa/')[-1].split('_hypno')[0]
        
        if sub_id in ["N1_001_PM", "HS_007_AM"] : continue
    
        this_savepath = os.path.join(path_usleep, 'yasa', 'features', f"{sub_id}.csv")
        
        if os.path.exists(this_savepath) : continue
        
        daytime = sub_id[-2:]
        sub_id = sub_id[:-3]
        subtype = sub_id[:2]
        
        temp_dict = {col : [] for col in cols_oi}
        
        print(f"Processing {sub_id}, n° {i+1} / {len(files)}...")
        
        rawpath = glob(os.path.join(path_data, 'experiment', f'*{sub_id}*' , f'*SART*{daytime}*.vhdr'))
        behav_paths = glob(os.path.join(
            path_data, "experiment", f"*{sub_id}*", "*.mat"
            ))
        
        if len(rawpath) < 1 : print("No EEG file found"); continue;
    
        raw = mne.io.read_raw_brainvision(rawpath[0], preload = True)
        events, event_id = mne.events_from_annotations(raw)
        timing_probes = events[events[:, 2] == 3, 0]/raw.info['sfreq']
        
        #### Extract Behav Infos
        if len(behav_paths) < 1 :
            print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
            continue
        
        if sub_id == 'HS_007' : behav_path = behav_paths[0]
        else :
            if daytime == "AM" :
                behav_path = behav_paths[0]
            else :
                daytime = behav_paths[1]
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
        
        if sub_id == "HI_002" and daytime == "AM" :
            ms_answers = ms_answers[1:]
            vol_answers = vol_answers[1:]
            sleepi_answers = sleepi_answers[1:]
        if sub_id == "N1_009" and daytime == "AM" :
            ms_answers = ms_answers[13:]
            vol_answers = vol_answers[13:]
            sleepi_answers = sleepi_answers[13:]
        if sub_id == "N1_015" and daytime == "PM" :
            ms_answers = ms_answers[:30]
            vol_answers = vol_answers[:30]
            sleepi_answers = sleepi_answers[:30]
        
        print(f"""\nIn {sub_id} file, were found :
            * {timing_probes.shape[0]} Probes (first question)
            -> {ms_answers.shape[0]} MS Answers
            -> {vol_answers.shape[0]} Voluntary Answers
            -> {sleepi_answers.shape[0]} Sleepiness answers""")
            
        if not len(timing_probes) == len(ms_answers) : 
            print(f"!!!\n{sub_id} : Careful, inconsistencies found between EEG and Behav\n!!!")
            input("\nWaiting for further instructions...")
    
        df_h = pd.read_csv(file)    
        del df_h['Unnamed: 0']
        
        n_epochs = df_h.epoch.values
        epoch_timings = n_epochs * 30
        
        epoch_idx = np.searchsorted(epoch_timings, timing_probes, side='right') - 1
        
        this_hd = df_h.loc[epoch_idx]
        print(this_hd.shape)
        
        for probe in range(ms_answers.shape[0]):
            temp_dict["sub_id"].append(sub_id)
            temp_dict["subtype"].append(subtype)
            temp_dict["daytime"].append(daytime)
            temp_dict["nprobe"].append(probe)
            temp_dict["mindstate"].append(ms_answers[probe])
            temp_dict["voluntary"].append(vol_answers[probe])
            temp_dict["sleepiness"].append(sleepi_answers[probe])
            temp_dict["scorred_stage"].append(this_hd.scorred_stage.iloc[probe])
            temp_dict["P_WAKE"].append(this_hd.W.iloc[probe])
            temp_dict["P_N1"].append(this_hd.N1.iloc[probe])
            temp_dict["P_N2"].append(this_hd.N2.iloc[probe])
            temp_dict["P_N3"].append(this_hd.N3.iloc[probe])
            temp_dict["P_REM"].append(this_hd.R.iloc[probe])
            
        df = pd.DataFrame.from_dict(temp_dict)
        df.to_csv(this_savepath)
        
    files = glob(os.path.join(path_usleep, "yasa", "features", "*.csv"))
    
    df = pd.concat([pd.read_csv(file) for file in files])
    
    df.to_csv(os.path.join(
        path_usleep, "yasa", "features", "all_hypnodensities.csv"
        ))    

# %% 

df_oi = df.loc[df.mindstate.isin(["ON", "MW", "MB", "HALLU", "FORGOT"])]

mean_df = df_oi[
    ["sub_id", "subtype", "nprobe", "mindstate", "sleepiness", 
    "P_WAKE", "P_N1", "P_N2", "P_N3", "P_REM"]
    ].groupby(["sub_id", "subtype", "mindstate"], 
              as_index=False
              ).mean()

# %% all sleep stage

data = mean_df.copy()
x = "subtype"
order = ["HS", "N1", "HI"]   

fig, ax = plt.subplots(
    nrows = 5, ncols = 1, figsize = (3, 8), sharex = True, sharey = True
    )
for i, y in enumerate(['P_WAKE', 'P_N1', 'P_N2', 'P_N3', 'P_REM']) :

    sns.pointplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        errorbar = 'se',
        capsize = 0.05,
        linestyle = 'none',
        palette = subtype_palette,
        dodge = 0.55,
        ax = ax[i]
        )   
    sns.violinplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        dodge = True,
        fill = True, 
        inner = None,
        cut = .1,
        palette = subtype_palette,
        legend = None,
        gap = .05,
        alpha = .1,
        linecolor = "white",
        ax = ax[i]
        )             
    sns.stripplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        alpha = 0.1,
        dodge = True,
        legend = None,
        palette = subtype_palette,
        ax = ax[i]
        )
    
    ax[i].set_ylabel(y, size = 14, font = bold_font)
    ax[i].set_xlabel('Group', size = 18, font = bold_font)
    ax[i].set_ylim(0, np.round(data[y].max(), 1))
    ax[i].set_xticks(
        ticks = np.linspace(0, 2, 3), 
        labels = ["HS", "N1", "HI"]   ,
        font = font, 
        fontsize = 12)
    ax[i].set_yticks(
        ticks = np.linspace(0, 1, 6), 
        labels = np.round(np.linspace(0, 1, 6), 1), 
        font = font, fontsize = 12)
    sns.despine()
fig.tight_layout()

plt.savefig(
    os.path.join(path_usleep, "yasa", "figs", "yasa_all_proba_per_group.png"), 
    dpi=200
    )
    
# %% Plots Subtype x Mindstate

poi = ["P_WAKE", "P_N1", "P_N2", "P_REM"]
# poi = ["P_WAKE", "P_REM"]
data = mean_df.copy()

hue = "subtype"
hue_order = ["HS", "N1", "HI"]   
x = "mindstate"
order = ["ON", "MW", "MB", "HALLU", "FORGOT"]   

fig, axs = plt.subplots(
    nrows = len(poi),
    ncols = 1,
    figsize = (6,12),
    sharex=True,
    sharey=True
    )
for i_p, p in enumerate(poi):

    y = p
    
    sns.pointplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        hue = hue,
        hue_order = hue_order,
        errorbar = 'se',
        capsize = 0.05,
        linestyle = 'none',
        palette = subtype_palette,
        dodge=.55,
        ax = axs[i_p],
        legend= None
        )                
    sns.stripplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        hue = hue,
        hue_order = hue_order,
        alpha = 0.2,
        dodge = True,
        legend = None,
        palette = subtype_palette,
        ax = axs[i_p],
        )
    
    axs[i_p].set_ylabel(p, size = 18, font = bold_font)
    axs[i_p].set_xlabel('Group', size = 18, font = bold_font)
    axs[i_p].set_ylim(0, 1)
    # ax.set_xticks(
    #     ticks = np.arange(0, 2, 1), 
    #     labels = ["HS", "N1", "HI"]   ,
    #     font = font, 
    #     fontsize = 10)
    
    axs[i_p].set_yticks(
        ticks = np.arange(0, 1.2, .2), 
        labels = np.arange(0, 120, 20), 
        font = font, fontsize = 12)
    sns.despine()

fig.tight_layout(pad=1.5)