#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/07/23

@author: arthur.lecoz

06_3_explore_aw.py
"""

# %%% Paths & Packages

import SLHIP_config_ALC as config
import mne
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection

from highlight_text import fig_text
from matplotlib.font_manager import FontProperties
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

cleanDataPath = config.cleanDataPath
wavesPath = config.wavesPath

allwaves_path = os.path.join(wavesPath, "all_waves")
allwaves_files = glob(os.path.join(allwaves_path, "*.csv"))
reports_path = os.path.join(wavesPath, "reports")
figs_path = os.path.join(wavesPath, "figs")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

channels = np.array(config.eeg_channels)

# %% Compute DF  

slope_range = [0.125, 2] # in uV/ms
positive_amp = [75] # in uV
amplitude_max = 150

# df_aw = pd.concat([pd.read_csv(file) for file in allwaves_files])

df_list = []

features = [
    "sub_id", "subtype", 
    "density", "ptp", "uslope", "dslope", "frequency", 
    "channel", "n_epoch", "nblock", 
    "nprobe", 'mindstate','voluntary', 'sleepiness'
    ]

for file in allwaves_files :
    
    bigdic = {feat : [] for feat in features}

    sub_id = file.split('all_waves/')[-1].split('.')[0]
    session = sub_id[-2:]
    sub_id = sub_id[:-3]
    subtype = sub_id[:2]

    df = pd.read_csv(file)
    del df['Unnamed: 0']
    
    df = df.loc[
        (df['PTP'] < 150) 
        & (df['PTP'] > 0)
        & (df['pos_amp_peak'] < 75)]
    df = df.loc[
        (df["pos_halfway_period"] <= slope_range[1])
        & (df["pos_halfway_period"] >= slope_range[0])
        ]
    
    # df['sub_id'] = [sub_id for i in range(df.shape[0])]
    # df['session'] = [session for i in range(df.shape[0])]
    # df['subtype'] = [subtype for i in range(df.shape[0])]
    
    # df_list.append(df)    
    
    epochs_observed = int(df.tot_epochs.unique()[0])
    epochs_files = glob(os.path.join(
        cleanDataPath, 
        "epochs_probes",
        f"*{sub_id}*{session}*epo.fif"
        ))[0]
    epochs = mne.read_epochs(epochs_files, preload = False)
    metadata = epochs.metadata
    del epochs
    
    for n_block in df.nblock.unique() :
        for n_epoch in range(epochs_observed) :
            
            if sub_id == 'HI_002' and session == "AM" and n_block == 0 and n_epoch == 0:
                continue
            
            sub_df = df.loc[
                (df.nblock == n_block)
                & (df.n_epoch == n_epoch)]
            
            if sub_df.empty :
                
                mindstate = metadata.iloc[n_epoch].mindstate
                voluntary = metadata.iloc[n_epoch].voluntary
                sleepiness = metadata.iloc[n_epoch].sleepiness
                # nblock = metadata.iloc[n_epoch].nblock
                nprobe = metadata.iloc[n_epoch].nprobe
            
            else : 
                mindstate = sub_df.mindstate.iloc[0]
                voluntary = sub_df.voluntary.iloc[0]
                sleepiness = sub_df.sleepiness.iloc[0]
                # nblock = sub_df.nblock.iloc[0]
                nprobe = sub_df.nprobe.iloc[0]
            
            for chan in channels :
                temp_df = sub_df.loc[sub_df["channel"] == chan]
            
                n_waves = temp_df.shape[0]
                
                bigdic['density'].append(n_waves)
                
                bigdic['sub_id'].append(sub_id)
                bigdic['subtype'].append(subtype)
                
                bigdic['channel'].append(chan)
                bigdic['n_epoch'].append(n_epoch)
                bigdic['nblock'].append(n_block)
                
                bigdic['nprobe'].append(nprobe)
                bigdic['mindstate'].append(mindstate)
                bigdic['voluntary'].append(voluntary)
                bigdic['sleepiness'].append(sleepiness)
                
                if n_waves == 0 :
                    bigdic['ptp'].append(np.nan)
                    bigdic['uslope'].append(np.nan)
                    bigdic['dslope'].append(np.nan)
                    bigdic['frequency'].append(np.nan)
                else : 
                    bigdic['ptp'].append(np.nanmean(temp_df.PTP))
                    bigdic['uslope'].append(np.nanmean(
                        temp_df.max_pos_slope_2nd_segment
                           ))
                    bigdic['dslope'].append(np.nanmean(
                        temp_df.inst_neg_1st_segment_slope))
                    bigdic['frequency'].append(np.nanmean(
                        1/temp_df.pos_halfway_period))
            
    df_feature = pd.DataFrame.from_dict(bigdic)
    # df_feature.to_csv(this_df_swfeat_savepath)
    df_list.append(df_feature)
    
df_aw = pd.concat(df_list)
 
mean_df = df_aw[[
    'sub_id', 'subtype', 'channel',  'mindstate', 
    'density', 'ptp', 'uslope', 'dslope', 'frequency', 'sleepiness']
    ].groupby(
           ['sub_id', 'subtype', 'channel', 'mindstate'], 
           as_index = False).mean()

# %% 

fig, ax = plt.subplots()
for subtype in ['HS', 'N1', 'HI']:
    ax.hist(
            mean_df.ptp.loc[mean_df.subtype == subtype], 
            bins = 30, 
            density = True, 
            histtype = 'step',
            label = subtype
            )
    ax.legend()

# %% simple pointplot

this_df = df_aw[
    ['sub_id', 'subtype', 'density', 'ptp', 'uslope', 'dslope', 'frequency']].groupby(
           ['sub_id', 'subtype'], as_index = False).mean()   

data = this_df     
y = 'ptp'
x = "subtype"
order = ['HS', 'N1', 'HI']
# hue = "subtype"
# hue_order = ['HS', 'N1']
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 14))
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    # hue = hue,
    order = order,
    # hue_order = hue_order,
    errorbar = 'se',
    capsize = 0.05,
    # dodge = .4,
    linestyle = 'none'
    )         
# sns.violinplot(
#     data = data, 
#     x = x,
#     y = y,
#     # hue = hue,
#     order = order,
#     # hue_order = hue_order,
#     fill = True,
#     alpha = 0.2,
#     dodge = True,
#     linecolor = 'white',
#     inner = None,
#     legend = None
#     )        
sns.stripplot(
    data = data, 
    x = x,
    y = y,
    # hue = hue,
    order = order,
    # hue_order = hue_order,
    alpha = 0.3,
    dodge = True,
    legend = None
    )    

# %% Topo | Density | NT1 & HS | Mindstate

epochs = mne.read_epochs(glob(os.path.join(
    cleanDataPath, "epochs_probes", "*epo.fif"))[0]
    )

# %% 

mindstates = ['ON', 'MW', 'MB', 'FORGOT', 'HALLU']
subtypes = ['HS', 'N1']

feature = 'density'

list_values = []
for i_ms, mindstate in enumerate(mindstates) :
    for subtype in subtypes :   
        for channel in channels :
            list_values.append(mean_df[feature].loc[
                (mean_df["mindstate"] == mindstate)
                & (mean_df["subtype"] == subtype)
                & (mean_df["channel"] == channel)
                ].mean())
vmin = min(list_values)
vmax = max(list_values)

fig, ax = plt.subplots(
    nrows = 2, 
    ncols = len(mindstates),
    figsize = (18,7),
    layout = 'tight'
    )
for i_ms, mindstate in enumerate(mindstates) :
    list_easy = []
    list_hard = []        
    for channel in channels :
        list_easy.append(mean_df[feature].loc[
            (mean_df["subtype"] == "HS")
            & (mean_df["mindstate"] == mindstate)
            & (mean_df["channel"] == channel)
            ].mean())
        list_hard.append(mean_df[feature].loc[
            (mean_df["subtype"] == "N1")
            & (mean_df["mindstate"] == mindstate)
            & (mean_df["channel"] == channel)
            ].mean())
    
    if i_ms == 4 :
        divider = make_axes_locatable(ax[0][i_ms])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_easy,
        epochs.info,
        axes = ax[0][i_ms],
        size = 2,
        # names = channels,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "viridis"
        )
    if i_ms == 4 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    # ax[0][i_sess].set_title(f"EASY - S{n_sess}")
    ax[0][i_ms].set_title(f"{subtypes[0]} - {mindstate}", font = bold_font)
    title = """
    Topographies of <Slow Wave Density> according to the <Mindstates> and by <Subtype>
    """
    fig_text(
       0.07, .94,
       title,
       fontsize=15,
       ha='left', va='center',
       color="k", font=font,
       highlight_textprops=[
          {'font': bold_font},
          {'font': bold_font},
          {'font': bold_font},
       ],
       fig=fig
    )
    
    if i_ms == 4 :
        divider = make_axes_locatable(ax[1][i_ms])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_hard,
        epochs.info,
        axes = ax[1][i_ms],
        size = 2,
        # names = channels,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "viridis"
        )
    # ax[1][i_sess].set_title(f"HARD - S{n_sess}")
    ax[1][i_ms].set_title(f"{subtypes[1]} - {mindstate}", font = bold_font)
    if i_ms == 4 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    plt.show(block = False)


# %% 
# %% Compute DF Delta Theta

slope_range = [0.125, 2] # in uV/ms
positive_amp = [75] # in uV
amplitude_max = 150

# df_aw = pd.concat([pd.read_csv(file) for file in allwaves_files])

df_freq_list = []

features = [
    "sub_id", "subtype", 
    "density", "ptp", "uslope", "dslope", "frequency", "frequency_range",
    "channel", "n_epoch", "nblock", 
    "nprobe", 'mindstate','voluntary', 'sleepiness', 
    ]

for file in allwaves_files :
    
    bigdic = {feat : [] for feat in features}

    sub_id = file.split('all_waves/')[-1].split('.')[0]
    session = sub_id[-2:]
    sub_id = sub_id[:-3]
    subtype = sub_id[:2]

    df = pd.read_csv(file)
    del df['Unnamed: 0']
    
    df = df.loc[
        (df['PTP'] < 150) 
        & (df['PTP'] > 0)
        & (df['pos_amp_peak'] < 75)]
    df = df.loc[
        (df["pos_halfway_period"] <= slope_range[1])
        & (df["pos_halfway_period"] >= slope_range[0])
        ]
    df['frequency'] = 1/df.pos_halfway_period
    df['frequency_range'] = ["delta" if frequency<=4 else "theta" 
                             for frequency in df.frequency]
    
    # df['sub_id'] = [sub_id for i in range(df.shape[0])]
    # df['session'] = [session for i in range(df.shape[0])]
    # df['subtype'] = [subtype for i in range(df.shape[0])]
    
    # df_list.append(df)    
    
    epochs_observed = int(df.tot_epochs.unique()[0])
    epochs_files = glob(os.path.join(
        cleanDataPath, 
        "epochs_probes",
        f"*{sub_id}*{session}*epo.fif"
        ))[0]
    epochs = mne.read_epochs(epochs_files, preload = False)
    metadata = epochs.metadata
    del epochs
    
    for n_block in df.nblock.unique() :
        for n_epoch in range(epochs_observed) :
            if sub_id == 'HI_002' and session == "AM" and n_block == 0 and n_epoch == 0:
                continue
            for frequency_range in df.frequency_range.unique() :
            
                sub_df = df.loc[
                    (df.nblock == n_block)
                    & (df.n_epoch == n_epoch)
                    & (df.frequency_range == frequency_range)
                    ]
                
                if sub_df.empty :
                    
                    mindstate = metadata.iloc[n_epoch].mindstate
                    voluntary = metadata.iloc[n_epoch].voluntary
                    sleepiness = metadata.iloc[n_epoch].sleepiness
                    # nblock = metadata.iloc[n_epoch].nblock
                    nprobe = metadata.iloc[n_epoch].nprobe
                
                else : 
                    mindstate = sub_df.mindstate.iloc[0]
                    voluntary = sub_df.voluntary.iloc[0]
                    sleepiness = sub_df.sleepiness.iloc[0]
                    # nblock = sub_df.nblock.iloc[0]
                    nprobe = sub_df.nprobe.iloc[0]
                
                for chan in channels :
                    temp_df = sub_df.loc[sub_df["channel"] == chan]
                
                    n_waves = temp_df.shape[0]
                    
                    bigdic['density'].append(n_waves)
                    
                    bigdic['sub_id'].append(sub_id)
                    bigdic['subtype'].append(subtype)
                    
                    bigdic['channel'].append(chan)
                    bigdic['n_epoch'].append(n_epoch)
                    bigdic['nblock'].append(n_block)
                    bigdic['frequency_range'].append(frequency_range)
                    
                    bigdic['nprobe'].append(nprobe)
                    bigdic['mindstate'].append(mindstate)
                    bigdic['voluntary'].append(voluntary)
                    bigdic['sleepiness'].append(sleepiness)
                    
                    if n_waves == 0 :
                        bigdic['ptp'].append(np.nan)
                        bigdic['uslope'].append(np.nan)
                        bigdic['dslope'].append(np.nan)
                        bigdic['frequency'].append(np.nan)
                    else : 
                        bigdic['ptp'].append(np.nanmean(temp_df.PTP))
                        bigdic['uslope'].append(np.nanmean(
                            temp_df.max_pos_slope_2nd_segment
                               ))
                        bigdic['dslope'].append(np.nanmean(
                            temp_df.inst_neg_1st_segment_slope))
                        bigdic['frequency'].append(np.nanmean(
                            1/temp_df.pos_halfway_period))
            
    df_feature = pd.DataFrame.from_dict(bigdic)
    # df_feature.to_csv(this_df_swfeat_savepath)
    df_freq_list.append(df_feature)
    
df_freq_aw = pd.concat(df_freq_list)
mean_freq_df = df_freq_aw[[
    'sub_id', 'subtype', 'channel',  'mindstate', 'frequency_range',
    'density', 'ptp', 'uslope', 'dslope', 'frequency', 'sleepiness']
    ].groupby(
           ['sub_id', 'subtype', 'channel', 'frequency_range'], 
           as_index = False).mean()

# %% hist - freq range 

this_freq_df = df_freq_aw[[
    'sub_id', 'subtype', 'channel',   'frequency_range',
    'density', 'ptp', 'uslope', 'dslope', 'frequency', 'sleepiness']
    ].groupby(
           ['sub_id', 'subtype', 'channel', 'frequency_range'], 
           as_index = False).mean()

fig, ax = plt.subplots(nrows = 1, ncols = 2)
for subtype in ['HS', 'N1', 'HI']:
    for i, frequency_range in enumerate(['delta', 'theta']) :
        subdf = this_freq_df.loc[this_freq_df.frequency_range == frequency_range]
        ax[i].hist(
                subdf.ptp.loc[subdf.subtype == subtype], 
                bins = 30, 
                density = True, 
                histtype = 'step',
                label = subtype
                )
        ax[i].set_title(frequency_range)
        ax[i].legend()

# %% 

epochs = mne.read_epochs(glob(os.path.join(
    cleanDataPath, "epochs_probes", "*epo.fif"))[0]
    )

# %%

this_mean_df = df_freq_aw[[
    'sub_id', 'subtype', 'channel', 'frequency_range',
    'density', 'ptp', 'uslope', 'dslope', 'frequency', 'sleepiness']
    ].groupby(
           ['sub_id', 'subtype', 'channel', 'frequency_range'], 
           as_index = False).mean()

frequency_ranges = ['delta', 'theta']
subtypes = ['HS', 'N1', 'HI']

feature = 'density'

# list_values = []
# for i_fr, frequency_range in enumerate(frequency_ranges) :
#     for subtype in subtypes :   
#         for channel in channels :
#             list_values.append(this_mean_df[feature].loc[
#                 (this_mean_df["frequency_range"] == frequency_range)
#                 & (this_mean_df["subtype"] == subtype)
#                 & (this_mean_df["channel"] == channel)
#                 ].mean())
# vmin = min(list_values)
# vmax = max(list_values)

fig, ax = plt.subplots(
    nrows = 3, 
    ncols = len(frequency_ranges),
    figsize = (18,7),
    layout = 'tight'
    )
for i_fr, frequency_range in enumerate(frequency_ranges) :
    list_hs = []
    list_n1 = []      
    list_hi = []      
    for channel in channels :
        list_hs.append(this_mean_df[feature].loc[
            (this_mean_df["subtype"] == "HS")
            & (this_mean_df["frequency_range"] == frequency_range)
            & (this_mean_df["channel"] == channel)
            ].mean())
        list_n1.append(this_mean_df[feature].loc[
            (this_mean_df["subtype"] == "N1")
            & (this_mean_df["frequency_range"] == frequency_range)
            & (this_mean_df["channel"] == channel)
            ].mean())
        list_hi.append(this_mean_df[feature].loc[
            (this_mean_df["subtype"] == "HI")
            & (this_mean_df["frequency_range"] == frequency_range)
            & (this_mean_df["channel"] == channel)
            ].mean())
    
    # if i_fr == 1 :
    divider = make_axes_locatable(ax[0][i_fr])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_hs,
        epochs.info,
        axes = ax[0][i_fr],
        size = 2,
        show = False,
        contours = 2,
        # vlim = (vmin, vmax),
        cmap = "viridis"
        )
    # if i_fr == 1 :
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    # ax[0][i_sess].set_title(f"EASY - S{n_sess}")
    ax[0][i_fr].set_title(f"{subtypes[0]} - {frequency_range}", font = bold_font)
    # title = """
    # Topographies of <Slow Wave Density> according to the <Frequency Range> and by <Subtype>
    # """
    # fig_text(
    #    0.07, .94,
    #    title,
    #    fontsize=15,
    #    ha='left', va='center',
    #    color="k", font=font,
    #    highlight_textprops=[
    #       {'font': bold_font},
    #       {'font': bold_font},
    #       {'font': bold_font},
    #    ],
    #    fig=fig
    # )
    
    # if i_fr == 1 :
    divider = make_axes_locatable(ax[1][i_fr])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_n1,
        epochs.info,
        axes = ax[1][i_fr],
        size = 2,
        show = False,
        contours = 2,
        # vlim = (vmin, vmax),
        cmap = "viridis"
        )
    # ax[1][i_sess].set_title(f"HARD - S{n_sess}")
    ax[1][i_fr].set_title(f"{subtypes[1]} - {frequency_range}", font = bold_font)
    # if i_ms == 1 :
    fig.colorbar(im, cax = cax, orientation = 'vertical')
        
    # if i_fr == 1 :
    divider = make_axes_locatable(ax[2][i_fr])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_hi,
        epochs.info,
        axes = ax[2][i_fr],
        size = 2,
        show = False,
        contours = 2,
        # vlim = (vmin, vmax),
        cmap = "viridis"
        )
    # ax[1][i_sess].set_title(f"HARD - S{n_sess}")
    ax[2][i_fr].set_title(f"{subtypes[2]} - {frequency_range}", font = bold_font)
    # if i_ms == 1 :
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    plt.show(block = False)

fig.tight_layout()

