#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/07/23

@author: arthur.lecoz

06_3_explore_sw.py
"""

# %%% Paths & Packages

import SLHIP_config_ALC as config
import mne
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

slowwaves_path = os.path.join(wavesPath, "slow_waves")
slowwaves_files = glob(os.path.join(slowwaves_path, "slow_waves*.csv"))
reports_path = os.path.join(wavesPath, "reports")
figs_path = os.path.join(wavesPath, "figs")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

channels = np.array(config.eeg_channels)
palette = ["#8d99ae", "#d00000", "#ffb703"]

# %% df load

#### DF
df = pd.read_csv(os.path.join(
    wavesPath, "features", "all_SW_features_0.5_4.csv"
    ))
del df['Unnamed: 0']

# channel_category = pd.Categorical(
#     df['channel'], 
#     categories = channels, 
#     ordered=True
#     )
# mean_df = df.loc[channel_category.argsort()]

#### DF MEAN
# mean_df = pd.read_csv(f"{swDataPath}/df_meansw_exgausscrit_computedS1_freq_0.5_4.0.csv")
# del mean_df['Unnamed: 0']
# mean_df = mean_df.loc[mean_df['sub_id'] != "2_pf"]
# mean_df = mean_df.loc[mean_df['sub_id'] != "6_yh"]
# mean_df = mean_df.loc[mean_df['sub_id'] != "26_eb"]

epochs = mne.read_epochs(glob(f"{cleanDataPath}/epochs_probes/*.fif")[0])

# df = df.loc[df.subtype != 'HI']
df = df.loc[~df.mindstate.isin(['DISTRACTED', 'MISS', 'FORGOT'])]

mindstates = ['ON', 'MW', 'MB', 'HALLU']
subtypes = ['HS', 'N1', 'HI']
channels = config.eeg_channels

mean_df = df[[
    'sub_id', 'subtype', 
    'density_20', 'density_90_hs_global','density_90_hs_chan', 
    'ptp_20', 'ptp_90_hs_global', 'ptp_90_hs_chan',
    'uslope_20', 'uslope_90_hs_global', 'uslope_90_hs_chan', 
    'dslope_20', 'dslope_90_hs_global', 'dslope_90_hs_chan', 
    'channel', 'mindstate', 'sleepiness', 'rt_go',
    'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
       ]].groupby(
           ['sub_id', 'subtype','channel', 'mindstate'], 
           as_index = False).mean()
    
# %% simple pointplot

import seaborn as sns

this_df = df[[
    'sub_id', 'subtype', 
    'density_20', 'density_90_hs_global','density_90_hs_chan', 
    'ptp_20', 'ptp_90_hs_global', 'ptp_90_hs_chan',
    'uslope_20', 'uslope_90_hs_global', 'uslope_90_hs_chan', 
    'dslope_20', 'dslope_90_hs_global', 'dslope_90_hs_chan', 
    'sleepiness', 'rt_go',
    'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
       ]].groupby(
           ['sub_id', 'subtype'], 
           as_index = False).mean() 

data = this_df     
y = 'density_20'
x = "subtype"
order = ['HS', 'N1', 'HI']
# hue = "subtype"
# hue_order = ['HS', 'N1', 'HI']    
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 4))
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
    linestyle = 'none', 
    palette = palette
    )         
sns.violinplot(
    data = data, 
    x = x,
    y = y,
    # hue = hue,
    order = order,
    # hue_order = hue_order,
    fill = True,
    alpha = 0.2,
    # dodge = True,
    linecolor = 'white',
    inner = None,
    legend = None,
    cut = 1, 
    palette = palette
    )        
sns.stripplot(
    data = data, 
    x = x,
    y = y,
    # hue = hue,
    order = order,
    # hue_order = hue_order,
    alpha = 0.3,
    # dodge = True,
    legend = None, 
    palette = palette
    )

ax.set_xticks(
    np.linspace(0, 2, 3), 
    ['HS', 'N1', 'HS'], 
    font = bold_font, 
    fontsize = 16
    )
ax.set_yticks(
    np.linspace(0, 3, 4), 
    np.linspace(0, 3, 4).astype(int), 
    font = font, 
    fontsize = 16
    )
ax.set_ylim(0, 3)
ax.set_ylabel("SW Density (n/Epoch)", font=bold_font, fontsize = 20)
ax.set_xlabel("", font=bold_font, fontsize = 20)
sns.despine(bottom=True)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    )

fig.tight_layout()

plt.savefig(os.path.join(wavesPath, "figs", f"sw_{y}_subtype.png"), dpi = 300)

print_stats = True
if print_stats :
    model = f"{y} ~ C(subtype, Treatment('N1'))"
    md = smf.mixedlm(model, this_df, groups = this_df['sub_id'], missing = 'drop')
    mdf = md.fit()
    print(mdf.summary())
           
# %% simple pointplot

import seaborn as sns

this_df = df[[
    'sub_id', 'subtype', 
    'density_20', 'density_90_hs_global','density_90_hs_chan', 
    'ptp_20', 'ptp_90_hs_global', 'ptp_90_hs_chan',
    'uslope_20', 'uslope_90_hs_global', 'uslope_90_hs_chan', 
    'dslope_20', 'dslope_90_hs_global', 'dslope_90_hs_chan', 
    'mindstate', 'sleepiness', 'rt_go',
    'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
       ]].groupby(
           ['sub_id', 'subtype', 'mindstate'], 
           as_index = False).mean() 

data = this_df     
y = 'density_20'
x = "mindstate"
order = ['ON', 'MW', 'MB', 'HALLU']
hue = "subtype"
hue_order = ['HS', 'N1', 'HI']    
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 14))
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    errorbar = 'se',
    capsize = 0.05,
    dodge = .4,
    linestyle = 'none'
    )         
sns.violinplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    fill = True,
    alpha = 0.2,
    dodge = True,
    linecolor = 'white',
    inner = None,
    legend = None,
    cut = 2
    )        
sns.stripplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    alpha = 0.3,
    dodge = True,
    legend = None
    )

print_stats = False
if print_stats :
    model = "density_20 ~ C(subtype, Treatment('HS')) * C(mindstate, Treatment('HALLU'))"
    md = smf.mixedlm(model, this_df, groups = this_df['sub_id'], missing = 'drop')
    mdf = md.fit()
    print(mdf.summary())
           
# %% Topo | Density | HS, NT1, HI

feature = 'dslope_20'

list_values = []
for i_ms, mindstate in enumerate(mindstates) :
    for subtype in subtypes :   
        for channel in channels :
            list_values.append(mean_df[feature].loc[
                (mean_df["subtype"] == subtype)
                & (mean_df["channel"] == channel)
                ].mean())
vmin = min(list_values)
vmax = max(list_values)

fig, ax = plt.subplots(
    nrows = 3, 
    ncols = 1,
    figsize = (4,7),
    )

list_hs = []
list_n1 = []        
list_hi = []        
for channel in channels :
    list_hs.append(mean_df[feature].loc[
        (mean_df["subtype"] == "HS")
        & (mean_df["channel"] == channel)
        ].mean())
    list_n1.append(mean_df[feature].loc[
        (mean_df["subtype"] == "N1")
        & (mean_df["channel"] == channel)
        ].mean())
    list_hi.append(mean_df[feature].loc[
        (mean_df["subtype"] == "HI")
        & (mean_df["channel"] == channel)
        ].mean())

if i_ms == 3 :
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
im, cm = mne.viz.plot_topomap(
    list_hs,
    epochs.info,
    axes = ax[0],
    size = 2,
    # names = channels,
    show = False,
    contours = 2,
    vlim = (vmin, vmax),
    cmap = "Purples"
    )
if i_ms == 3 :
    fig.colorbar(im, cax = cax, orientation = 'vertical')
# ax[0][i_sess].set_title(f"EASY - S{n_sess}")
ax[0].set_title(f"{subtypes[0]}", font = bold_font)
title = f"""
<Slow Wave {feature}> by <Subtype>
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
   ],
   fig=fig
)

if i_ms == 3 :
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
im, cm = mne.viz.plot_topomap(
    list_n1,
    epochs.info,
    axes = ax[1],
    size = 2,
    # names = channels,
    show = False,
    contours = 2,
    vlim = (vmin, vmax),
    cmap = "Purples"
    )
# ax[1][i_sess].set_title(f"HARD - S{n_sess}")
ax[1].set_title(f"{subtypes[1]}", font = bold_font)
if i_ms == 3 :
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    
if i_ms == 3 :
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
im, cm = mne.viz.plot_topomap(
    list_hi,
    epochs.info,
    axes = ax[2],
    size = 2,
    # names = channels,
    show = False,
    contours = 2,
    vlim = (vmin, vmax),
    cmap = "Purples"
    )
# ax[1][i_sess].set_title(f"HARD - S{n_sess}")
ax[2].set_title(f"{subtypes[2]}", font = bold_font)
if i_ms == 3 :
    fig.colorbar(im, cax = cax, orientation = 'vertical')
plt.show(block = False)

figsavename = os.path.join(
    wavesPath, 'figs', f'topo_mindstates_subtypes_{feature}.png'
    )
plt.savefig(figsavename, dpi = 300)

# %% Topo | Density | HS, NT1, HI | Mindstate

feature = 'uslope_20'

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
    nrows = 3, 
    ncols = len(mindstates),
    figsize = (18,7),
    # layout = 'tight'
    )
for i_ms, mindstate in enumerate(mindstates) :
    list_hs = []
    list_n1 = []        
    list_hi = []        
    for channel in channels :
        list_hs.append(np.nanmean(mean_df[feature].loc[
            (mean_df["subtype"] == "HS")
            & (mean_df["mindstate"] == mindstate)
            & (mean_df["channel"] == channel)
            ], axis = 0))
        list_n1.append(np.nanmean(mean_df[feature].loc[
            (mean_df["subtype"] == "N1")
            & (mean_df["mindstate"] == mindstate)
            & (mean_df["channel"] == channel)
            ], axis = 0))
        list_hi.append(np.nanmean(mean_df[feature].loc[
            (mean_df["subtype"] == "HI")
            & (mean_df["mindstate"] == mindstate)
            & (mean_df["channel"] == channel)
            ], axis = 0))
        
    if np.any(np.isnan(list_hi)):
        for t, tval in enumerate(list_hi) :
            if np.isnan(tval):
                list_hi[t] = np.nanmean(list_hi)
    if np.any(np.isnan(list_hs)):
        for t, tval in enumerate(list_hs) :
            if np.isnan(tval):
                list_hs[t] = np.nanmean(list_hs)
        
    if i_ms == 3 :
        divider = make_axes_locatable(ax[0][i_ms])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_hs,
        epochs.info,
        axes = ax[0][i_ms],
        size = 2,
        # names = channels,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "Purples"
        )
    if i_ms == 3 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    # ax[0][i_sess].set_title(f"EASY - S{n_sess}")
    ax[0][i_ms].set_title(f"{subtypes[0]} - {mindstate}", font = bold_font)
    title = """
    Topographies of <Slow Wave Upward Slope> according to the <Mindstates> and by <Subtype>
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
    
    if i_ms == 3 :
        divider = make_axes_locatable(ax[1][i_ms])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_n1,
        epochs.info,
        axes = ax[1][i_ms],
        size = 2,
        # names = channels,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "Purples"
        )
    # ax[1][i_sess].set_title(f"HARD - S{n_sess}")
    ax[1][i_ms].set_title(f"{subtypes[1]} - {mindstate}", font = bold_font)
    if i_ms == 3 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
        
    if i_ms == 3 :
        divider = make_axes_locatable(ax[2][i_ms])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_hi,
        epochs.info,
        axes = ax[2][i_ms],
        size = 2,
        # names = channels,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "Purples"
        )
    # ax[1][i_sess].set_title(f"HARD - S{n_sess}")
    ax[2][i_ms].set_title(f"{subtypes[2]} - {mindstate}", font = bold_font)
    if i_ms == 3 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    plt.show(block = False)

    figsavename = os.path.join(
        wavesPath, 'figs', f'topo_mindstates_subtypes_{feature}.png'
        )
    # plt.savefig(figsavename, dpi = 300)

# %% Topo | LME - MS > ON effect

interest = 'density_20'

model = f"{interest} ~ C(mindstate, Treatment('ON'))" 

fig, ax = plt.subplots(
    nrows = 1, ncols = len(mindstates[1:]), figsize = (18, 4))
for i, mindstate in enumerate(mindstates[1:]):
    temp_tval = []; temp_pval = []; chan_l = []
    cond_df = mean_df.loc[mean_df.mindstate.isin(['ON', mindstate])]
    for chan in channels :
        subdf = cond_df[
            ['sub_id', 'subtype', 'mindstate', 'channel', f'{interest}']
            ].loc[(cond_df.channel == chan)].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
        mdf = md.fit()
        temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('ON'))[T.{mindstate}]"])
        temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('ON'))[T.{mindstate}]"])
        chan_l.append(chan)
        
    if np.any(np.isnan(temp_tval)) :
        temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
         
    # _, corrected_pval = fdrcorrection(temp_pval)
    
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval,
        pos = epochs.info,
        axes = ax[i],
        contours = 3,
        mask = np.asarray(temp_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=6),
        cmap = "viridis",
        # vlim = (-4, 4)
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')

    ax[i].set_title(f"{mindstate}", fontweight = "bold")


title = f"""Topographies of <Slow Wave {interest}> according to the <Mindstate> VS <ON>"""
fig_text(
   0.07, .98,
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

# %% Topo | LME - MS > ON effect

interest = 'density_20'

model = f"{interest} ~ C(mindstate, Treatment('ON'))" 

for subtype in subtypes :
    fig, ax = plt.subplots(
        nrows = 1, ncols = len(mindstates[1:]), figsize = (18, 4))
    for i, mindstate in enumerate(mindstates[1:]):
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = mean_df.loc[
            (mean_df.mindstate.isin(['ON', mindstate]))
            & (mean_df.subtype == subtype)
            ]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'subtype', 'mindstate', 'channel', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
            mdf = md.fit()
            temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('ON'))[T.{mindstate}]"])
            temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('ON'))[T.{mindstate}]"])
            chan_l.append(chan)
            
        if np.any(np.isnan(temp_tval)) :
            temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
             
        # _, corrected_pval = fdrcorrection(temp_pval)
        
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = temp_tval,
            pos = epochs.info,
            axes = ax[i],
            contours = 3,
            mask = np.asarray(temp_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=6),
            cmap = "viridis",
            # vlim = (-4, 4)
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[i].set_title(f"{subtype} - {mindstate}", fontweight = "bold")
    
    
    title = f"""Topographies of <Slow Wave {interest}> according to the <Mindstate> VS <ON>"""
    fig_text(
       0.07, .98,
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

# %% Topo | LME - Explo Subtype ME

feature = "ptp_20"

model = f"{feature} ~ sleepiness + C(subtype, Treatment('HS'))" 

this_df = mean_df[[
    'sub_id', 'subtype', 
    'density_20', 'density_90_hs_global','density_90_hs_chan', 
    'ptp_20', 'ptp_90_hs_global', 'ptp_90_hs_chan',
    'uslope_20', 'uslope_90_hs_global', 'uslope_90_hs_chan', 
    'dslope_20', 'dslope_90_hs_global', 'dslope_90_hs_chan', 
    'channel', 'mindstate', 'sleepiness', 'rt_go',
    'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
     ]].groupby(
         ['sub_id', 'subtype', 'mindstate', 'channel'], as_index = False).mean()

fig, ax = plt.subplots(
    nrows = 1, ncols = 3, figsize = (12, 5))

for i_s, subtype in enumerate(['N1', 'HI']) :
    temp_tval = []; temp_pval = []; chan_l = []
    for chan in channels :
        subdf = this_df[
            ['sub_id', 'subtype', 'channel', feature, 'mindstate', 'sleepiness']
            ].loc[
            (this_df.channel == chan)
            & (this_df.subtype.isin([subtype, 'HS']))
            ].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'drop')
        mdf = md.fit()
        temp_tval.append(mdf.tvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"])
        temp_pval.append(mdf.pvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"])
        chan_l.append(chan)
        
    if np.any(np.isnan(temp_tval)) :
        temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
         
    _, corrected_pval = fdrcorrection(temp_pval)
    
    # divider = make_axes_locatable(ax[i_s])
    # cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval,
        pos = epochs.info,
        axes = ax[i_s],
        contours = 3,
        mask = np.asarray(temp_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=12),
        cmap = "viridis",
        vlim = (-3, 3),
        size = 2.5
        )
    # fig.colorbar(im, cax = cax, orientation = 'vertical')
    
    ax[i_s].set_title(f"{subtype} vs HS", fontweight = "bold", fontsize = 12)
    

model = f"{feature} ~ C(subtype, Treatment('HI'))" 
temp_tval = []; temp_pval = []; chan_l = []
for chan in channels :
    subdf = this_df[
        ['sub_id', 'subtype', 'channel', feature, 'mindstate', 'sleepiness']
        ].loc[
        (this_df.channel == chan)
        & (this_df.subtype.isin(["N1", "HI"]))
        ].dropna()
    md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'drop')
    mdf = md.fit()
    temp_tval.append(mdf.tvalues["C(subtype, Treatment('HI'))[T.N1]"])
    temp_pval.append(mdf.pvalues["C(subtype, Treatment('HI'))[T.N1]"])
    chan_l.append(chan)
    
if np.any(np.isnan(temp_tval)) :
    temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
     
_, corrected_pval = fdrcorrection(temp_pval)

divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size = "5%", pad=0.05)
im, cm = mne.viz.plot_topomap(
    data = temp_tval,
    pos = epochs.info,
    axes = ax[2],
    contours = 3,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=12),
    cmap = "viridis",
    vlim = (-3, 3),
    size = 2.5
    )
cb = plt.colorbar(im, cax = cax, orientation = 'vertical')
cb.ax.set_yticks(np.linspace(-3, 3, 5), np.linspace(-3, 3, 5), font = bold_font, fontsize = 16)

ax[2].set_title("N1 vs HI", fontweight = "bold", fontsize = 12)
    
# title = """ Topographies of <Subtype Main Effect> on <Slow Wave Density> """
# fig_text(
#    0.07, .94,
#    title,
#    fontsize=15,
#    ha='left', va='center',
#    color="k", font=font,
#    highlight_textprops=[
#       {'font': bold_font},
#       {'font': bold_font},
#    ],
#    fig=fig
# )

# plt.savefig(os.path.join(wavesPath, "figs", f"me_subtype_{feature}.png"), dpi = 300)

#%% DELTA THETA

df = pd.read_csv(os.path.join(
    wavesPath, "features", "all_SW_features_delta_theta_sep.csv"
    ))
del df['Unnamed: 0']

# channel_category = pd.Categorical(
#     df['channel'], 
#     categories = channels, 
#     ordered=True
#     )
# mean_df = df.loc[channel_category.argsort()]

#### DF MEAN
# mean_df = pd.read_csv(f"{swDataPath}/df_meansw_exgausscrit_computedS1_freq_0.5_4.0.csv")
# del mean_df['Unnamed: 0']
# mean_df = mean_df.loc[mean_df['sub_id'] != "2_pf"]
# mean_df = mean_df.loc[mean_df['sub_id'] != "6_yh"]
# mean_df = mean_df.loc[mean_df['sub_id'] != "26_eb"]

epochs = mne.read_epochs(glob(f"{cleanDataPath}/epochs_probes/*.fif")[0])

# df = df.loc[df.subtype != 'HI']
df = df.loc[~df.mindstate.isin(['DISTRACTED', 'MISS', 'FORGOT'])]

mindstates = ['ON', 'MW', 'MB', 'HALLU']
subtypes = ['HS', 'N1', 'HI']
channels = config.eeg_channels

mean_df = df[[
    'sub_id', 'subtype', "freq_range",
    'density_20', 'density_90_hs_global','density_90_hs_chan', 
    'ptp_20', 'ptp_90_hs_global', 'ptp_90_hs_chan',
    'uslope_20', 'uslope_90_hs_global', 'uslope_90_hs_chan', 
    'dslope_20', 'dslope_90_hs_global', 'dslope_90_hs_chan', 
    'channel', 'mindstate', 'sleepiness', 'rt_go',
    'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
       ]].groupby(
           ['sub_id', 'subtype', 'freq_range', 'channel', 'mindstate'], 
           as_index = False).mean()
           
           
# %% simple pointplot

import seaborn as sns

this_df = df[[
    'sub_id', 'subtype', 'freq_range',
    'density_20', 'density_90_hs_global','density_90_hs_chan', 
    'ptp_20', 'ptp_90_hs_global', 'ptp_90_hs_chan',
    'uslope_20', 'uslope_90_hs_global', 'uslope_90_hs_chan', 
    'dslope_20', 'dslope_90_hs_global', 'dslope_90_hs_chan', 
    'sleepiness', 'rt_go',
    'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
       ]].groupby(
           ['sub_id', 'subtype', 'freq_range'], 
           as_index = False).mean() 
 
y = 'density_20'
x = "subtype"
order = ['HS', 'N1', 'HI']
# hue = "subtype"
# hue_order = ['HS', 'N1', 'HI']   
 
         
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (4, 7))
for i, freq_range in enumerate(['delta', 'theta']) :
    temp_df = this_df.loc[this_df.freq_range == freq_range]
    data = temp_df
    
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
        linestyle = 'none', 
        palette = palette,
        ax = ax[i]
        )         
    sns.violinplot(
        data = data, 
        x = x,
        y = y,
        # hue = hue,
        order = order,
        # hue_order = hue_order,
        fill = True,
        alpha = 0.2,
        # dodge = True,
        linecolor = 'white',
        inner = None,
        legend = None,
        cut = 1, 
        palette = palette,
        ax = ax[i]
        )        
    sns.stripplot(
        data = data, 
        x = x,
        y = y,
        # hue = hue,
        order = order,
        # hue_order = hue_order,
        alpha = 0.3,
        # dodge = True,
        legend = None, 
        palette = palette,
        ax = ax[i]
        )
    
    ax[i].set_xticks(
        np.linspace(0, 2, 3), 
        ['HS', 'N1', 'HS'], 
        font = bold_font, 
        fontsize = 16
        )
    # ax.set_yticks(
    #     np.linspace(0, 6, 4), 
    #     np.linspace(0, 6, 4).astype(int), 
    #     font = font, 
    #     fontsize = 16
    #     )
    # ax.set_ylim(0, 6)
    ax[i].set_ylabel("SW Density", font=bold_font, fontsize = 20)
    ax[i].set_xlabel(f"Group {freq_range}", font=bold_font, fontsize = 20)
    sns.despine(bottom=True)
    ax[i].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        )
    
    fig.tight_layout()
    
    # plt.savefig(os.path.join(wavesPath, "figs", "sw_density_20_subtype.png"), dpi = 300)
    
    print_stats = True
    if print_stats :
        model = f"{y} ~ C(subtype, Treatment('N1'))"
        md = smf.mixedlm(model, temp_df, groups = temp_df['sub_id'], missing = 'drop')
        mdf = md.fit()
        print(mdf.summary())
           
# %% simple pointplot

import seaborn as sns

this_df = df[[
    'sub_id', 'subtype', 
    'density_20', 'density_90_hs_global','density_90_hs_chan', 
    'ptp_20', 'ptp_90_hs_global', 'ptp_90_hs_chan',
    'uslope_20', 'uslope_90_hs_global', 'uslope_90_hs_chan', 
    'dslope_20', 'dslope_90_hs_global', 'dslope_90_hs_chan', 
    'mindstate', 'sleepiness', 'rt_go',
    'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
       ]].groupby(
           ['sub_id', 'subtype', 'mindstate'], 
           as_index = False).mean() 

data = this_df     
y = 'density_20'
x = "mindstate"
order = ['ON', 'MW', 'MB', 'HALLU']
hue = "subtype"
hue_order = ['HS', 'N1', 'HI']    
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 14))
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    errorbar = 'se',
    capsize = 0.05,
    dodge = .4,
    linestyle = 'none'
    )         
sns.violinplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    fill = True,
    alpha = 0.2,
    dodge = True,
    linecolor = 'white',
    inner = None,
    legend = None,
    cut = 2
    )        
sns.stripplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    alpha = 0.3,
    dodge = True,
    legend = None
    )

print_stats = False
if print_stats :
    model = "density_20 ~ C(subtype, Treatment('HS')) * C(mindstate, Treatment('HALLU'))"
    md = smf.mixedlm(model, this_df, groups = this_df['sub_id'], missing = 'drop')
    mdf = md.fit()
    print(mdf.summary())           

# %% Topo | LME - ST ME - need fix

feature = "density_20"

model = f"{feature} ~ C(subtype, Treatment('HS'))" 

this_df = mean_df[[
    'sub_id', 'subtype', 'freq_range',
    'density_20', 'density_90_hs_global','density_90_hs_chan', 
    'ptp_20', 'ptp_90_hs_global', 'ptp_90_hs_chan',
    'uslope_20', 'uslope_90_hs_global', 'uslope_90_hs_chan', 
    'dslope_20', 'dslope_90_hs_global', 'dslope_90_hs_chan', 
    'channel', 'sleepiness', 'rt_go',
    'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
     ]].groupby(
         ['sub_id', 'subtype', 'freq_range', 'channel'], as_index = False).mean()

         
fig, ax = plt.subplots(
    nrows = 2, ncols = 2, figsize = (8, 5))

for i_f, band in enumerate(['delta', 'theta']):
    for i_s, subtype in enumerate(['N1', 'HI']) :
        temp_tval = []; temp_pval = []; chan_l = []
        for chan in channels :
            subdf = this_df[
                ['sub_id', 'subtype', 'channel', feature]
                ].loc[
                (this_df.channel == chan)
                & (this_df.subtype.isin([subtype, 'HS']))
                & (this_df.freq_range == band)
                ].dropna()
            try:
                md = smf.mixedlm(model, subdf, groups=subdf['sub_id'], missing='drop')
                mdf = md.fit()
                temp_tval.append(mdf.tvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"])
                temp_pval.append(mdf.pvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"])
            except np.linalg.LinAlgError:
                print(f"Singular matrix error for channel {chan}. Skipping this channel.")
                temp_tval.append(np.nan)
                temp_pval.append(np.nan)
            chan_l.append(chan)
            
        if np.any(np.isnan(temp_tval)) :
            temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
             
        _, corrected_pval = fdrcorrection(temp_pval)
        
        divider = make_axes_locatable(ax[i_f][i_s])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = temp_tval,
            pos = epochs.info,
            axes = ax[i_f][i_s],
            contours = 3,
            mask = np.asarray(temp_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=8),
            cmap = "viridis",
            # vlim = (-4, 4)
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
        
        ax[i_s][i_f].set_title(f"{subtype} {band} vs HS", fontweight = "bold")
        
    title = """ Topographies of <Subtype Main Effect> on <Slow Wave Density> """
    fig_text(
       0.07, .94,
       title,
       fontsize=15,
       ha='left', va='center',
       color="k", font=font,
       highlight_textprops=[
          {'font': bold_font},
          {'font': bold_font},
       ],
       fig=fig
    )

# %% Topo | LME - Explo MS ME

model = "density ~ C(subtype) * C(mindstate)" 

fig, ax = plt.subplots(
    nrows = 1, ncols = 1, figsize = (6, 6), layout = 'tight')

temp_tval = []; temp_pval = []; chan_l = []
for chan in channels :
    subdf = mean_df[
        ['sub_id', 'subtype', 'mindstate', 'channel', 'density', 'sw_thresh']
        ].loc[
        (mean_df.channel == chan)
        ].dropna()
    md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
    mdf = md.fit()
    temp_tval.append(mdf.tvalues['C(subtype)[T.N1]'])
    temp_pval.append(mdf.pvalues['C(subtype)[T.N1]'])
    chan_l.append(chan)
    
if np.any(np.isnan(temp_tval)) :
    temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
     
_, corrected_pval = fdrcorrection(temp_pval)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "5%", pad=0.05)
im, cm = mne.viz.plot_topomap(
    data = temp_tval,
    pos = epochs.info,
    axes = ax,
    contours = 3,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6),
    cmap = "viridis",
    vlim = (-4, 4)
    )
fig.colorbar(im, cax = cax, orientation = 'vertical')

ax.set_title("NT1 vs CTL", fontweight = "bold")

title = """
Topographies of <Subtype Main Effect>
on <Slow Wave Density>
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
   ],
   fig=fig
)

# %% Topo | LME - Subtype effect

model = "density ~ C(subtype) * C(mindstate)" 

fig, ax = plt.subplots(
    nrows = 1, ncols = len(mindstates), figsize = (18, 5), layout = 'tight')

for i, mindstate in enumerate(mindstates):
    temp_tval = []; temp_pval = []; chan_l = []
    cond_df = mean_df.loc[mean_df.mindstate == mindstate]
    for chan in channels :
        subdf = cond_df[
            ['sub_id', 'subtype', 'mindstate', 'channel', 'density', 'sw_thresh']
            ].loc[
            (cond_df.channel == chan)
            ].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
        mdf = md.fit()
        temp_tval.append(mdf.tvalues['C(subtype)[T.N1]'])
        temp_pval.append(mdf.pvalues['C(subtype)[T.N1]'])
        chan_l.append(chan)
        
    if np.any(np.isnan(temp_tval)) :
        temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
         
    _, corrected_pval = fdrcorrection(temp_pval)
    
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval,
        pos = epochs.info,
        axes = ax[i],
        contours = 3,
        mask = np.asarray(temp_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=6),
        cmap = "viridis",
        vlim = (-4, 4)
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')

    ax[i].set_title(f"{mindstate}", fontweight = "bold")

title = """
Topographies of mindstate <Subtype Main Effect> on <Slow Wave Density> according to the <Mindstate>
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

# %% Topo | LME - Behavior

interest = "false_alarms"
predict = "density_20"
model = f"{interest} ~ {predict}" 

# fig, ax = plt.subplots(
#     nrows = 1, ncols = len(mindstates), figsize = (18, 8), layout = 'tight')

fig, ax = plt.subplots(
    nrows = 1, ncols = 3, figsize = (12, 4), layout = 'tight')

for i, subtype in enumerate(subtypes) :
    temp_tval = []; temp_pval = []; chan_l = []
    for chan in channels :
        subdf = mean_df[
            ['sub_id', 'subtype', 'channel', predict, interest]
            ].loc[
            (mean_df.channel == chan) 
            & (mean_df.subtype == subtype)
            ].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
        mdf = md.fit()
        temp_tval.append(mdf.tvalues[predict])
        temp_pval.append(mdf.pvalues[predict])
        chan_l.append(chan)
        
    if np.any(np.isnan(temp_tval)):
        for t, tval in enumerate(temp_tval) :
            if np.isnan(tval):
                temp_tval[t] = np.nanmean(temp_tval)
         
    # _, corrected_pval = fdrcorrection(temp_pval)
    
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval,
        pos = epochs.info,
        axes = ax[i],
        contours = 3,
        mask = np.asarray(temp_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=6),
        cmap = "viridis",
        # vlim = (-4, 4)
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    
    ax[i].set_title(f"{interest}_{subtype}", fontweight = "bold")

# title = f"""
# Topographies of mindstate <Subtype Main Effect> on <Slow Wave Density> according to the <Mindstate> for <{subtype}>
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
#       {'font': bold_font},
#    ],
#    fig=fig
# )

# %% Roids

# import pandas as pd
# import numpy as np
# import statsmodels.formula.api as smf

def run_mixedlm(data, channel, interest):
    """Run a mixed linear model for a given channel."""
    subdf = data.loc[data.channel == channel].dropna()
    md = smf.mixedlm("density ~ n_session*C(difficulty)", subdf, groups=subdf['sub_id'])
    mdf = md.fit()
    return (mdf.pvalues[interest], 
            mdf.tvalues[interest])

def permute_and_run(data, interest, num_permutations=100):
    """Permute the difficulty column and run models to get p-values distribution."""
    original_pvals, original_tvals = [], []
    permuted_stats = []

    for chan in data.channel.unique():
        pval, tval = run_mixedlm(data, chan, interest)
        original_pvals.append(pval)
        original_tvals.append(tval)

    for _ in range(num_permutations):
        shuffled_data = data.copy()
        shuffled_data['difficulty'] = np.random.permutation(shuffled_data['difficulty'].values)
        permuted_pvals = [run_mixedlm(shuffled_data, chan, interest)[0] for chan in shuffled_data.channel.unique()]
        # For simplicity, we're not identifying clusters in permuted data here, but you should for actual correction
        permuted_stats.extend(permuted_pvals)  # Adjust as necessary for cluster-level stats
    
    return original_pvals, original_tvals, permuted_stats

def identify_significant_clusters(
        original_pvals, 
        original_tvals, 
        channels, 
        permuted_stats, 
        threshold=0.05
        ):
    """
    Identify significant clusters comparing to null distribution of permuted stats.
    
    original_pvals: List of p-values from the original analysis.
    original_tvals: List of t-values corresponding to each channel in the original analysis.
    channels: List of channels.
    permuted_stats: List of cluster-level statistics from permuted data.
    threshold: Significance threshold for p-values.
    
    return: A mask indicating significant channels based on cluster permutation correction.
    """
    # Step 1: Identify original clusters based on p-value threshold
    clusters = []
    current_cluster = []
    for i, pval in enumerate(original_pvals):
        if pval < threshold:
            if not current_cluster or i == current_cluster[-1] + 1:  # Adjacent channels
                current_cluster.append(i)
            else:
                clusters.append(current_cluster)
                current_cluster = [i]
    if current_cluster:  # Add the last cluster if it exists
        clusters.append(current_cluster)
    
    # Step 2: Calculate cluster-level statistic for original clusters (sum of t-values)
    """For each cluster, we calculate a "cluster-level statistic." 
    This statistic aims to represent the cumulative evidence of an effect within the cluster. 
    By summing the t-values of all channels within a cluster, 
    we get a single metric that captures the overall strength and direction 
    of the effect across the cluster. 
    This sum is what we refer to as the cluster-level statistic."""
    
    cluster_stats = [sum(original_tvals[idx] for idx in cluster) for cluster in clusters]
    
    # Step 3: Compare original cluster stats to null distribution
    """For each original cluster statistic, 
    we compare it to the null distribution to compute a p-value. 
    This p-value tells us the probability of observing a cluster statistic 
    as extreme as the one we calculated if the null hypothesis were true 
    (i.e., there is no real effect, and any observed effect is due to chance)."""
    
    significant_clusters = []
    for stat in cluster_stats:
        # Compute p-value for this cluster's stat against the null distribution
        p_value = (np.sum(np.array(permuted_stats) >= stat) + 1) / (len(permuted_stats) + 1)
        if p_value <= threshold:
            significant_clusters.append(stat)
    
    # Step 4: Create a mask for visualization
    significant_mask = np.zeros(len(channels), dtype=bool)
    for cluster in clusters:
        if sum(original_tvals[idx] for idx in cluster) in significant_clusters:
            significant_mask[cluster] = True
            
    return significant_mask

def visualize_clusters(tvals, channels, significant_mask, info):
    """Visualize significant clusters using topomap."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    # pos = np.array([np.random.rand(2) for _ in channels])  # Update with actual positions

    im, cm = mne.viz.plot_topomap(
        data=np.array(tvals),
        pos=info,
        mask=significant_mask,
        axes=ax,
        show=False,
        contours=2,
        # sensors='k.',
        mask_params=dict(
            marker='o', 
            markerfacecolor='w', 
            markeredgecolor='k', 
            linewidth=0, 
            markersize=10
            ),
        cmap = "viridis"
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    
    ax.set_title("Interaction Effect", fontweight = "bold")

    fig.suptitle("T-values, Cluster Permutation (100) Corrected", 
             fontsize = "xx-large", 
             fontweight = "bold")
    plt.show()
    plt.savefig(f"{swDataPath}{os.sep}Figs{os.sep}SW_Dens_interaction_session_difficulty_clustercorr.png")

# Example usage - This assumes permuted_stats, original_pvals, original_tvals, and channels are already defined

original_pvals, original_tvals, permuted_stats = permute_and_run(
    mean_df, 
    'n_session:C(difficulty)[T.HARD]', 
    100
    )
significant_mask = identify_significant_clusters(
    original_pvals, 
    original_tvals, 
    list(mean_df.channel.unique()), 
    permuted_stats
    )
visualize_clusters(
    original_tvals, 
    mean_df.channel.unique(), 
    significant_mask, epochs.info
    )




