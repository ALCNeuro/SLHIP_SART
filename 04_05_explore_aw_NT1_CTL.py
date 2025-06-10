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
from scipy.stats import sem

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
figs_path = os.path.join(wavesPath, "figs", "NT1_CTL")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

channels = np.array(config.eeg_channels)
palette = ["#8d99ae", "#d00000"]


# %% Compute DF  

if os.path.exists(os.path.join(wavesPath, "figs", "df_aw.csv")) and os.path.exists(os.path.join(wavesPath, "figs", "df_aw_mean.csv")):
    df_aw = pd.read_csv(os.path.join(wavesPath, "figs", "df_aw.csv"))
    mean_df = pd.read_csv(os.path.join(wavesPath, "figs", "df_aw_mean.csv"))

else : 
        
    slope_range = [0.125, 2] # in uV/ms
    positive_amp = [75] # in uV
    amplitude_max = 150
    
    # df_aw = pd.concat([pd.read_csv(file) for file in allwaves_files])
    
    df_list = []
    
    features = [
        "sub_id", "subtype", "daytime",
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
            & (df['PTP'] > 5)
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
                    bigdic['daytime'].append(session)
                    
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
            
    df_aw.to_csv(os.path.join(wavesPath, "figs", "df_aw.csv"))
    mean_df.to_csv(os.path.join(wavesPath, "figs", "df_aw_mean.csv"))
    
# %% df density ptp bin

slope_range = [0.125, 2] # in uV/ms
positive_amp = [75] # in uV
amplitude_max = 150

ptp_bins = np.arange(4, 102, 2)
nblock = 4
nprobes = 10

# df_aw = pd.concat([pd.read_csv(file) for file in allwaves_files])

df_list = []

features = ["sub_id", "subtype", "density", "ptp_bin", "channel"]

for file in allwaves_files :
    
    bigdic = {feat : [] for feat in features}

    sub_id = file.split('all_waves/')[-1].split('.')[0]
    
    print(f"...Processing {sub_id}...")
    
    session = sub_id[-2:]
    sub_id = sub_id[:-3]
    subtype = sub_id[:2]

    df = pd.read_csv(file)
    del df['Unnamed: 0']
    
    df = df.loc[
        (df['PTP'] < 150) 
        & (df['PTP'] > 5)
        & (df['pos_amp_peak'] < 75)]
    df = df.loc[
        (df["pos_halfway_period"] <= slope_range[1])
        & (df["pos_halfway_period"] >= slope_range[0])
        ]
    
    for channel in channels :
        this_df = df.loc[
            (df.channel == channel)
            ]
        for i, ptpbin in enumerate(ptp_bins) :
            if ptpbin < ptp_bins.max() :
            
                temp_df = this_df.loc[
                    (this_df["PTP"] > ptpbin) 
                    & (this_df["PTP"] < ptp_bins[i+1])
                    ]
                n_waves = temp_df.shape[0]/(nblock*nprobes)
                bigdic['density'].append(n_waves)
                bigdic['sub_id'].append(sub_id)
                bigdic['subtype'].append(subtype)
                bigdic['channel'].append(channel)
                # bigdic['mindstate'].append(mindstate)
                bigdic['ptp_bin'].append(ptpbin)
        
    df_feature = pd.DataFrame.from_dict(bigdic)
    # df_feature.to_csv(this_df_swfeat_savepath)
    df_list.append(df_feature)
    
df_density = pd.concat(df_list)
 
mean_df_density = df_density[[
    'sub_id', 'subtype', 'channel', 'density', 'ptp_bin']
    ].groupby(
           ['sub_id', 'subtype', 'channel', 'ptp_bin'], 
           as_index = False).mean()
        
# df_aw.to_csv(os.path.join(wavesPath, "figs", "df_aw.csv"))
# mean_df.to_csv(os.path.join(wavesPath, "figs", "df_aw_mean.csv"))

# %% density plot

thresh_90 = {group:np.percentile(df_aw.ptp.loc[df_aw.subtype==group].dropna(), 90) for group in ['HS', 'N1']}
thresh_80 = {group:np.percentile(df_aw.ptp.loc[df_aw.subtype==group].dropna(), 80) for group in ['HS', 'N1']}
thresh_70 = {group:np.percentile(df_aw.ptp.loc[df_aw.subtype==group].dropna(), 70) for group in ['HS', 'N1']}
thresh_60 = {group:np.percentile(df_aw.ptp.loc[df_aw.subtype==group].dropna(), 70) for group in ['HS', 'N1']}

c_90 = ["#8d99ae", "#d00000"]

plot_se = False   
        
fig, ax = plt.subplots(figsize=(6,5))
sns.lineplot(
    data = mean_df_density, 
    x = 'ptp_bin', 
    y = 'density', 
    hue = 'subtype',
    hue_order = ['HS', "N1"],
    palette = palette,
    ax = ax,
    legend = False,
    linewidth = 3,
    alpha = .9
    )
for i, st in enumerate(['HS', 'N1']) :
    plt.axvline(
        x = thresh_90[st],
        ymin = 0,
        ymax = 1.5,
        label = f"90th P {st}",
        ls = "-",
        color = c_90[i]
        )
    plt.axvline(
        x = thresh_80[st],
        ymin = 0,
        ymax = 1.5,
        label = f"80th P {st}",
        ls = "--",
        color = c_90[i]
        )
    plt.axvline(
        x = thresh_70[st],
        ymin = 0,
        ymax = 1.5,
        label = f"70th P {st}",
        ls = "-.",
        color = c_90[i]
        )
    plt.axvline(
        x = thresh_60[st],
        ymin = 0,
        ymax = 1.5,
        label = f"60th P {st}",
        ls = ":",
        color = c_90[i]
        )

ax.set_xticks(np.linspace(0, 80, 5), np.linspace(0, 80, 5).astype(int), font = font, fontsize = 16)
ax.set_xlim(5, 80)
ax.set_yticks(np.linspace(0, 1.5, 3), np.linspace(0, 1.5, 3), font = font, fontsize = 16)
ax.set_ylim(0,1.5)
ax.set_ylabel("SW Density (nSW / Probe)", font = bold_font, fontsize = 24)
ax.set_xlabel("Amplitude all waves", font = bold_font, fontsize = 24)
ax.legend()

sns.despine()   
fig.tight_layout()
plt.savefig(os.path.join(figs_path, "lineplot_swdensity_ptpbin.png"), dpi = 200)
    

# %% 

temp_df = df_aw[[
    'sub_id', 'subtype', 'channel',  'mindstate', 
    'density', 'ptp', 'uslope', 'dslope', 'frequency', 'sleepiness']
    ].groupby(
           ['sub_id', 'subtype', 'channel', 'mindstate'], 
           as_index = False).mean()

fig, ax = plt.subplots()
for subtype in ['HS', 'N1']:
    ax.hist(
        temp_df.ptp.loc[temp_df.subtype == subtype], 
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
y = 'density'
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

# %% MS - Density

mindstates = ['ON', 'MW', 'MB', 'HALLU', 'FORGOT']
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
        cmap = "Purples"
        )
    if i_ms == 4 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    # ax[0][i_sess].set_title(f"EASY - S{n_sess}")
    ax[0][i_ms].set_title(f"{subtypes[0]} - {mindstate}", font = bold_font)
    
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
        cmap = "Purples"
        )
    # ax[1][i_sess].set_title(f"HARD - S{n_sess}")
    ax[1][i_ms].set_title(f"{subtypes[1]} - {mindstate}", font = bold_font)
    if i_ms == 4 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    plt.show(block = False)

# %% Topo | LME - MS > ON effect

interest = 'uslope'

model = f"{interest} ~ C(subtype, Treatment('HS'))" 

temp_tval = []; temp_pval = []; chan_l = []
cond_df = df_aw.loc[df_aw.subtype.isin(['HS', "N1"])]
for chan in channels :
    subdf = cond_df[
        ['sub_id', 'subtype', 'mindstate', 'channel', f'{interest}']
        ].loc[(cond_df.channel == chan)].dropna()
    md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
    mdf = md.fit()
    temp_tval.append(mdf.tvalues["C(subtype, Treatment('HS'))[T.N1]"])
    temp_pval.append(mdf.pvalues["C(subtype, Treatment('HS'))[T.N1]"])
    chan_l.append(chan)
    
if np.any(np.isnan(temp_tval)) :
    temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
     
_, corrected_pval = fdrcorrection(temp_pval)
fig, ax = plt.subplots(
    nrows = 1, ncols = 1, figsize = (4, 4), layout = 'tight')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "5%", pad=0.05)
im, cm = mne.viz.plot_topomap(
    data = temp_tval,
    pos = epochs.info,
    axes = ax,
    contours = 3,
    mask = np.asarray(corrected_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=8),
    cmap = "coolwarm",
    # vlim = (-4, 4)
    )
fig.colorbar(im, cax = cax, orientation = 'vertical')

ax.set_title(f"{interest} NT1 > CTL", fontweight = "bold")
plt.savefig(os.path.join(figs_path, f"corrected_{interest}_AllWaves_ME_Group.png"), dpi = 300)

# %% Topo | LME - MS > ON effect

compa_df = df_aw.loc[df_aw.subtype=="N1"]
# compa_df = compa_df.drop(columns=["n_epoch", "nprobe"]).groupby(
#     ["sub_id", "subtype", "channel", "mindstate", "nblock", "daytime"], as_index=False
#     ).mean()

features = ['density', 'ptp', 'uslope', 'dslope']

these_ms = ["MW", "MB", "HALLU", "FORGOT"]

fig, axs = plt.subplots(
    nrows = len(mindstates[1:]), ncols = len(features), figsize = (16, 16))

for i, mindstate in enumerate(these_ms):
    ax = axs[i]
    for j, feat in enumerate(features) :
        
        model = f"{feat} ~ C(mindstate, Treatment('ON'))" 
        
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = compa_df.loc[compa_df.mindstate.isin(['ON', mindstate])]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'subtype', 'mindstate', 'channel', feat]
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
            mdf = md.fit()
            temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('ON'))[T.{mindstate}]"])
            temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('ON'))[T.{mindstate}]"])
            chan_l.append(chan)
            
        if np.any(np.isnan(temp_tval)) :
            temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
             
        # _, corrected_pval = fdrcorrection(temp_pval)
        
        divider = make_axes_locatable(ax[j])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = temp_tval,
            pos = epochs.info,
            axes = ax[j],
            contours = 3,
            mask = np.asarray(temp_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=6),
            cmap = "viridis",
            # vlim = (-4, 4)
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[j].set_title(f"{feat} {mindstate} > ON", fontweight = "bold", fontsize = 12)
fig.tight_layout()
plt.savefig(
    os.path.join(
        figs_path, "AW_NT1_MS_fulldf_vson.png"), dpi = 300)
    
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
        ))
    if len(epochs_files) < 1 : print("carefully inspect why this file is missing"); continue
    epochs_files = epochs_files[0]
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
           ['sub_id', 'subtype', 'channel', 'mindstate', 'frequency_range'], 
           as_index = False).mean()
        
df_freq_aw.to_csv(os.path.join(figs_path, "df_aw_theta_delta.csv"))

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
subtypes = ['HS', 'N1']

feature = 'uslope'

fig, ax = plt.subplots(
    nrows = 2, 
    ncols = 2,
    figsize = (6,4),
    layout = 'tight'
    )

for i_fr, frequency_range in enumerate(frequency_ranges) :
    
    list_values = []
    for subtype in subtypes :   
        for channel in channels :
            list_values.append(this_mean_df[feature].loc[
                (this_mean_df["frequency_range"] == frequency_range)
                & (this_mean_df["subtype"] == subtype)
                & (this_mean_df["channel"] == channel)
                ].mean())
    vmin = min(list_values)
    vmax = max(list_values)
    
    list_hs = []
    list_n1 = []      
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
    
    # if i_fr == 1 :
    divider = make_axes_locatable(ax[i_fr][0])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_hs,
        epochs.info,
        axes = ax[i_fr][0],
        size = 2,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "Purples"
        )
    # if i_fr == 1 :
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    ax[i_fr][0].set_title(f"{subtypes[0]} - {frequency_range}", font = bold_font)
 
    
    # if i_fr == 1 :
    divider = make_axes_locatable(ax[i_fr][1])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_n1,
        epochs.info,
        axes = ax[i_fr][1],
        size = 2,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "Purples"
        )

    ax[i_fr][1].set_title(f"{subtypes[1]} - {frequency_range}", font = bold_font)
    # if i_ms == 1 :
    fig.colorbar(im, cax = cax, orientation = 'vertical')

    plt.show(block = False)

fig.tight_layout()

# %% Topo | LME - ST ∆ > HS effect

interest = 'density'

model = f"{interest} ~ C(subtype, Treatment('HS'))" 

# fig, ax = plt.subplots(
#     nrows = 1, ncols = len(mindstates), figsize = (18, 8), layout = 'tight')

fig, ax = plt.subplots(
    nrows = 1, ncols = len(subtypes[1:]), figsize = (12, 4), layout = 'tight')
for i, subtype in enumerate(["N1", "HI"]):
    temp_tval = []; temp_pval = []; chan_l = []
    for chan in channels :
        subdf = cond_df[
            ['sub_id', 'subtype', 'channel', f'{interest}']
            ].loc[(cond_df.channel == chan)].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'drop')
        mdf = md.fit()
        temp_tval.append(mdf.tvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"])
        temp_pval.append(mdf.pvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"])
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
        cmap = "coolwarm",
        # vlim = (-4, 4)
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')

    ax[i].set_title(f"{subtype}", fontweight = "bold")

# title = f"""Topographies of <Slow Wave {interest}> according to the <Mindstate> VS <ON>"""
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
    