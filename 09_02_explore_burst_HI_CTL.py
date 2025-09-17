#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/07/25

@author: arthur.lecoz

09_02_explore_burst_NT1_CTL.py
"""
# %%% Paths & Packages

import SLHIP_config_ALC as config
import mne
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import false_discovery_control

from matplotlib.font_manager import FontProperties
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

cleanDataPath = config.cleanDataPath
burstPath = config.burstPath

feature_path = os.path.join(burstPath, "features")
figs_path = os.path.join(burstPath, "figs")

# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

channels = np.array(config.eeg_channels)
palette = ["#8d99ae", "#d00000", "#ffb703"]

# %% df load

#### DF
df = pd.read_csv(os.path.join(
    burstPath, 
    "features", 
    "All_Bursts_features.csv"
    ))
del df['Unnamed: 0']

# %% Remove IH from df

epochs = mne.read_epochs(glob(f"{cleanDataPath}/epochs_probes/*.fif")[0])
epochs.drop_channels(['TP9', 'TP10', 'VEOG', 'HEOG', 'ECG', 'RESP'])
neighbours = config.prepare_neighbours_from_layout(epochs.info, ch_type='eeg')

df = df.loc[
    (~df.mindstate.isin(['DISTRACTED', 'MISS']))
    & (df.subtype != 'N1')
    ]

mindstates = ['ON', 'MW', 'MB', 'HALLU', 'FORGOT']
subtypes = ['HS', 'N1']
channels = config.eeg_channels

mean_df = df.copy().drop(columns = ["daytime", "n_probe"])
mean_df = mean_df.groupby(
           ['sub_id', 'subtype', 'channel', 'mindstate', 'burst_type'], 
           as_index = False).mean()
           
# %% Topo | Density | HS, NT1

feature = 'density'
burst_types = ['Alpha', 'Theta']

fig, ax = plt.subplots(
    nrows = 2, 
    ncols = len(burst_types),
    figsize = (6,4),
    )

for i_bt, burst_type in enumerate(burst_types) :
    ax_hs = ax[0]
    ax_nt = ax[1]

    list_values = []
    for i_ms, mindstate in enumerate(mindstates) :
        for subtype in subtypes :   
            for channel in channels :
                list_values.append(mean_df[feature].loc[
                    (mean_df["subtype"] == subtype)
                    & (mean_df["channel"] == channel)
                    & (mean_df["burst_type"] == burst_type)
                    ].mean())
    vmin = min(list_values)
    vmax = max(list_values)
      
    list_hs = []
    list_n1 = []        
    for channel in channels :
        list_hs.append(mean_df[feature].loc[
            (mean_df["subtype"] == "HS")
            & (mean_df["channel"] == channel)
            & (mean_df["burst_type"] == burst_type)
            ].mean())
        list_n1.append(mean_df[feature].loc[
            (mean_df["subtype"] == "HI")
            & (mean_df["channel"] == channel)
            & (mean_df["burst_type"] == burst_type)
            ].mean())
    
    divider = make_axes_locatable(ax_hs[i_bt])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_hs,
        epochs.info,
        axes = ax_hs[i_bt],
        size = 2,
        # names = channels,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "Purples"
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    ax_hs[i_bt].set_title(f"CTL - {burst_type}", font = bold_font, fontsize = 12)
    
    divider = make_axes_locatable(ax_nt[i_bt])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        list_n1,
        epochs.info,
        axes = ax_nt[i_bt],
        size = 2,
        # names = channels,
        show = False,
        contours = 2,
        vlim = (vmin, vmax),
        cmap = "Purples"
        )
    ax_nt[i_bt].set_title(f"HI - {burst_type}", font = bold_font, fontsize = 12)
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    plt.show(block = False)
    fig.tight_layout()
    
    figsavename = os.path.join(
        figs_path, 'HI_CTL', f'topo_{feature}_subtypes.png'
        )
    plt.savefig(figsavename, dpi = 300)
    
# %% Topo | LME - Subtype ME

burst_types = ['Alpha', 'Theta']

feature = "density"

fig, ax = plt.subplots(
    nrows = 1, ncols = len(burst_types), figsize = (6, 3)
    )

for i_bt, burst_type in enumerate(burst_types):
    model = f"{feature} ~ C(subtype, Treatment('HS'))" 
    temp_tval = []; temp_pval = []; chan_l = []
    for chan in channels :
        subdf = df[
            ['sub_id', 'subtype', 'channel', feature]
            ].loc[
            (df.channel == chan)
            & (df.subtype.isin(['HI', 'HS']))
            & (df.burst_type == burst_type)
            ].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'drop')
        mdf = md.fit()
        temp_tval.append(mdf.tvalues["C(subtype, Treatment('HS'))[T.HI]"])
        temp_pval.append(mdf.pvalues["C(subtype, Treatment('HS'))[T.HI]"])
        chan_l.append(chan)
        
    if np.any(np.isnan(temp_tval)) :
        temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
    
    if i_bt == len(burst_types) - 1 :
        divider = make_axes_locatable(ax[i_bt])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval,
        pos = epochs.info,
        axes = ax[i_bt],
        contours = 3,
        mask = np.asarray(temp_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=8),
        cmap = "coolwarm",
        vlim = (-2.5, 2.5),
        size = 2.5
        )
    if i_bt == len(burst_types) - 1 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
    ax[i_bt].set_title(f"{burst_type} HI > HS", fontweight = "bold", fontsize = 12)
    
    fig.tight_layout()
        
plt.savefig(os.path.join(
    figs_path, 'HI_CTL', f'LME_topo_{feature}_ME_group_averagedblocks.png'
    ), dpi = 300)

# %% Diff MS within GROUP

vlims = {
    "Alpha" : {
        "HS" : (-3, 3),
        "HI" : (-2.5, 2.5)
        },
    "Theta" : {
        "HS" : (-3, 3),
        "HI" : (-5, 5)
        }
    }

kindaburst = "Theta"

this_df = df.loc[df.burst_type==kindaburst]

interest = 'density'
contrasts = [("ON", "MW"), ("ON", "MB"), ("ON", "HALLU"), ("ON", "FORGOT")]

subtypes = ['HS', 'HI']

for i_s, subtype in enumerate(subtypes) :
    fig, this_ax = plt.subplots(
        nrows = 1, 
        ncols = len(contrasts), # MW > ON ; MB > ON ; MW > MB (for start)
        figsize = (10,2.5),
        )
    for i_c, contrast in enumerate(contrasts) :
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = this_df.loc[
            (this_df.mindstate.isin(contrast))
            & (this_df.subtype == subtype)
            ]
        
        model = f"{interest} ~ C(mindstate, Treatment('{contrast[0]}'))" 
    
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'subtype', 'mindstate', 'channel', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'drop')
            mdf = md.fit()
            
            temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('{contrast[0]}'))[T.{contrast[1]}]"])
            temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('{contrast[0]}'))[T.{contrast[1]}]"])
            chan_l.append(chan)
            
        if np.any(np.isnan(temp_tval)) :
            for pos in np.where(np.isnan(temp_tval))[0] :
                temp_tval[pos] = np.nanmean(temp_tval)
             
        # _, corrected_pval = fdrcorrection(temp_pval)
        
        if i_c == len(contrasts) - 1 :
            divider = make_axes_locatable(this_ax[i_c])
            cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = temp_tval,
            pos = epochs.info,
            axes = this_ax[i_c],
            contours = 3,
            mask = np.asarray(temp_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=8),
            cmap = "coolwarm",
            vlim = vlims[kindaburst][subtype]
            )
        if i_c == len(contrasts) - 1 :
            fig.colorbar(im, cax = cax, orientation = 'vertical')
        # this_ax[i_c].set_title(f"{contrast[1]} > {contrast[0]}", fontweight = "bold")

    # fig.suptitle(f"{interest}", font = bold_font, fontsize = 24)
    fig.tight_layout()
    figsavename = os.path.join(
        figs_path, 'HI_CTL', f'LME_topo_{kindaburst}_{interest}_ME_MS_{subtype}.png'
        )
    plt.savefig(figsavename, dpi = 300)

# %% Diff MS within GROUP vs hallu

vlims = {
    "HS" : (-3, 3),
    "N1" : (-4, 4)
    }
this_df = df.loc[df.burst_type=="Theta"]

interest = 'density'
contrasts = [("HALLU", "ON"), ("HALLU", "MW"), ("HALLU", "MB"), ("HALLU", "FORGOT")]

subtypes = ['HS', 'N1']

for i_s, subtype in enumerate(subtypes) :
    fig, this_ax = plt.subplots(
        nrows = 1, 
        ncols = len(contrasts), # MW > ON ; MB > ON ; MW > MB (for start)
        figsize = (10,2.5),
        )
    for i_c, contrast in enumerate(contrasts) :
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = this_df.loc[
            (this_df.mindstate.isin(contrast))
            & (this_df.subtype == subtype)
            ]
        
        model = f"{interest} ~ C(mindstate, Treatment('{contrast[0]}'))" 
    
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'subtype', 'mindstate', 'channel', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'drop')
            mdf = md.fit()
            
            if f"C(mindstate, Treatment('{contrast[0]}'))[T.{contrast[1]}]" not in mdf.tvalues.index :
                temp_tval.append(np.nan)
                temp_pval.append(1)
                chan_l.append(chan)
            else : 
                temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('{contrast[0]}'))[T.{contrast[1]}]"])
                temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('{contrast[0]}'))[T.{contrast[1]}]"])
                chan_l.append(chan)
            
        if np.any(np.isnan(temp_tval)) :
            for pos in np.where(np.isnan(temp_tval))[0] :
                temp_tval[pos] = np.nanmean(temp_tval)
             
        # _, corrected_pval = fdrcorrection(temp_pval)
        
        if i_c == len(contrasts) - 1 :
            divider = make_axes_locatable(this_ax[i_c])
            cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = -np.asarray(temp_tval),
            pos = epochs.info,
            axes = this_ax[i_c],
            contours = 3,
            mask = np.asarray(temp_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=8),
            cmap = "coolwarm",
            vlim = vlims[subtype]
            )
        if i_c == len(contrasts) - 1 :
            fig.colorbar(im, cax = cax, orientation = 'vertical')
            
        this_ax[i_c].set_title(f"{contrast[0]} > {contrast[1]}", fontweight = "bold")

# fig.suptitle(f"{interest}", font = bold_font, fontsize = 24)
    fig.tight_layout()
    figsavename = os.path.join(
        figs_path, 'NT1_CTL', f'LME_topo_{feature}_ME_MS_{subtype}_VS_HALLU.png'
        )
    plt.savefig(figsavename, dpi = 300)

# %% Topo | LME - MS behav

kindaburst = "Alpha"
behav_interest = "miss"
burst_interest = "density"
group_oi = "HI"

vlims = {
    "false_alarms" : (-4, 4),
    "miss" : (-15, 15),
    "rt_go" : (-4.5, 4.5)
    }

compa_df = df.loc[
    (df.subtype==group_oi)
    & (df.burst_type == kindaburst)
    ]

model = f"{behav_interest} ~ {burst_interest}" 

fig, ax = plt.subplots(
    nrows = 1, 
    ncols = 1, 
    figsize = (4, 4)
    )

temp_tval = []; temp_pval = []; chan_l = []
for chan in channels :
    subdf = compa_df[
        ['sub_id', 'subtype', 'channel', behav_interest, burst_interest]
        ].loc[(compa_df.channel == chan)].dropna()
    md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
    try :
        mdf = md.fit()
        temp_tval.append(mdf.tvalues[burst_interest])
        temp_pval.append(mdf.pvalues[burst_interest])
    except :
        temp_tval.append(np.nan)
        temp_pval.append(1)
    
    chan_l.append(chan)
    
if np.any(np.isnan(temp_tval)) :
    temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
     
# _, corrected_pval = fdrcorrection(temp_pval)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "5%", pad=0.05)
im, cm = mne.viz.plot_topomap(
    data = temp_tval,
    pos = epochs.info,
    axes = ax,
    contours = 2,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=8),
    cmap = "coolwarm",
    vlim = vlims[behav_interest]
    )
fig.colorbar(im, cax = cax, orientation = 'vertical')

ax.set_title(f"{behav_interest} ~ {burst_interest}", fontweight = "bold", fontsize = 12)
fig.tight_layout()
plt.savefig(os.path.join(
    figs_path, 'HI_CTL', f'LME_{behav_interest}_burst_{kindaburst}_{burst_interest}_in_{group_oi}.png'
    ), dpi = 300)   

# %% Corrected | ME Group Burst

info = epochs.info  # or info from epochs
neighbours = config.prepare_neighbours_from_layout(info, ch_type='eeg')

vlims = (-2.5, 2.5)
clus_alpha = 0.05        # uncorrected threshold for candidate electrodes
montecarlo_alpha = 0.05  # threshold for permutation cluster-level test
num_permutations = 100    # adjust as needed
min_cluster_size = 2     # keep clusters with at least 2 channels

burst_type = 'Theta'

feature = "density"

save_fname = os.path.join(
    figs_path,
    "HI_CTL",
    f"CPerm_{num_permutations}_{burst_type}_{feature}_ME_Group.pkl"
    )
savepath = os.path.join(
    figs_path,
    "HI_CTL",
    f"CPerm_{num_permutations}_{burst_type}_{feature}_ME_Group.png"
    )

subdf = df[
    ['sub_id', 'subtype', 'channel', feature]
    ].loc[
    (df.subtype.isin(['HI', 'HS']))
    & (df.burst_type == burst_type)
    ].dropna()

if os.path.exists(save_fname) :
    big_dic = pd.read_pickle(save_fname)

    orig_tvals           = big_dic['orig_tvals']
    channels             = big_dic['channels']
    significant_clusters = big_dic['significant_clusters']
    
else : 

    model = f"{feature} ~ C(subtype, Treatment('HS'))" 
    interest = "C(subtype, Treatment('HS'))[T.HI]"
    to_permute = "subtype"
    
    clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, orig_pvals, orig_tvals = config.permute_and_cluster(
        subdf,
        model, 
        interest,
        to_permute,
        num_permutations,
        neighbours,     
        clus_alpha,
        min_cluster_size,
        channels
        )
    
    # Determine significant clusters based on permutation statistics.
    significant_clusters = config.identify_significant_clusters(
        clusters_pos, 
        clusters_neg, 
        perm_stats_pos, 
        perm_stats_neg, 
        montecarlo_alpha,
        num_permutations
        )
    
    # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # Save everything for later use (before you do the plotting!)
    out = {
        'clusters_pos':            clusters_pos,
        'clusters_neg':            clusters_neg,
        'perm_stats_pos':          perm_stats_pos,
        'perm_stats_neg':          perm_stats_neg,
        'orig_pvals':              orig_pvals,
        'orig_tvals':              orig_tvals,
        'significant_clusters':    significant_clusters
    }
    
    
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    with open(save_fname, 'wb') as fp:
        pickle.dump(out, fp)
    
    print(f"Saved permutation‐cluster results to {save_fname}")
    
# Build the mask from these merged clusters
significant_mask = np.zeros(len(channels), dtype=bool)
for sign, clust_labels, stat, pval in significant_clusters:
    for ch in clust_labels:
        idx = channels.index(ch)
        significant_mask[idx] = True

# Visualize using the original t-values

config.visualize_clusters(
    orig_tvals, channels, significant_mask, info, savepath, vlims
    )

# Optionally, print the significant clusters for inspection.
for s in significant_clusters:
    print(f"Sign: {s[0]}, Channels: {sorted(list(s[1]))}, Cluster Stat: {s[2]:.3f}, p-value: {s[3]:.3f}")
    
    
# %% Corrected Within Group - ME MS Burst

info = epochs.info  # or info from epochs
neighbours = config.prepare_neighbours_from_layout(info, ch_type='eeg')

clus_alpha = 0.05        # uncorrected threshold for candidate electrodes
montecarlo_alpha = 0.05  # threshold for permutation cluster-level test
num_permutations = 100    # adjust as needed
min_cluster_size = 2     # keep clusters with at least 2 channels

vlims = {
    "false_alarms" : (-4, 4),
    "miss" : (-15, 15),
    "rt_go" : (-4.5, 4.5)
    }

subtype = "HS"
behav_interest = "rt_go"
burst_interest = "density"
burst_type = 'Theta'

save_fname = os.path.join(
    figs_path,
    "HI_CTL",
    f"CPerm_{num_permutations}_ME_MS_{burst_type}_{subtype}_{burst_interest}_{behav_interest}.pkl"
    )
savepath = os.path.join(
    figs_path,
    "HI_CTL",
    f"CPerm_{num_permutations}_ME_MS_{burst_type}_{subtype}_{burst_interest}_{behav_interest}.png"
    )

this_df = df[
    ['sub_id', 'subtype', 'channel', burst_interest, behav_interest]].loc[
    (df.burst_type == burst_type)
    & (df.subtype == subtype)
    ]

if os.path.exists(save_fname) :
    big_dic = pd.read_pickle(save_fname)

    orig_tvals           = big_dic['orig_tvals']
    significant_clusters = big_dic['significant_clusters']
    
else : 

    model = f"{behav_interest} ~ {burst_interest}" 
    interest = burst_interest
    to_permute = burst_interest
    
    clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, orig_pvals, orig_tvals = config.permute_and_cluster(
        this_df,
        model, 
        interest,
        to_permute,
        num_permutations,
        neighbours,     
        clus_alpha,
        min_cluster_size,
        channels
        )
    
    # Determine significant clusters based on permutation statistics.
    significant_clusters = config.identify_significant_clusters(
        clusters_pos, 
        clusters_neg, 
        perm_stats_pos, 
        perm_stats_neg, 
        montecarlo_alpha,
        num_permutations
        )
    
    # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # Save everything for later use (before you do the plotting!)
    out = {
        'clusters_pos':            clusters_pos,
        'clusters_neg':            clusters_neg,
        'perm_stats_pos':          perm_stats_pos,
        'perm_stats_neg':          perm_stats_neg,
        'orig_pvals':              orig_pvals,
        'orig_tvals':              orig_tvals,
        'significant_clusters':    significant_clusters
    }
    
    
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    with open(save_fname, 'wb') as fp:
        pickle.dump(out, fp)
    
    print(f"Saved permutation‐cluster results to {save_fname}")
    
# Build the mask from these merged clusters
significant_mask = np.zeros(len(channels), dtype=bool)
for sign, clust_labels, stat, pval in significant_clusters:
    for ch in clust_labels:
        idx = channels.index(ch)
        significant_mask[idx] = True

# Visualize using the original t-values

config.visualize_clusters(
    orig_tvals, channels, significant_mask, info, savepath, vlims[behav_interest]
    )

# Optionally, print the significant clusters for inspection.
for s in significant_clusters:
    print(f"Sign: {s[0]}, Channels: {sorted(list(s[1]))}, Cluster Stat: {s[2]:.3f}, p-value: {s[3]:.3f}")
    
# %% Corrected | ME Behav Burst

info = epochs.info  # or info from epochs
neighbours = config.prepare_neighbours_from_layout(info, ch_type='eeg')

vlims = (-2.5, 2.5)
clus_alpha = 0.05        # uncorrected threshold for candidate electrodes
montecarlo_alpha = 0.05  # threshold for permutation cluster-level test
num_permutations = 100    # adjust as needed
min_cluster_size = 2     # keep clusters with at least 2 channels

burst_type = 'Theta'

feature = "density"

save_fname = os.path.join(
    figs_path,
    "HI_CTL",
    f"CPerm_{num_permutations}_{burst_type}_{feature}_ME_Group.pkl"
    )
savepath = os.path.join(
    figs_path,
    "HI_CTL",
    f"CPerm_{num_permutations}_{burst_type}_{feature}_ME_Group.png"
    )

subdf = df[
    ['sub_id', 'subtype', 'channel', feature]
    ].loc[
    (df.subtype.isin(['HI', 'HS']))
    & (df.burst_type == burst_type)
    ].dropna()

if os.path.exists(save_fname) :
    big_dic = pd.read_pickle(save_fname)

    orig_tvals           = big_dic['orig_tvals']
    significant_clusters = big_dic['significant_clusters']
    
else : 

    model = f"{feature} ~ C(subtype, Treatment('HS'))" 
    interest = "C(subtype, Treatment('HS'))[T.HI]"
    to_permute = "subtype"
    
    clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, orig_pvals, orig_tvals = config.permute_and_cluster(
        subdf,
        model, 
        interest,
        to_permute,
        num_permutations,
        neighbours,     
        clus_alpha,
        min_cluster_size,
        channels
        )
    
    # Determine significant clusters based on permutation statistics.
    significant_clusters = config.identify_significant_clusters(
        clusters_pos, 
        clusters_neg, 
        perm_stats_pos, 
        perm_stats_neg, 
        montecarlo_alpha,
        num_permutations
        )
    
    # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # Save everything for later use (before you do the plotting!)
    out = {
        'clusters_pos':            clusters_pos,
        'clusters_neg':            clusters_neg,
        'perm_stats_pos':          perm_stats_pos,
        'perm_stats_neg':          perm_stats_neg,
        'orig_pvals':              orig_pvals,
        'orig_tvals':              orig_tvals,
        'channels':                channels,
        'significant_clusters':    significant_clusters
    }
    
    
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    with open(save_fname, 'wb') as fp:
        pickle.dump(out, fp)
    
    print(f"Saved permutation‐cluster results to {save_fname}")
    
# Build the mask from these merged clusters
significant_mask = np.zeros(len(channels), dtype=bool)
for sign, clust_labels, stat, pval in significant_clusters:
    for ch in clust_labels:
        idx = channels.index(ch)
        significant_mask[idx] = True

# Visualize using the original t-values

config.visualize_clusters(
    orig_tvals, channels, significant_mask, info, savepath, vlims
    )

# Optionally, print the significant clusters for inspection.
for s in significant_clusters:
    print(f"Sign: {s[0]}, Channels: {sorted(list(s[1]))}, Cluster Stat: {s[2]:.3f}, p-value: {s[3]:.3f}")
