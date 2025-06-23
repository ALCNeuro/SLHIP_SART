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
from scipy.stats import ttest_ind
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

# %% Function Cluster Perm

from scipy.spatial import Delaunay

def prepare_neighbours_from_layout(info, ch_type='eeg'):
    """
    Create a neighbours structure similar to FieldTrip's ft_prepare_neighbours
    using Delaunay triangulation.

    Parameters
    ----------
    info : MNE Info object
        contains sensor locations..
    ch_type : str, optional
        Type of channels to consider (e.g., 'eeg'). The default is 'eeg'.

    Returns
    -------
    neighbours_list : A list of dicts.
        With each channel's label and its neighbours.

    """
    # Get the 2D positions of the channels from info
    pos = []
    labels = []
    for ch in info['chs']:
        # Filter by channel type if needed, e.g., check ch['kind'] or use mne.pick_types.
        # Here we simply assume that the info is for the ch_type of interest.
        if 'loc' in ch:
            # Use the first two coordinates from the sensor location as 2D projection
            pos.append(ch['loc'][:2])
            labels.append(ch['ch_name'])
    
    pos = np.array(pos)
    
    # Perform Delaunay triangulation
    tri = Delaunay(pos)
    
    # Create a dictionary where each channel has a set of neighbours
    neighbours = {label: set() for label in labels}
    
    # For each simplex (triangle) in the triangulation, add edges between sensors
    for simplex in tri.simplices:
        for i in range(3):
            ch_i = labels[simplex[i]]
            for j in range(i + 1, 3):
                ch_j = labels[simplex[j]]
                neighbours[ch_i].add(ch_j)
                neighbours[ch_j].add(ch_i)
    
    # Format the neighbours info as a list of dictionaries (similar to FieldTrip's structure)
    neighbours_list = []
    for label in labels:
        neighbours_list.append({
            'label': label,
            'neighblabel': list(neighbours[label])
        })
    
    return neighbours_list

# ============================================================================
# 1. Run mixed-effects model for a given channel (unchanged)
def run_mixedlm(data, channel, interest, model):
    """
    Run a mixed linear model for a given channel

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing the values to run LME.
    channel : List
        List of channels in the correct order for later plotting.
    interest : str
        Name of effect of interest to collect p and t values.
    model : str
        Models to run LME.

    Returns
    -------
    p_values_oi : list
        p values of interest.
    t_values_oi : list
        t values of interest.
    """
    subdf = data.loc[data.channel == channel].dropna()
    md = smf.mixedlm(
        model, subdf, groups=subdf['sub_id']
        )
    try:
        mdf = md.fit(method='lbfgs', reml=False)
        t_values_oi = mdf.tvalues[interest]
        p_values_oi = mdf.pvalues[interest]
    except Exception as e:
        print(f"Model failed for channel {channel}: {e}")
        t_values_oi = np.nan
        p_values_oi = 1
    
    return (p_values_oi, t_values_oi)

# ============================================================================
# 2. Updated helper function: Cluster significant channels using spatial neighbours,
#    ensuring only candidate channels (uncorrected p < clus_alpha) are included.
def cluster_significant_channels(
        channels, 
        pvals, 
        tvals, 
        neighbours, 
        clus_alpha, 
        min_cluster_size, 
        sign='pos'
        ):
    """
    Create clusters of significant channels based solely on candidate channels.
    1) Select candidate channels (p < clus_alpha, correct sign).
    2) Seed one‐channel clusters for each candidate.
    3) Iteratively merge any two clusters if any channel in A is a neighbour
       of any channel in B (using the supplied neighbours map).
    4) Discard clusters smaller than min_cluster_size.

    Parameters
    ----------
    channels : List
        List of channel labels.
    pvals : List
        List of p-values for each channel.
    tvals : List
        List of t-values for each channel.
    neighbours : List of dicts
        List of dicts. Each dict has keys 'label' (channel label)
        and 'neighblabel' (list of neighbouring channel labels).
    clus_alpha : float
        The uncorrected p-value threshold.
    min_cluster_size : float
        Minimum number of channels required for a valid cluster.
    sign : str, optional
        'pos' for positive effects, 'neg' for negative. The default is 'pos'.

    Returns
    -------
    clusters : List
        A list of clusters. Each cluster is a dict with keys:
        'labels'  : a set of channel labels that are candidates and belong to the cluster,
        'tstats'  : a list of t-values for those channels,
        'neighbs' : the union of candidate neighbour labels for all channels in the cluster.
    """
    # 1) Build candidate set
    candidate_set = {
        channels[i]
        for i in range(len(channels))
        if (pvals[i] < clus_alpha)
           and ((sign=='pos' and tvals[i]>0) or (sign=='neg' and tvals[i]<0))
    }
    # Quick neighbour lookup: channel -> set of its neighbours (intersected with candidates)
    neigh_map = {
        n['label']: set(n['neighblabel']).intersection(candidate_set)
        for n in neighbours
        if n['label'] in candidate_set
    }
    # 2) Seed initial one‐channel clusters
    clusters = []
    for ch in candidate_set:
        clusters.append({
            'labels': {ch},
            'tstats': [tvals[channels.index(ch)]]
        })
    
    # 3) Iteratively merge any two clusters that touch
    merged = True
    while merged:
        merged = False
        new_clusters = []
        used = [False]*len(clusters)
        for i, ci in enumerate(clusters):
            if used[i]:
                continue
            # try to absorb any later cluster that’s adjacent
            for j in range(i+1, len(clusters)):
                if used[j]:
                    continue
                cj = clusters[j]
                # check adjacency: any channel in ci is neighbour of any in cj?
                if any(
                    (label in neigh_map and neigh_map[label] & cj['labels'])
                    for label in ci['labels']
                ) or any(
                    (label in neigh_map and neigh_map[label] & ci['labels'])
                    for label in cj['labels']
                ):
                    # fuse j into i
                    ci['labels'] |= cj['labels']
                    ci['tstats']  += cj['tstats']
                    used[j] = True
                    merged = True
            new_clusters.append(ci)
        clusters = new_clusters
    
    # 4) Filter by minimum cluster size
    clusters = [c for c in clusters if len(c['labels']) >= min_cluster_size]
    return clusters

# ============================================================================
# 3. Permutation procedure with clustering using neighbour information
def permute_and_cluster(
        data, 
        model,
        interest, 
        to_permute,
        num_permutations, 
        neighbours, 
        clus_alpha, 
        min_cluster_size
        ):
    """
    Compute original channel p-values and t-values, build clusters based on 
    neighbours, and then generate a null distribution via permutation clustering.

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing the data.
    model : str
        The model to run linear mixed models.
    interest : str
        The effect of interest (e.g., 'n_session:C(difficulty)[T.HARD]').
    num_permutations : int
        Number of permutations.
    neighbours : List of Dict
        Neighbours structure (list of dicts as produced by e.g., prepare_neighbours_from_layout).
    clus_alpha : float
        Uncorrected p-value threshold (e.g., 0.05).
    min_cluster_size : int
        Minimum number of channels per cluster..

    Returns
    -------
    clusters_pos : List (to complete)
        Clusters from the real (non-permuted) data (for positive effects)..
    clusters_neg : List (to complete)
        Clusters from the real (non-permuted) data (for negative effects)..
    perm_stats_pos : List
        Lists of maximum cluster stats from each permutation.
    perm_stats_neg : List
        Lists of maximum cluster stats from each permutation.
    original_pvals : List
        Original channel-level statistics.
    original_tvals : List
        Original channel-level statistics.
    channels : List
        Original channel-level statistics.

    """
    channels = list(data.channel.unique())
    original_pvals = []
    original_tvals = []
    for chan in channels:
        p, t = run_mixedlm(data, chan, interest, model)
        original_pvals.append(p)
        original_tvals.append(t)
    
    if np.any(np.isnan(original_tvals)) :
        for pos in np.where(np.isnan(original_tvals))[0] :
            original_tvals[pos] = np.nanmean(original_tvals)
    
    # Form clusters separately for positive and negative effects.
    clusters_pos = cluster_significant_channels(
        channels, 
        original_pvals, 
        original_tvals,
        neighbours, 
        clus_alpha, 
        min_cluster_size, 
        sign='pos'
        )
    clusters_neg = cluster_significant_channels(
        channels, 
        original_pvals, 
        original_tvals,
        neighbours, 
        clus_alpha, 
        min_cluster_size, 
        sign='neg'
        )
    
    perm_stats_pos = []  # one value per permutation: maximum cluster t-sum (for positive clusters)
    perm_stats_neg = []  # one value per permutation: minimum (most negative) cluster t-sum (for negative clusters)
    
    for _ in range(num_permutations):
        shuffled_data = data.copy()
        shuffled_data[to_permute] = np.random.permutation(shuffled_data[to_permute].values)
        perm_pvals = []
        perm_tvals = []
        for chan in channels:
            p, t = run_mixedlm(shuffled_data, chan, interest, model)
            perm_pvals.append(p)
            perm_tvals.append(t)
        
        perm_clusters_pos = cluster_significant_channels(
            channels, 
            perm_pvals, 
            perm_tvals,
            neighbours, 
            clus_alpha, 
            min_cluster_size, 
            sign='pos')
        perm_clusters_neg = cluster_significant_channels(
            channels,
            perm_pvals, 
            perm_tvals,
            neighbours, 
            clus_alpha, 
            min_cluster_size, 
            sign='neg'
            )
        
        if perm_clusters_pos:
            perm_stat_pos = max(sum(clust['tstats']) for clust in perm_clusters_pos)
        else:
            perm_stat_pos = 0
        perm_stats_pos.append(perm_stat_pos)
        
        if perm_clusters_neg:
            perm_stat_neg = min(sum(clust['tstats']) for clust in perm_clusters_neg)
        else:
            perm_stat_neg = 0
        perm_stats_neg.append(perm_stat_neg)
    
    return clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, original_pvals, original_tvals, channels

# ============================================================================
# 4. Determine which real clusters are significant via permutation comparison
def identify_significant_clusters(
        clusters_pos,
        clusters_neg, 
        perm_stats_pos, 
        perm_stats_neg, 
        montecarlo_alpha, 
        num_permutations
        ):
    """
    Compare each original cluster statistic against its permutation distribution and return
    those clusters that are significant.

    Parameters
    ----------
    clusters_pos : List (to complete)
        Clusters from the real (non-permuted) data (for positive effects)..
    clusters_neg : List (to complete)
        Clusters from the real (non-permuted) data (for negative effects)..
    perm_stats_pos : List
        Lists of maximum cluster stats from each permutation.
    perm_stats_neg : List
        Lists of maximum cluster stats from each permutation.
    montecarlo_alpha : TYPE
        DESCRIPTION.
    num_permutations : TYPE
        DESCRIPTION.

    Returns
    -------
    significant_clusters : A list of tuple
        (sign, cluster_labels, cluster_stat, p_value)
        where sign is 'pos' or 'neg'.

    """
    
    significant_clusters = []
    for clust in clusters_pos:
        stat = sum(clust['tstats'])
        p_value = (np.sum(np.array(perm_stats_pos) >= stat) + 1) / (num_permutations + 1)
        if p_value < montecarlo_alpha:
            significant_clusters.append(('pos', clust['labels'], stat, p_value))
    
    for clust in clusters_neg:
        stat = sum(clust['tstats'])
        p_value = (np.sum(np.array(perm_stats_neg) <= stat) + 1) / (num_permutations + 1)
        if p_value < montecarlo_alpha:
            significant_clusters.append(('neg', clust['labels'], stat, p_value))
    
    return significant_clusters


# ============================================================================
# 5. Visualization function
def visualize_clusters(tvals, channels, significant_mask, info, vlims, savepath):
    """
    Visualize significant clusters using topomap.

    Parameters
    ----------
    tvals : TYPE
        DESCRIPTION.
    channels : TYPE
        DESCRIPTION.
    significant_mask : TYPE
        DESCRIPTION.
    info : TYPE
        DESCRIPTION.
    savepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im, cm = mne.viz.plot_topomap(
        data=np.array(tvals),
        pos=info,  # expects sensor positions from the info object
        mask=significant_mask,
        axes=ax,
        show=False,
        contours=2,
        mask_params=dict(
            marker='o',
            markerfacecolor='w',
            markeredgecolor='k',
            linewidth=0,
            markersize=8
        ),
        cmap="coolwarm",
        vlim = vlims
        )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cax.set_title("t-values", fontsize=10)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=10)
    fig.colorbar(im, cax=cax)
    fig.tight_layout()
    # ax.set_title("Interaction Effect", fontweight="bold")
    # fig.suptitle("T-values, Cluster Permutation Corrected", fontsize="xx-large", fontweight="bold")
    plt.savefig(savepath, dpi = 300)
    plt.show()

# %% Remove IH from df

epochs = mne.read_epochs(glob(f"{cleanDataPath}/epochs_probes/*.fif")[0])
epochs.drop_channels(['TP9', 'TP10', 'VEOG', 'HEOG', 'ECG', 'RESP'])
neighbours = prepare_neighbours_from_layout(epochs.info, ch_type='eeg')

df = df.loc[
    (~df.mindstate.isin(['DISTRACTED', 'MISS']))
    & (df.subtype != 'HI')
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
            (mean_df["subtype"] == "N1")
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
    ax_nt[i_bt].set_title(f"NT1 - {burst_type}", font = bold_font, fontsize = 12)
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    plt.show(block = False)
    fig.tight_layout()
    
    figsavename = os.path.join(
        figs_path, 'NT1_CTL', f'topo_{feature}_subtypes.png'
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
            & (df.subtype.isin(['N1', 'HS']))
            & (df.burst_type == burst_type)
            ].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'drop')
        mdf = md.fit()
        temp_tval.append(mdf.tvalues["C(subtype, Treatment('HS'))[T.N1]"])
        temp_pval.append(mdf.pvalues["C(subtype, Treatment('HS'))[T.N1]"])
        chan_l.append(chan)
        
    if np.any(np.isnan(temp_tval)) :
        temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
         
    # _, corrected_pval = fdrcorrection(temp_pval)
    corrected_pval = false_discovery_control(temp_pval, method ='by')
    
    if i_bt == len(burst_types) - 1 :
        divider = make_axes_locatable(ax[i_bt])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval,
        pos = epochs.info,
        axes = ax[i_bt],
        contours = 3,
        mask = np.asarray(corrected_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=8),
        cmap = "coolwarm",
        vlim = (-2.5, 2.5),
        size = 2.5
        )
    if i_bt == len(burst_types) - 1 :
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
    ax[i_bt].set_title(f"{burst_type} N1 > HS", fontweight = "bold", fontsize = 12)
    
    fig.tight_layout()
        
# plt.savefig(os.path.join(
#     figs_path, 'NT1_CTL', f'LME_topo_{feature}_ME_group.png'
#     ), dpi = 300)

# %% Diff MS within GROUP

vlims = {
    "Alpha" : {
        "HS" : (-3, 3),
        "N1" : (-7.5, 7.5)
        },
    "Theta" : {
        "HS" : (-3, 3),
        "N1" : (-4, 4)
        }
    }

kindaburst = "Theta"

this_df = df.loc[df.burst_type==kindaburst]

interest = 'density'
contrasts = [("ON", "MW"), ("ON", "MB"), ("ON", "HALLU"), ("ON", "FORGOT")]

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
            
            # if f"C(mindstate, Treatment('{contrast[0]}'))[T.{contrast[1]}]" not in mdf.tvalues.index :
            #     temp_tval.append(np.nan)
            #     temp_pval.append(1)
            #     chan_l.append(chan)
            # else : 
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
        this_ax[i_c].set_title(f"{contrast[1]} > {contrast[0]}", fontweight = "bold")

    # fig.suptitle(f"{interest}", font = bold_font, fontsize = 24)
    fig.tight_layout()
    figsavename = os.path.join(
        figs_path, 'NT1_CTL', f'LME_topo_{kindaburst}_{feature}_ME_MS_{subtype}.png'
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

kindaburst = "Theta"
behav_interest = "miss"
burst_interest = "density"
group_oi = "N1"

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
    mdf = md.fit()
    temp_tval.append(mdf.tvalues[burst_interest])
    temp_pval.append(mdf.pvalues[burst_interest])
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
    contours = 3,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=8),
    cmap = "viridis",
    # vlim = (-4, 4)
    )
fig.colorbar(im, cax = cax, orientation = 'vertical')

ax.set_title(f"{behav_interest} ~ {burst_interest}", fontweight = "bold", fontsize = 12)
fig.tight_layout()
plt.savefig(os.path.join(
    figs_path, 'NT1_CTL', f'LME_{behav_interest}_burst_{kindaburst}_{burst_interest}_in_{group_oi}.png'
    ), dpi = 300)   

# %% Corrected | ME Group Burst

info = epochs.info  # or info from epochs
neighbours = prepare_neighbours_from_layout(info, ch_type='eeg')

clus_alpha = 0.05        # uncorrected threshold for candidate electrodes
montecarlo_alpha = 0.05  # threshold for permutation cluster-level test
num_permutations = 200    # adjust as needed
min_cluster_size = 2     # keep clusters with at least 2 channels

save_fname = os.path.join(
    figs_path,
    "NT1_CTL",
    f"CPerm_{num_permutations}_ME_Group.pkl"
    )
savepath = os.path.join(
    figs_path,
    f"CPerm_{num_permutations}_ME_Group.png"
    )

burst_type = 'Theta'

feature = "density"

subdf = df[
    ['sub_id', 'subtype', 'channel', feature]
    ].loc[
    (df.subtype.isin(['N1', 'HS']))
    & (df.burst_type == burst_type)
    ].dropna()

if os.path.exists(save_fname) :
    big_dic = pd.read_pickle(save_fname)

    orig_tvals           = big_dic['orig_tvals']
    channels             = big_dic['channels']
    significant_clusters = big_dic['significant_clusters']
    
else : 

    model = f"{feature} ~ C(subtype, Treatment('HS'))" 
    interest = "C(subtype, Treatment('HS'))[T.N1]"
    to_permute = "subtype"
    
    clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, orig_pvals, orig_tvals, channels = permute_and_cluster(
        subdf,
        model, 
        interest,
        to_permute,
        num_permutations,
        neighbours,     
        clus_alpha,
        min_cluster_size
        )
    
    # Determine significant clusters based on permutation statistics.
    significant_clusters = identify_significant_clusters(
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

visualize_clusters(
    orig_tvals, channels, significant_mask, info, savepath
    )

# Optionally, print the significant clusters for inspection.
for s in significant_clusters:
    print(f"Sign: {s[0]}, Channels: {sorted(list(s[1]))}, Cluster Stat: {s[2]:.3f}, p-value: {s[3]:.3f}")
    
    
# %% Corrected Within Group - ME MS Burst

info = epochs.info  # or info from epochs

clus_alpha = 0.05        # uncorrected threshold for candidate electrodes
montecarlo_alpha = 0.05  # threshold for permutation cluster-level test
num_permutations = 200    # adjust as needed
min_cluster_size = 2     # keep clusters with at least 2 channels

vlims = (-3.5, 3.5)
subtype = "HS"
contrast = ["FORGOT", "ON"]
burst_type = 'Theta'
feature = "density"

save_fname = os.path.join(
    figs_path,
    "NT1_CTL",
    f"CPerm_{num_permutations}_ME_MS_{burst_type}_{subtype}_{feature}_{contrast[1]}_vs_{contrast[0]}.pkl"
    )
savepath = os.path.join(
    figs_path,
    "NT1_CTL",
    f"CPerm_{num_permutations}_ME_MS_{burst_type}_{subtype}_{feature}_{contrast[1]}_vs_{contrast[0]}.png"
    )

this_df = df[
    ['sub_id', 'subtype', 'mindstate', 'channel', f'{feature}']].loc[
    (df.burst_type == burst_type)
    & (df.subtype == subtype)
    & (df.mindstate.isin(contrast))
    ]

if os.path.exists(save_fname) :
    big_dic = pd.read_pickle(save_fname)

    orig_tvals           = big_dic['orig_tvals']
    channels             = big_dic['channels']
    significant_clusters = big_dic['significant_clusters']
    
else : 

    model = f"{feature} ~ C(mindstate, Treatment('{contrast[0]}'))" 
    interest = f"C(mindstate, Treatment('{contrast[0]}'))[T.{contrast[1]}]"
    to_permute = "mindstate"
    
    clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, orig_pvals, orig_tvals, channels = permute_and_cluster(
        this_df,
        model, 
        interest,
        to_permute,
        num_permutations,
        neighbours,     
        clus_alpha,
        min_cluster_size
        )
    
    # Determine significant clusters based on permutation statistics.
    significant_clusters = identify_significant_clusters(
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

visualize_clusters(
    orig_tvals, channels, significant_mask, info, vlims, savepath
    )

# Optionally, print the significant clusters for inspection.
for s in significant_clusters:
    print(f"Sign: {s[0]}, Channels: {sorted(list(s[1]))}, Cluster Stat: {s[2]:.3f}, p-value: {s[3]:.3f}")
    
    
