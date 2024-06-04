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

# %% df load

#### DF
df = pd.read_csv(os.path.join(
    wavesPath, "features", "all_SW_features_0.5_7.csv"
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

df = df.loc[df.subtype != 'HI']
df = df.loc[~df.mindstate.isin(['DISTRACTED', 'MISS'])]

mindstates = ['ON', 'MW', 'MB', 'FORGOT', 'HALLU']
subtypes = ['HS', 'N1']
channels = config.eeg_channels
    
# %% Topo | Density | NT1 & HS | Mindstate

feature = 'density'

list_values = []
for i_ms, mindstate in enumerate(mindstates) :
    for subtype in subtypes :   
        for channel in channels :
            list_values.append(df[feature].loc[
                (df["mindstate"] == mindstate)
                & (df["subtype"] == subtype)
                & (df["channel"] == channel)
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
        list_easy.append(df[feature].loc[
            (df["subtype"] == "HS")
            & (df["mindstate"] == mindstate)
            & (df["channel"] == channel)
            ].mean())
        list_hard.append(df[feature].loc[
            (df["subtype"] == "N1")
            & (df["mindstate"] == mindstate)
            & (df["channel"] == channel)
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

    # figsavename = f"{swDataPath}{os.sep}Figs{os.sep}S1_topoplot_density_freq_05_4.png"
    # plt.savefig(figsavename, dpi = 300)

# %% Topo | LME - Subtype effect

model = "density ~ sw_thresh + C(subtype) * C(mindstate)" 

fig, ax = plt.subplots(
    nrows = 1, ncols = len(mindstates), figsize = (18, 5), layout = 'tight')

for i, mindstate in enumerate(mindstates):
    temp_tval = []; temp_pval = []; chan_l = []
    cond_df = df.loc[df.mindstate == mindstate]
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

# %% Topo | LME - MS > ON effect

model = "density ~ sw_thresh + C(mindstate, Treatment('ON'))" 

# fig, ax = plt.subplots(
#     nrows = 1, ncols = len(mindstates), figsize = (18, 8), layout = 'tight')

for i_st, subtype in enumerate(subtypes) :
    fig, ax = plt.subplots(
        nrows = 1, ncols = len(mindstates[1:]), figsize = (18, 8), layout = 'tight')
    for i, mindstate in enumerate(mindstates[1:]):
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = df.loc[df.mindstate.isin(['ON', mindstate])]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'subtype', 'mindstate', 'channel', 'density', 'sw_thresh']
                ].loc[
                (cond_df.channel == chan) &  (cond_df.subtype == subtype)
                ].dropna()
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
            vlim = (-4, 4)
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[i].set_title(f"{subtype} {mindstate}", fontweight = "bold")

    title = f"""
    Topographies of mindstate <Subtype Main Effect> on <Slow Wave Density> according to the <Mindstate> for <{subtype}>
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
          {'font': bold_font},
       ],
       fig=fig
    )


# %% LME | Density ~ Session + Difficulty

indexes = [
        'n_session',
        'C(difficulty)[T.HARD]', 
       'n_session:C(difficulty)[T.HARD]'
       ]

sw_features = ["density", "ptp", "d_slope", "u_slope"]

fig, ax = plt.subplots(
    nrows = 4, ncols = 3, figsize = (12, 8), layout = 'tight')

# Add text to the left of each row
row_labels = ["Density", "PTP", "D_slope", "U_slope"]
for i_sw, sw_feat in enumerate(sw_features):
    ax[i_sw][0].text(-0.5, 0.5, row_labels[i_sw], 
                     verticalalignment='center', 
                     horizontalalignment='center',
                     transform=ax[i_sw][0].transAxes,
                     fontsize='medium', fontweight='bold')
for i_sw, sw_feat in enumerate(sw_features) :
    models = [
        f"{sw_feat} ~ n_session + C(difficulty)",
        f"{sw_feat} ~ n_session + C(difficulty)",
        f"{sw_feat} ~ n_session*C(difficulty)"
        ]
    for i, model in enumerate(models):
        temp_tval = []; temp_pval = []; chan_l = []
        for chan in mean_df.channel.unique():
            subdf = mean_df[
                ['sub_id', 'difficulty', 'n_session', 'channel', 
                 sw_feat]
                ].loc[
                (mean_df.channel == chan)
                ].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'])
            mdf = md.fit()
            temp_tval.append(mdf.tvalues[indexes[i]])
            temp_pval.append(mdf.pvalues[indexes[i]])
            chan_l.append(chan)
             
        _, corrected_pval = fdrcorrection(temp_pval)
        
        divider = make_axes_locatable(ax[i_sw][i])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = temp_tval,
            pos = epochs.info,
            axes = ax[i_sw][i],
            contours = 3,
            mask = np.asarray(corrected_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=4),
            cmap = "viridis",
            # vlim = (-2.5, 2.5)
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    ax[0][0].set_title("Session Effect", fontweight = "bold")
    ax[0][1].set_title("Difficulty Effect", fontweight = "bold")
    ax[0][2].set_title("Interaction Effect", fontweight = "bold")
        
    fig.suptitle("T-values, p-val FDR corrected", 
                 fontsize = "xx-large", 
                 fontweight = "bold")
    plt.savefig(
        os.path.join(
            swDataPath, "Figs", "topo_LME_swfeat.png"
            ),
        dpi = 200
        )

# %% LME | Density ~ Session : Difficulty

sw_features = ["density", "ptp", "d_slope", "u_slope"]

fig, ax = plt.subplots(
    nrows = 2, ncols = 4, figsize = (16, 8), layout = 'tight')
for i_sw, sw_feat in enumerate(sw_features) :
    temp_tval_e = []; temp_pval_e = []; chan_l = []
    temp_tval_h = []; temp_pval_h = []; 
    for chan in mean_df.channel.unique():
        subdf = mean_df[
            ['sub_id', 'difficulty', 'n_session', 'channel', 
             sw_feat]
            ].loc[
            (mean_df.channel == chan)
            ].dropna()
        md = smf.mixedlm(f"{sw_feat} ~ n_session:C(difficulty)", 
                         subdf, groups = subdf['sub_id'])
        mdf = md.fit()
        temp_tval_e.append(mdf.tvalues['n_session:C(difficulty)[EASY]'])
        temp_pval_e.append(mdf.pvalues['n_session:C(difficulty)[EASY]'])
        temp_tval_h.append(mdf.tvalues['n_session:C(difficulty)[HARD]'])
        temp_pval_h.append(mdf.pvalues['n_session:C(difficulty)[HARD]'])
        chan_l.append(chan)
         
    _, corrected_pval_e = fdrcorrection(temp_pval_e)
    _, corrected_pval_h = fdrcorrection(temp_pval_h)
    
    divider = make_axes_locatable(ax[0][i_sw])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval_e,
        pos = epochs.info,
        axes = ax[0][i_sw],
        contours = 3,
        mask = np.asarray(corrected_pval_e) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='w',
                    linewidth=0, markersize=6),
        cmap = "viridis",
        # vlim = (0, 5)
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    ax[0][i_sw].set_title(f"EASY ~ {sw_feat}")
    
    divider = make_axes_locatable(ax[1][i_sw])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval_h,
        pos = epochs.info,
        axes = ax[1][i_sw],
        contours = 3,
        mask = np.asarray(corrected_pval_h) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='w',
                    linewidth=0, markersize=6),
        cmap = "viridis",
        # vlim = (0, 5)
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    ax[1][i_sw].set_title(f"HARD ~ {sw_feat}")
        
fig.suptitle(
    "T-values, p-val FDR corrected\nSW feature ~ n_session:C(difficulty) + (1|sub_id)",
    fontsize = "xx-large",
    fontweight = "bold"
    )
plt.savefig(
    os.path.join(
        swDataPath, "Figs", "topo_correl_LME_swfeat_inter_sess_diff.png"
        ),
    dpi = 200
        )


# %% LME | Density ~ Session x Difficulty

indexes = 'n_session:C(difficulty)[T.HARD]'
       
sw_feature = "density"

fig, ax = plt.subplots(
    nrows = 1, ncols = 1, figsize = (8, 8), layout = 'tight')

temp_tval = []; temp_pval = []; chan_l = []
for chan in mean_df.channel.unique():
    subdf = mean_df[
        ['sub_id', 'difficulty', 'n_session', 'channel', sw_feature]
        ].loc[
        (mean_df.channel == chan)
        ].dropna()
    md = smf.mixedlm(
        "density ~ n_session*C(difficulty)", 
        subdf, 
        groups = subdf['sub_id']
        )
    mdf = md.fit()
    temp_tval.append(mdf.tvalues[indexes])
    temp_pval.append(mdf.pvalues[indexes])
    chan_l.append(chan)
     
_, corrected_pval = fdrcorrection(temp_pval)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "5%", pad=0.05)
im, cm = mne.viz.plot_topomap(
    data = temp_tval,
    pos = epochs.info,
    axes = ax,
    contours = 3,
    mask = np.asarray(corrected_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6),
    cmap = "viridis",
    # vlim = (-2.5, 2.5)
    )
fig.colorbar(im, cax = cax, orientation = 'vertical')
ax.set_title("Interaction Effect", fontweight = "bold")

fig.suptitle("T-values, Uncorrected", 
         fontsize = "xx-large", 
         fontweight = "bold")

savepath = f"{swDataPath}{os.sep}Figs{os.sep}SW_Dens_interaction_session_difficulty_nocorr.png"
plt.savefig(savepath, dpi = 300)

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




