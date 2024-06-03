#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/07/23

@author: arthur.lecoz

06_3_explore_sw.py
"""

# %%% Paths & Packages

import ALC_EEGFATIGUE_config as config
import mne
import scipy
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
import os
from mne import Report

epoching_type = "ntask" # 'ntask' or 'choices'
if epoching_type == "ntask" :
    cleanDataPath = config.cleanDatadir
    swDataPath = config.SWDatadir
elif epoching_type == "choices" :
    cleanDataPath = config.cleanDatadir + "/choices"
    swDataPath = config.SWDatadir + "/choices"

palette_hardeasy = ["#0a9396", "#8e7dbe"]

# %% df load

#### DF
df = pd.read_csv(f"{swDataPath}/df_allsw_exgausscrit_S1_freq_0.5_4.0.csv")
del df['Unnamed: 0']
df['n_session'] = df['n_session'].astype(int)
mean_df = df.groupby(
    by = ["sub_id", "difficulty", "n_session", "channel"], 
    as_index = False).mean()

channel_category = pd.Categorical(
    mean_df['channel'], 
    categories = config.channels, 
    ordered=True
    )

mean_df = mean_df.loc[channel_category.argsort()]
#### DF MEAN
# mean_df = pd.read_csv(f"{swDataPath}/df_meansw_exgausscrit_computedS1_freq_0.5_4.0.csv")
# del mean_df['Unnamed: 0']
# mean_df = mean_df.loc[mean_df['sub_id'] != "2_pf"]
# mean_df = mean_df.loc[mean_df['sub_id'] != "6_yh"]
# mean_df = mean_df.loc[mean_df['sub_id'] != "26_eb"]

epochs = mne.read_epochs(glob(f"{cleanDataPath}/*.fif")[0])

#### DF MEAN normalized (substract sess 1)

thisSavingPath = f"{swDataPath}{os.sep}Figs{os.sep}swdensity_normalized_S1_freq_05_4.csv"

if os.path.exists(thisSavingPath) : 
    density_norm_df = pd.read_csv(thisSavingPath)
    del density_norm_df['Unnamed: 0']
else : 
    normBigDic = {
        "sub_id" : [],
        "channel" : [],
        "n_session" : [],
        "difficulty" : [],
        "normalized_density" : [],
        "normalized_ptp" : [],
        "normalized_dslope" : [],
        }
    
    for i, sub_id in enumerate(mean_df.sub_id.unique()) : 
        if sub_id in ["2_pf"] : 
            continue
        print(f"...Processing {sub_id} : {i+1}/{len(mean_df.sub_id.unique())}")
        for n_ch in mean_df.channel.unique() :
            if not sum(
                    mean_df.loc[mean_df['sub_id'] == sub_id].channel == n_ch):
                continue
            for n_sess in [2, 3, 4, 5, 1] :
                normBigDic['sub_id'].append(sub_id)
                normBigDic['channel'].append(n_ch)
                normBigDic['n_session'].append(n_sess)
                normBigDic['difficulty'].append(
                    mean_df.difficulty.loc[
                        mean_df.sub_id == sub_id
                        ].unique()[0]
                    )
                normBigDic['normalized_density'].append(
                    mean_df.density.loc[
                        (mean_df['sub_id'] == sub_id)
                        & (mean_df['channel'] == n_ch)
                        & (mean_df['n_session'] == n_sess)
                        ].iloc[0] - mean_df.density.loc[
                            (mean_df['sub_id'] == sub_id)
                            & (mean_df['channel'] == n_ch)
                            & (mean_df['n_session'] == 1)
                            ].iloc[0]
                    )
                normBigDic['normalized_ptp'].append(
                    mean_df.ptp.loc[
                        (mean_df['sub_id'] == sub_id)
                        & (mean_df['channel'] == n_ch)
                        & (mean_df['n_session'] == n_sess)
                        ].iloc[0] - mean_df.ptp.loc[
                            (mean_df['sub_id'] == sub_id)
                            & (mean_df['channel'] == n_ch)
                            & (mean_df['n_session'] == 1)
                            ].iloc[0]
                    )
                normBigDic['normalized_dslope'].append(
                    mean_df.d_slope.loc[
                        (mean_df['sub_id'] == sub_id)
                        & (mean_df['channel'] == n_ch)
                        & (mean_df['n_session'] == n_sess)
                        ].iloc[0] - mean_df.d_slope.loc[
                            (mean_df['sub_id'] == sub_id)
                            & (mean_df['channel'] == n_ch)
                            & (mean_df['n_session'] == 1)
                            ].iloc[0]
                    )
                
    
    density_norm_df = pd.DataFrame(
        normBigDic, 
        columns = [
            "sub_id", "channel", "n_session", "difficulty", 
            "normalized_density", "normalized_ptp", "normalized_dslope",
            ]
        )
    density_norm_df.to_csv(thisSavingPath)
    
# %% Topo | LME - Session effect

model = "density ~ n_session"
conditions = ["EASY", "HARD"]

fig, ax = plt.subplots(
    nrows = 2, ncols = 1, figsize = (4, 8), layout = 'tight')

for i, condition in enumerate(conditions):
    temp_tval = []; temp_pval = []; chan_l = []
    cond_df = mean_df.loc[mean_df.difficulty == condition]
    for chan in mean_df.channel.unique():
        subdf = cond_df[
            ['sub_id', 'difficulty', 'n_session', 'channel', 'density']
            ].loc[
            (cond_df.channel == chan)
            ].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'])
        mdf = md.fit()
        temp_tval.append(mdf.tvalues["n_session"])
        temp_pval.append(mdf.pvalues["n_session"])
        chan_l.append(chan)
         
    _, corrected_pval = fdrcorrection(temp_pval)
    
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval,
        pos = epochs.info,
        axes = ax[i],
        contours = 3,
        mask = np.asarray(corrected_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=6),
        cmap = "viridis",
        vlim = (-4, 4)
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')

ax[0].set_title("EASY", fontweight = "bold")
ax[1].set_title("HARD", fontweight = "bold")
    
fig.suptitle("T-values, p-val FDR corrected", 
             fontsize = "xx-large", 
             fontweight = "bold")
plt.savefig(
    os.path.join(
        swDataPath, "Figs", "topo_LME_sessioneffect.png"
        ),
    dpi = 200
    )

# %% TOPO LME - Normed Data Difficulty 

this_df = mean_df.loc[
    mean_df.n_session != 1
    ]

model = "density ~ C(difficulty)"

fig, ax = plt.subplots(
    nrows = 1, ncols = 1, figsize = (5, 5), layout = 'tight')

temp_tval = []; temp_pval = []; chan_l = []
for chan in mean_df.channel.unique():
    subdf = this_df[
        ['sub_id', 'difficulty', 'n_session', 'channel', 'density']
        ].loc[
        (this_df.channel == chan)
        ].dropna()
    md = smf.mixedlm(model, subdf, groups = subdf['sub_id'])
    mdf = md.fit()
    temp_tval.append(mdf.tvalues["C(difficulty)[T.HARD]"])
    temp_pval.append(mdf.pvalues["C(difficulty)[T.HARD]"])
    chan_l.append(chan)
     
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
    # vlim = (-4, 4)
    )
fig.colorbar(im, cax = cax, orientation = 'vertical')

ax[0].set_title("Difficulty Effect", fontweight = "bold")
    
fig.suptitle("T-values, p-val FDR corrected", 
             fontsize = "xx-large", 
             fontweight = "bold")
plt.savefig(
    os.path.join(
        swDataPath, "Figs", "topo_LME_difficultyeffect.png"
        ),
    dpi = 200
    )

# %% Topo | FEATURE | T-values | Difficulty | S to S1 

# list_values = []
# for i_sess, n_sess in enumerate([1, 2, 3, 4, 5]) :
#     for difficulty in ["EASY", "HARD"] :   
#         for channel in mean_df.channel.unique() :
#             list_values.append(mean_df.density.loc[
#                 (mean_df["n_session"] == n_sess)
#                 & (mean_df["difficulty"] == difficulty)
#                 & (mean_df["channel"] == channel)
#                 ].mean())
# densvmin = min(list_values)
# densvmax = max(list_values)

feature = "density"

fig, ax = plt.subplots(
    nrows = 2, 
    ncols = 4,
    figsize = (18, 8),
    layout = 'tight'
    )
for i_diff, difficulty in enumerate(["EASY", "HARD"]) :
    for i_sess, n_sess in enumerate([2, 3, 4, 5]):
        temp_tval = []; temp_pval = []; chan_l = []
        for chan in mean_df.channel.unique():
            t_val, p_val = ttest_ind(
                mean_df[feature].loc[
                    (mean_df['channel'] == chan)
                    & (mean_df['difficulty'] == difficulty)
                    & (mean_df['n_session'] == n_sess)
                    ],
                mean_df[feature].loc[
                    (mean_df['channel'] == chan)
                    & (mean_df['difficulty'] == difficulty)
                    & (mean_df['n_session'] == 1)
                    ],
                nan_policy = "omit"
                )
            temp_pval.append(p_val)
            temp_tval.append(t_val)
            chan_l.append(chan)
            
        if n_sess == 5 :
            divider = make_axes_locatable(ax[i_diff][i_sess])
            cax = divider.append_axes("right", size = "5%", pad=0.05)
        mne.viz.plot_topomap(
            data = temp_tval,
            pos = epochs.info,
            # names = chan_l,
            axes = ax[i_diff][i_sess],
            mask = np.asarray(temp_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=6),
            # vlim = (densvmin, densvmax),
            size = 8,
            cmap = "viridis"
            )
        if  n_sess == 5:
            fig.colorbar(im, cax = cax, orientation = 'vertical')
        # ax[i_diff][i_sess].set_title(f"{difficulty} - S{n_sess} v S1")
    # fig.suptitle("T-values S2-5 to S1 according to the conditions")
    
plt.savefig(f"{swDataPath}/Figs/S1_{feature}_topomap_sessions_to_s1_f054.png", dpi = 300)

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




