#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:13:31 2025

@author: arthurlecoz

03_08_explore_band_periodic.py
"""
# %% Paths & Packages
import pickle
import mne
import os

import pandas as pd
import numpy as np
import SLHIP_config_ALC as config
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.font_manager import FontProperties
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

cleanDataPath = config.cleanDataPath
powerPath = config.powerPath

bandpowerPath = os.path.join(powerPath, 'bandperiodic')
reports_path = os.path.join(bandpowerPath, "reports")

channels = np.array(config.eeg_channels)

subtypes = ["C1", "HI", "N1"]
psd_palette = ["#8d99ae", "#d00000", "#ffb703"]

freqs = np.linspace(0.5, 40, 159)

freq_bands = {
     'delta': (.5, 4),
     'theta': (4, 8),
     'alpha': (8, 12),
     'beta': (12, 30),
     'gamma': (30, 40)
     }

cols_power = [
    'abs_delta','abs_theta','abs_alpha','abs_beta','abs_gamma',
    'rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma'
    ]

files = glob(os.path.join(cleanDataPath, "epochs_probes", "*.fif"))
epochs = mne.read_epochs(files[0], preload = True)
epochs.pick('eeg')
epochs.drop_channels(['TP9', 'TP10'])

info = epochs.info
channel_order = np.array(epochs.ch_names)

df = pd.read_csv(os.path.join(bandpowerPath, "bandpower.csv"))
del df['Unnamed: 0']

subtypes = ["HS", "N1", "HI"]
power_types = ["abs", "rel"]
bands = ["delta", "theta", "alpha", "beta", "gamma"]

clus_alpha = 0.05        
montecarlo_alpha = 0.05  
min_cluster_size = 2

# %% DataFrame Manipulation

neighbours = config.prepare_neighbours_from_layout(epochs.info)

mean_df = df.copy().drop(columns="daytime").groupby(
    ['sub_id', 'subtype', 'channel', 'mindstate'], 
    as_index=False
    ).mean()

this_df = df.loc[df.subtype!="N1"]

# %% Distrib Values 

for power_type in power_types :
    
    fig, axs = plt.subplots(
        nrows = len(subtypes), 
        ncols = len(bands), 
        figsize = (16, 6), 
        layout = 'tight'
        )
    
    for i_b, band in enumerate(bands) :
        
        feature = f"{power_type}_{band}"
        
        columns = [feature, "channel"]
        
        vmin = df[columns].groupby(
            'channel', as_index = False).mean().min()[feature]
        vmax = df[columns].groupby(
            'channel', as_index = False).mean().max()[feature]
        
        for i_st, subtype in enumerate(subtypes) :
            ax = axs[i_st]
            
            temp_values = df[columns].loc[
                (df.subtype == subtype)
                ]
            
            temp_values = temp_values.groupby('channel', as_index = False).mean()
            temp_values['channel'] = pd.Categorical(
                temp_values['channel'], 
                categories=channel_order, 
                ordered=True
                )
            df_sorted = temp_values.sort_values('channel')
    
            values = df_sorted[feature].to_numpy()
        
            divider = make_axes_locatable(ax[i_b],)
            cax = divider.append_axes("right", size = "5%", pad=0.05)
            im, cm = mne.viz.plot_topomap(
                data = values,
                pos = info,
                axes = ax[i_b],
                contours = 2,
                cmap = "Purples",
                vlim = (vmin, vmax)
                )
            fig.colorbar(im, cax = cax, orientation = 'vertical')
            ax[i_b].set_title(
                f"{subtype} | {power_type}_{band}", 
                fontweight = "bold", 
                fontsize = 10
                ) 
        
        plt.show()  
    fig.tight_layout()
    plt.savefig(os.path.join(
        bandpowerPath, "figs", f"{power_type}_across_subtype.png"
        ), dpi = 300)

# %% ME - Subtype LME

vlims = {
    "HI" : (-2, 2),
    "N1" : (-2.5, 2.5)
    }

fdr_corrected = 1
    
for i_subtype, subtype in enumerate(["N1", "HI"]):
    
    fig, axs = plt.subplots(
        nrows = len(power_types), 
        ncols = len(bands), 
        figsize = (12, 5)
        )
    
    for i_pt, power_type in enumerate(power_types) :
        ax = axs[i_pt]
        for i_b, band in enumerate(bands) :
            
            interest = f"{power_type}_{band}"
            model = f"{interest} ~ C(subtype, Treatment('HS'))" 
    
            temp_tval = []; temp_pval = []; chan_l = []
            cond_df = df.loc[df.subtype.isin(['HS', subtype])]
            for chan in channels :
                subdf = cond_df[
                    ['sub_id', 'subtype', 'channel', 'sleepiness', f'{interest}']
                    ].loc[(cond_df.channel == chan)].dropna()
                md = smf.mixedlm(
                    model, subdf, groups = subdf['sub_id'], missing = 'omit'
                    )
                mdf = md.fit()
                temp_tval.append(
                    mdf.tvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"]
                    )
                temp_pval.append(
                    mdf.pvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"]
                    )
                chan_l.append(chan)
                
            if np.any(np.isnan(temp_tval)) :
                temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(
                    temp_tval
                    )
                 
            if fdr_corrected :
                _, corrected_pval = fdrcorrection(temp_pval)
                display_pval = corrected_pval
            else : 
                display_pval = temp_pval
            
            if i_b == len(bands) - 1 :
                divider = make_axes_locatable(ax[i_b])
                cax = divider.append_axes("right", size = "5%", pad=0.05)
            im, cm = mne.viz.plot_topomap(
                data = temp_tval,
                pos = epochs.info,
                axes = ax[i_b],
                contours = 3,
                mask = np.asarray(display_pval) <= 0.05,
                mask_params = dict(
                    marker='o', 
                    markerfacecolor='w', 
                    markeredgecolor='k',
                    linewidth=0, 
                    markersize=6
                    ),
                cmap = "coolwarm",
                vlim = (vlims[subtype])
                )
            if i_b == len(bands) - 1 :
                fig.colorbar(im, cax = cax, orientation = 'vertical')
        
            ax[i_b].set_title(f"{interest}", font = bold_font, fontsize=12)
        fig.suptitle(f"{subtype} > HS", font = bold_font, fontsize=16)
        fig.tight_layout()
        
    plt.savefig(os.path.join(
        bandpowerPath, "figs", f"{subtype}_vs_HS.png"
        ), dpi=300)

# %% NT1 vs HI

fig, axs = plt.subplots(
    nrows = len(power_types), 
    ncols = len(bands), 
    figsize = (12, 5)
    )

for i_pt, power_type in enumerate(power_types) :
    ax = axs[i_pt]
    for i_b, band in enumerate(bands) :
        
        interest = f"{power_type}_{band}"
        model = f"{interest} ~ C(subtype, Treatment('HI'))" 

        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = df.loc[df.subtype.isin(['HI', 'N1'])]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'subtype', 'channel', 'sleepiness', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(
                model, subdf, groups = subdf['sub_id'], missing = 'omit'
                )
            mdf = md.fit()
            temp_tval.append(
                mdf.tvalues["C(subtype, Treatment('HI'))[T.N1]"]
                )
            temp_pval.append(
                mdf.pvalues["C(subtype, Treatment('HI'))[T.N1]"]
                )
            chan_l.append(chan)
            
        if np.any(np.isnan(temp_tval)) :
            temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(
                temp_tval
                )
             
        if fdr_corrected :
            _, corrected_pval = fdrcorrection(temp_pval)
            display_pval = corrected_pval
        else : 
            display_pval = temp_pval
        
        if i_b == len(bands) - 1 :
            divider = make_axes_locatable(ax[i_b])
            cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = temp_tval,
            pos = epochs.info,
            axes = ax[i_b],
            contours = 3,
            mask = np.asarray(display_pval) <= 0.05,
            mask_params = dict(
                marker='o', 
                markerfacecolor='w', 
                markeredgecolor='k',
                linewidth=0, 
                markersize=6
                ),
            cmap = "coolwarm",
            vlim = (-3.5, 3.5)
            )
        if i_b == len(bands) - 1 :
            fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[i_b].set_title(f"{interest}", font = bold_font, fontsize=12)
    fig.suptitle("NT1 > IH", font = bold_font, fontsize=16)
    fig.tight_layout()
    
plt.savefig(os.path.join(
    bandpowerPath, "figs", "N1_vs_HI.png"
    ), dpi=300)

# %% ME - MS LME

fdr_corrected = 0

for interest in cols_power :
    model = f"{interest} ~ C(mindstate, Treatment('HALLU'))" 
    
    fig, ax = plt.subplots(
        nrows = 1, ncols = 4, figsize = (10, 3))
    for i, ms in enumerate(["ON", "MW", "MB", "FORGOT"]):
        model = f"{interest} ~ C(mindstate, Treatment('{ms}'))" 
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = this_df.loc[this_df.mindstate.isin(['HALLU', ms])]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'mindstate', 'channel', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
            mdf = md.fit()
            temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('{ms}'))[T.HALLU]"])
            temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('{ms}'))[T.HALLU]"])
            chan_l.append(chan)
            
        if np.any(np.isnan(temp_tval)) :
            temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
             
        if fdr_corrected :
            _, corrected_pval = fdrcorrection(temp_pval)
            display_pval = corrected_pval
        else : 
            display_pval = temp_pval
        
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = temp_tval,
            pos = epochs.info,
            axes = ax[i],
            contours = 3,
            mask = np.asarray(display_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=6),
            cmap = "viridis",
            vlim = (np.percentile(temp_tval, 5), np.percentile(temp_tval, 95))
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[i].set_title(f"HALLU > {ms}", font = bold_font, fontsize=12)
    fig.suptitle(f"{interest}", font = bold_font, fontsize=16)
    fig.tight_layout(pad = 2)
    plt.savefig(os.path.join(
        bandpowerPath, "figs", f"{interest}_ME_mentalstate_compared_to_hallu.png"
        ), dpi=300)
    
# %% MS LME SUBTYPE

investigate_st = "HI"
which_ms = "ON"
ms_list = ["MW", "MB", "HALLU", "FORGOT"]

n1_df = df.loc[df.subtype==investigate_st]

fdr_corrected = 0

for interest in cols_power :
    model = f"{interest} ~ sleepiness + C(mindstate, Treatment('{which_ms}'))" 
    fig, ax = plt.subplots(
        nrows = 1, ncols = len(ms_list), figsize = (10, 3))
    
    for i, mindstate in enumerate(ms_list):
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = n1_df.loc[n1_df.mindstate.isin([which_ms, mindstate])]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'mindstate', 'channel', 'sleepiness', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
            mdf = md.fit()
            temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('{which_ms}'))[T.{mindstate}]"])
            temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('{which_ms}'))[T.{mindstate}]"])
            chan_l.append(chan)
            
        if np.any(np.isnan(temp_tval)) :
            temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
             
        if fdr_corrected :
            _, corrected_pval = fdrcorrection(temp_pval)
            display_pval = corrected_pval
        else : 
            display_pval = temp_pval
        
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = np.asarray(temp_tval),
            pos = epochs.info,
            axes = ax[i],
            contours = 3,
            mask = np.asarray(display_pval) <= 0.05,
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=6),
            cmap = "coolwarm",
            # vlim = (np.percentile(temp_tval, 5), np.percentile(temp_tval, 95))
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[i].set_title(f"{mindstate} > {which_ms}", font = bold_font, fontsize=12)
    fig.suptitle(f"{interest}", font = bold_font, fontsize=16)
    fig.tight_layout(pad = 2)
    plt.savefig(os.path.join(
        bandpowerPath, "figs", f"{interest}_{investigate_st}_compared_to_{which_ms}_sleepi_corr.png"
        ), 
                dpi=300)
    
# %% MS LME SUBTYPE – common colorbar per feature

vlims = {
    "N1" : {
        "delta" : (-4, 4),
        "theta" : (-5.5, 5.5),
        "alpha" : (-5.5, 5.5),
        "beta" : (-4, 4),
        "gamma" : (-4, 4)
        },
    "HS" : {
        "delta" : (-3, 3),
        "theta" : (-3, 3),
        "alpha" : (-3, 3),
        "beta" : (-3, 3),
        "gamma" : (-3, 3)
        },
    "HI" : {
        "delta" : (-2.5, 2.5),
        "theta" : (-4.5, 4.5),
        "alpha" : (-2.5, 2.5),
        "beta" : (-4, 4),
        "gamma" : (-4, 4)
        }
    }

investigate_st = "HI"
n1_df = df[df.subtype == investigate_st]
fdr_corrected = False
mindstates = ["MW", "MB", "HALLU", "FORGOT"]
ms_ref = "ON"

for interest in cols_power:
    model = f"{interest} ~ C(mindstate, Treatment('{ms_ref}'))"
    
    sub_p = interest.split('_')[-1]

    # 1) first pass: fit all contrasts, store t- & p-values
    tvals_dict = {}
    pvals_dict = {}
    all_tvals = []
    for ms in mindstates:
        temp_tval = []
        temp_pval = []
        cond_df = n1_df[n1_df.mindstate.isin([ms_ref, ms])]

        for chan in channels:
            subdf = (
                cond_df
                .loc[cond_df.channel == chan, ['sub_id','mindstate','channel', interest]]
                .dropna()
            )
            md = smf.mixedlm(model, subdf, groups=subdf['sub_id'], missing='omit')
            mdf = md.fit()
            term = f"C(mindstate, Treatment('{ms_ref}'))[T.{ms}]"

            if term in mdf.tvalues.index:
                temp_tval.append(mdf.tvalues[term])
                temp_pval.append(mdf.pvalues[term])
            else:
                temp_tval.append(np.nan)
                temp_pval.append(1.0)

        # replace any NaN with the row‐mean
        if np.any(np.isnan(temp_tval)):
            mean_t = np.nanmean(temp_tval)
            temp_tval = [mean_t if np.isnan(t) else t for t in temp_tval]

        tvals_dict[ms] = temp_tval
        pvals_dict[ms] = temp_pval
        all_tvals.extend(temp_tval)

    # 2) compute common color limits
    vmin, vmax = np.percentile(all_tvals, [5, 95])

    # 3) now plot with shared vlim
    fig, axes = plt.subplots(1, len(mindstates), figsize=(10, 3))
    for i, ms in enumerate(mindstates):
        temp_tval = tvals_dict[ms]
        temp_pval = pvals_dict[ms]

        if fdr_corrected:
            _, disp_pval = fdrcorrection(temp_pval)
        else:
            disp_pval = temp_pval

        if i == len(mindstates) - 1:
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)

        im, _ = mne.viz.plot_topomap(
            data=-np.asarray(temp_tval),
            pos=epochs.info,
            axes=axes[i],
            contours=2,
            mask=np.array(disp_pval) <= 0.05,
            mask_params=dict(
                marker='o', markerfacecolor='w',
                markeredgecolor='k', linewidth=0, markersize=6
            ),
            cmap="coolwarm",
            vlim=vlims[investigate_st][sub_p]
            )
        if i == len(mindstates) - 1:
            fig.colorbar(im, cax=cax, orientation='vertical')
        # axes[i].set_title(f"{ms_ref} > {ms}", font=bold_font, fontsize=12)

    # fig.suptitle(interest, font=bold_font, fontsize=16)
    fig.tight_layout(pad=2)
    outpath = os.path.join(
        bandpowerPath, "figs",
        f"{interest}_{investigate_st}_compared_to_ON.png"
    )
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

# %% ME Group, Corrected

to_permute = "subtype"
gp_1 = "HS"
gp_2 = "N1"
num_permutations = 200
redo = 0

vlims = (-2.5, 2.5)

for i_pt, power_type in enumerate(power_types) :
    
    fig, ax = plt.subplots(
        nrows = 1, 
        ncols = len(bands), 
        figsize = (12, 2.5)
        )
    
    for i_b, band in enumerate(bands) :
        
        feature = f"{power_type}_{band}"
        save_fname = os.path.join(
            bandpowerPath, 
            "figs", 
            f"CPerm_{num_permutations}_{power_type}_{band}_ME_Group.pkl"
            )
        
        if os.path.exists(save_fname) and not redo:
            print(f"Loading {power_type}_{band}...")
            out = pd.read_pickle(save_fname)
            
            orig_tvals           = out['orig_tvals']
            significant_clusters = out['significant_clusters']
        
        else :
            print(f"Processing {power_type}_{band}...")
            interest = f"C(subtype, Treatment('{gp_1}'))[T.{gp_2}]"
            model = f"{feature} ~ C(subtype, Treatment('{gp_1}'))" 
            cond_df = df.loc[df.subtype.isin([gp_1, gp_2])]
            
            clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, orig_pvals, orig_tvals = config.permute_and_cluster(
                cond_df,
                model, 
                interest,
                to_permute,
                num_permutations,
                neighbours,     
                clus_alpha,
                min_cluster_size,
                channels,
                )
            
            significant_clusters = config.identify_significant_clusters(
                clusters_pos, 
                clusters_neg, 
                perm_stats_pos, 
                perm_stats_neg, 
                montecarlo_alpha,
                num_permutations
                )
            
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
            
        significant_mask = np.zeros(len(channels), dtype=bool)
        for sign, clust_labels, stat, pval in significant_clusters:
            for ch in clust_labels:
                idx = np.where(channels == ch)[0][0]
                significant_mask[idx] = True
        
        if i_b == len(bands) - 1 :
            divider = make_axes_locatable(ax[i_b])
            cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = orig_tvals,
            pos = epochs.info,
            axes = ax[i_b],
            contours = 3,
            mask = significant_mask,
            mask_params = dict(
                marker='o', 
                markerfacecolor='w', 
                markeredgecolor='k',
                linewidth=0, 
                markersize=6
                ),
            cmap = "coolwarm",
            vlim = vlims
            )
        if i_b == len(bands) - 1 :
            fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[i_b].set_title(f"{feature}", font = bold_font, fontsize=12)
    # fig.suptitle(f"{gp_2}> {gp_1}", font = bold_font, fontsize=16)
    fig.tight_layout()
    
plt.savefig(os.path.join(
    bandpowerPath, "figs", f"{gp_2}_vs_{gp_1}_cluster_perm_{num_permutations}.png"
    ), dpi=300)

# %% Display Cluster Stats

for i_pt, power_type in enumerate(power_types) :
    for i_b, band in enumerate(bands) :
        
        feature = f"{power_type}_{band}"
        save_fname = os.path.join(
            bandpowerPath, 
            "figs", 
            f"CPerm_{num_permutations}_{power_type}_{band}_ME_Group.pkl"
            )
        
        if os.path.exists(save_fname) and not redo:
            print(f"{power_type}_{band}...")
            out = pd.read_pickle(save_fname)
            
            print(out['significant_clusters'])

# %% ME MS within GROUP | Corrected

to_permute = "mindstate"
num_permutations = 200
redo = 0

vlims = {
    "N1" : {
        "delta" : (-4, 4),
        "theta" : (-5.5, 5.5),
        "alpha" : (-5.5, 5.5),
        "beta" : (-4, 4),
        "gamma" : (-4, 4)
        },
    "HS" : {
        "delta" : (-3, 3),
        "theta" : (-3, 3),
        "alpha" : (-3, 3),
        "beta" : (-3, 3),
        "gamma" : (-3, 3)
        }
    }

investigate_st = "N1"
n1_df = df[df.subtype == investigate_st]
fdr_corrected = False
mindstates = ["ON", "MW", "MB", "FORGOT"]
ms_ref = "HALLU"

for feature in cols_power:
    
    fig, axes = plt.subplots(1, len(mindstates), figsize=(10, 3))
    model = f"{feature} ~ C(mindstate, Treatment('{ms_ref}'))"
    
    sub_p = feature.split('_')[-1]

    for i, ms in enumerate(mindstates):
        
        save_fname = os.path.join(
            bandpowerPath, 
            "figs", 
            f"CPerm_{num_permutations}_{feature}_in_{investigate_st}_MS_{ms_ref}_vs_{ms}.pkl"
            )
        if os.path.exists(save_fname) and not redo:
            print(f"Loading {feature}...")
            out = pd.read_pickle(save_fname)
            
            orig_tvals           = out['orig_tvals']
            significant_clusters = out['significant_clusters']
            
        else:
            print(f"Computing {feature} for {ms}...")
            ms_1 = ms_ref
            ms_2 = ms
            interest = f"C(mindstate, Treatment('{ms_ref}'))[T.{ms}]"
            cond_df = n1_df[n1_df.mindstate.isin([ms_ref, ms])]
            
            clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, orig_pvals, orig_tvals = config.permute_and_cluster(
                cond_df,
                model, 
                interest,
                to_permute,
                num_permutations,
                neighbours,     
                clus_alpha,
                min_cluster_size,
                channels,
                )
            
            significant_clusters = config.identify_significant_clusters(
                clusters_pos, 
                clusters_neg, 
                perm_stats_pos, 
                perm_stats_neg, 
                montecarlo_alpha,
                num_permutations
                )
            
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
            
        significant_mask = np.zeros(len(channels), dtype=bool)
        for sign, clust_labels, stat, pval in significant_clusters:
            for ch in clust_labels:
                idx = np.where(channels == ch)[0][0]
                significant_mask[idx] = True
                
        if i == len(mindstates) - 1 :
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
        im, _ = mne.viz.plot_topomap(
            data=-np.asarray(orig_tvals),
            pos=epochs.info,
            axes=axes[i],
            contours=3,
            mask=significant_mask,
            mask_params=dict(
                marker='o', 
                markerfacecolor='w',
                markeredgecolor='k', 
                linewidth=0, 
                markersize=6
            ),
            cmap="coolwarm",
            vlim=vlims[investigate_st][sub_p]
            )
        if i == len(mindstates) - 1 :
            fig.colorbar(im, cax=cax, orientation='vertical')
        axes[i].set_title(f"{ms_ref} > {ms}", font=bold_font, fontsize=12)
    
    # fig.suptitle(interest, font=bold_font, fontsize=16)
    fig.tight_layout(pad=2)
    outpath = os.path.join(
        bandpowerPath, 
        "figs",
        f"CPERM_{num_permutations}_{feature}_{investigate_st}_compared_to_{ms_ref}.png"
        )
    plt.savefig(outpath, dpi=300)
        
# %% 

cols_oi = ['rel_theta', 'rel_beta']

for feature in cols_oi:

    for i, ms in enumerate(mindstates):
        
        save_fname = os.path.join(
            bandpowerPath, 
            "figs", 
            f"CPerm_{num_permutations}_{feature}_in_{investigate_st}_MS_{ms_ref}_vs_{ms}.pkl"
            )
        if os.path.exists(save_fname) and not redo:
            print(f"\n{feature} | HA > {ms}...")
            out = pd.read_pickle(save_fname)
            
            orig_tvals           = out['orig_tvals']
            significant_clusters = out['significant_clusters']
            
            print(significant_clusters)
        