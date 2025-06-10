#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:13:31 2025

@author: arthurlecoz

03_06_explore_band_power.py
"""
# %% Paths & Packages

import mne, os, numpy as np, pandas as pd
import SLHIP_config_ALC as config
import matplotlib.pyplot as plt

from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import zscore

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection

from matplotlib.font_manager import FontProperties
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

cleanDataPath = config.cleanDataPath
powerPath = config.powerPath

bandpowerPath = os.path.join(powerPath, 'bandpower')
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

info = epochs.info
channel_order = np.array(epochs.ch_names)

df = pd.read_csv(os.path.join(bandpowerPath, "bandpower_all_ms.csv"))
df['daytime'] = [sub_id[-2:] for sub_id in df.sub_id.values]
df['sub_id'] = [sub_id[:-3] for sub_id in df.sub_id.values]
del df['Unnamed: 0']

# %% DataFrame Manipulation

mean_df = df[['sub_id', 'subtype', 'channel', 'mindstate', 'abs_delta', 'abs_theta',
       'abs_alpha', 'abs_beta', 'abs_gamma', 'rel_delta', 'rel_theta',
       'rel_alpha', 'rel_beta', 'rel_gamma', ]].groupby(
    ['sub_id', 'subtype', 'channel', 'mindstate'], 
    as_index=False
    ).mean()

# %% Plotting 

subtypes = ["HS", "N1", "HI"]

for feature in cols_power :
    fig, ax = plt.subplots(
        nrows = 1, ncols = len(subtypes), figsize = (16, 4), layout = 'tight')
    columns = [feature, "channel"]
    
    vmin = df[columns].groupby(
        'channel', as_index = False).mean().min()[feature]
    vmax = df[columns].groupby(
        'channel', as_index = False).mean().max()[feature]
    
    for i_st, subtype in enumerate(subtypes) :
        
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
    
        divider = make_axes_locatable(ax[i_st],)
        cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = values,
            pos = info,
            axes = ax[i_st],
            contours = 2,
            cmap = "Purples",
            vlim = (vmin, vmax)
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
        ax[i_st].set_title(f"{subtype}", fontweight = "bold")
    
    fig.suptitle(f"{feature}\n", 
             fontsize = "xx-large", 
             fontweight = "bold")   
    
    plt.show()  
    plt.savefig(os.path.join(
                bandpowerPath, "figs", f"{feature}_topo.png"
                ), dpi = 300
        )

# %% ME - Subtype LME

fdr_corrected = 0

for interest in cols_power :
    model = f"{interest} ~ C(subtype, Treatment('HS'))" 
    
    fig, ax = plt.subplots(
        nrows = 1, ncols = 2, figsize = (8, 4))
    for i, subtype in enumerate(["N1", "HI"]):
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = df.loc[df.subtype.isin(['HS', subtype])]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'subtype', 'channel', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
            mdf = md.fit()
            temp_tval.append(mdf.tvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"])
            temp_pval.append(mdf.pvalues[f"C(subtype, Treatment('HS'))[T.{subtype}]"])
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
            cmap = "coolwarm",
            vlim = (np.percentile(temp_tval, 5), np.percentile(temp_tval, 95))
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[i].set_title(f"{subtype} > HS", font = bold_font, fontsize=12)
    fig.suptitle(f"{interest}", font = bold_font, fontsize=16)
    fig.tight_layout(pad = 2)
    plt.savefig(os.path.join(
        bandpowerPath, "figs", f"{interest}_ME_group_compared_to_HS.png"
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
        cond_df = df.loc[df.mindstate.isin(['HALLU', ms])]
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

investigate_st = "N1"

n1_df = df.loc[df.subtype==investigate_st]

fdr_corrected = 0

for interest in cols_power :
    model = f"{interest} ~ C(mindstate, Treatment('HALLU'))" 
    fig, ax = plt.subplots(
        nrows = 1, ncols = 4, figsize = (10, 3))
    
    for i, mindstate in enumerate(["ON", "MW", "MB", "FORGOT"]):
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = n1_df.loc[n1_df.mindstate.isin(['HALLU', mindstate])]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'mindstate', 'channel', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
            mdf = md.fit()
            temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('HALLU'))[T.{mindstate}]"])
            temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('HALLU'))[T.{mindstate}]"])
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
    
        ax[i].set_title(f"{mindstate} > HALLU", font = bold_font, fontsize=12)
    fig.suptitle(f"{interest}", font = bold_font, fontsize=16)
    fig.tight_layout(pad = 2)
    plt.savefig(os.path.join(
        bandpowerPath, "figs", f"{interest}_{investigate_st}_compared_to_on.png"
        ), 
                dpi=300)

# %% Test - OFF & FORGOT

onoff_df = df.copy()
onoff_df['mindstate'] = onoff_df['mindstate'].replace(
    {'MW_I': 'OFF', 'MB': 'OFF'}
    ) 

# %% MS LME per SUBTYPE

investigate_st = "N1"

n1_df = onoff_df.loc[onoff_df.subtype==investigate_st]

fdr_corrected = 0

for interest in cols_power :
    model = f"{interest} ~ C(mindstate, Treatment('HALLU'))" 
    fig, ax = plt.subplots(
        nrows = 1, ncols = 2, figsize = (6, 3))
    
    for i, mindstate in enumerate(["ON", "OFF"]):
        temp_tval = []; temp_pval = []; chan_l = []
        cond_df = n1_df.loc[n1_df.mindstate.isin(['HALLU', mindstate])]
        for chan in channels :
            subdf = cond_df[
                ['sub_id', 'mindstate', 'channel', f'{interest}']
                ].loc[(cond_df.channel == chan)].dropna()
            md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
            mdf = md.fit()
            temp_tval.append(mdf.tvalues[f"C(mindstate, Treatment('HALLU'))[T.{mindstate}]"])
            temp_pval.append(mdf.pvalues[f"C(mindstate, Treatment('HALLU'))[T.{mindstate}]"])
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
            cmap = "autumn",
            vlim = (np.percentile(temp_tval, 5), np.percentile(temp_tval, 95))
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
        ax[i].set_title(f"{mindstate} > HALLU", font = bold_font, fontsize=12)
    fig.suptitle(f"{interest}", font = bold_font, fontsize=16)
    fig.tight_layout(pad = 2)
    plt.savefig(os.path.join(
        bandpowerPath, "figs", f"{interest}_onoff_compared_to_hallu.png"
        ), 
                dpi=300)

