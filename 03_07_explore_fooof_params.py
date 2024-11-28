#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:09:25 2024

@author: arthurlecoz

03_07_explore_fooof_params.py
"""
# %% Paths
import mne 
import os 

import numpy as np
import pandas as pd
import SLHIP_config_ALC as config
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.font_manager import FontProperties
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

cleanDataPath = config.cleanDataPath
powerPath = config.powerPath

fooofparamsPath = os.path.join(powerPath, 'fooof_params')
reports_path = os.path.join(fooofparamsPath, "reports")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

channels = np.array(config.eeg_channels)

subtypes = ["C1", "HI", "N1"]
psd_palette = ["#8d99ae", "#d00000", "#ffb703"]

freqs = np.linspace(0.5, 40, 159)

method = "welch"
fmin = 0.5
fmax = 40
n_fft = 1024
n_per_seg = n_fft
n_overlap = int(n_per_seg/2)
window = "hamming"

df = pd.read_csv(os.path.join(
    fooofparamsPath, "fooof_params.csv"
    ))
df = df.loc[(df.r_squared > .95) & (df.error < .1)]
del df['Unnamed: 0']
mean_df = df.groupby(
    ['sub_id', 'subtype', 'mindstate', 'channel'],
    as_index = False
    ).mean()

# %% Plotting (should be on another script for cleanness)

epochs = mne.read_epochs(
    glob(os.path.join(cleanDataPath, "epochs_probes", "*.fif")
         )[0], 
    preload = True
    )

info = epochs.info
channel_order = np.array(epochs.ch_names)
subtypes = ["HS", "N1", "HI"]

for feature in ['offset', 'exponent'] :
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

# %% Statistical comparisons LME

mean_df = df.groupby(
    ['sub_id', 'subtype', 'channel', 'mindstate'], 
    as_index = False).mean()

interest = 'offset'
fdr_corrected = 0

model = f"{interest} ~ C(subtype, Treatment('HS'))" 

fig, ax = plt.subplots(
    nrows = 1, ncols = 2, figsize = (8, 4))
for i, subtype in enumerate(["N1", "HI"]):
    temp_tval = []; temp_pval = []; chan_l = []
    cond_df = mean_df.loc[mean_df.subtype.isin(['HS', subtype])]
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
        cmap = "viridis",
        # vlim = (np.percentile(temp_tval, 5), np.percentile(temp_tval, 95))
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')

    ax[i].set_title(f"{subtype} > HS", font = bold_font, fontsize=12)
fig.suptitle(f"{interest}", font = bold_font, fontsize=16)
fig.tight_layout(pad = 2)

