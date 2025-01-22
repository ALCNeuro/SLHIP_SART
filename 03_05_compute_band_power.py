#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Nov 12 16:10:39 2024

@author: arthurlecoz

03_05_compute_band_power.py

"""
# %% Paths
import mne, os, numpy as np, pandas as pd
import SLHIP_config_ALC as config

from glob import glob

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

freq_bands = {
     'delta': (.5, 4),
     'theta': (4, 8),
     'alpha': (8, 12),
     'beta': (12, 30),
     'gamma': (30, 40)
     }

threshold = dict(eeg = 400e-6)

files = glob(os.path.join(cleanDataPath, "epochs_probes", "*.fif"))

coi = ['sub_id', 'subtype', 'channel', 'mindstate', 
   'abs_delta','abs_theta','abs_alpha','abs_beta','abs_gamma',
   'rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma']

cols_power = [
    'abs_delta','abs_theta','abs_alpha','abs_beta','abs_gamma',
    'rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma'
    ]

# %% Loop

mindstates = ['ON', 'MW', 'HALLU', 'MB']

bigdic = {f : [] for f in coi}
    
for i_file, file in enumerate(files) :
    sub_id = file.split('probes/')[-1].split('_epo')[0]
    subtype = sub_id[:2]

    print(f"...processing {sub_id}")
    
    epochs = mne.read_epochs(file, preload = True)
    epochs.drop_bad(threshold) # If you already cleaned the data or don't care, comment this line
    
    metadata = epochs.metadata
    
    for ms in mindstates:
        print(f'processing {ms}')
        if ms not in metadata.mindstate.unique() : continue
        temp_list = []
        temp_power = epochs[epochs.metadata.mindstate == ms].compute_psd(
                method = method,
                fmin = fmin, 
                fmax = fmax,
                n_fft = n_fft,
                n_overlap = n_overlap,
                n_per_seg = n_per_seg,
                window = window,
                picks = channels
                )
            
        for i_epoch in range(
                len(epochs[epochs.metadata.mindstate == ms])
                ) :
            this_power = np.squeeze(temp_power[i_epoch])
            
            abs_bandpower_ch = {
                 f"abs_{band}": np.nanmean(this_power[:,
                         np.logical_and(freqs >= borders[0], freqs <= borders[1])
                         ], axis = 1)
                 for band, borders in freq_bands.items()}
            
            total_power = np.sum(
                [abs_bandpower_ch[f"abs_{band}"] 
                 for band, borders in freq_bands.items()]
                )
            
            rel_bandpower_ch = {
                f"rel_{band}" : abs_bandpower_ch[f"abs_{band}"] / total_power
                for band in freq_bands.keys()
                }
            
            for i_ch, channel in enumerate(channels) :
                bigdic['sub_id'].append(sub_id)
                bigdic['subtype'].append(subtype)
                bigdic['mindstate'].append(ms)
                bigdic['channel'].append(channel)
                for col in cols_power :
                    if col.startswith('abs'):
                        bigdic[col].append(abs_bandpower_ch[col][i_ch])
                    if col.startswith('rel'):
                        bigdic[col].append(rel_bandpower_ch[col][i_ch])
                
df = pd.DataFrame.from_dict(bigdic)
df.to_csv(os.path.join(bandpowerPath, "bandpower.csv"))

# %% Plotting (should be on another script for cleanness)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if epochs not in locals() :
    epochs = mne.read_epochs(files[0], preload = True)
    epochs.pick('eeg')

info = epochs.info
channel_order = np.array(epochs.ch_names)
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

# %% Statistical comparisons LME

mean_df = df.groupby(
    ['sub_id', 'subtype', 'channel', 'mindstate'], 
    as_index = False).mean()

interest = 'abs_beta'
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
        vlim = (np.percentile(temp_tval, 5), np.percentile(temp_tval, 95))
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')

    ax[i].set_title(f"{subtype} > HS", font = bold_font, fontsize=12)
fig.suptitle(f"{interest}", font = bold_font, fontsize=16)
fig.tight_layout(pad = 2)
