#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:29:28 2024

@author: arthurlecoz

03_02_explore_global_power.py
"""
# %% Paths
import os, numpy as np, pandas as pd
import SLHIP_config_ALC as config

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

from scipy.ndimage import gaussian_filter
from glob import glob
from scipy.stats import sem
from scipy.stats import t
from scipy.ndimage import label
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager

# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')
prop = font_manager.FontProperties(fname=font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()

cleanDataPath = config.cleanDataPath
powerPath = config.powerPath

aperiodicPath = os.path.join(powerPath, 'aperiodic')
reports_path = os.path.join(aperiodicPath, "reports")
fig_path = os.path.join(aperiodicPath, "figs")

aperiodic_files = glob(os.path.join(aperiodicPath, "*psd.pickle"))

channels = np.array(config.eeg_channels)

subtypes = ["HS", "N1"]
mindstates = ['ON', 'MW', 'MB', 'HALLU', 'FORGOT']
subtype_palette = ["#8d99ae", "#d00000"]
ms_palette = ["#FFC000", "#00B050", "#0070C0", "#7030A0", "#000000"]
midline = ["Fz", "Cz", "Pz", "Oz"]
freqs = np.linspace(0.5, 40, 159)

df = pd.read_csv(os.path.join(
    aperiodicPath, "all_aperiodic_psd.csv"
    ))
del df['Unnamed: 0']        

# %% df Manip

this_df = df.loc[df.subtype.isin(subtypes)]

# %% Pz ME Subtype

chan_oi = "Pz"

temp_df = this_df.loc[this_df.channel == chan_oi].copy().drop(
    columns=["n_probe", "n_block", "mindstate", "voluntary", "channel", "sleepiness"]
    ).groupby(
        ['sub_id', 'subtype', 'freq_bin'],
        as_index=False
        ).mean()

fig, ax = plt.subplots(
    nrows=1, 
    ncols=1, 
    figsize=(5, 5)
    )

sns.lineplot(
    data = temp_df,
    x = "freq_bin",
    y = "power_value", 
    hue = "subtype",
    hue_order = subtypes,
    palette = subtype_palette,
    legend=None,
    linewidth = 2,
    ax=ax
    )
sns.despine()

# Set the title and labels
ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize = 14)
ax.set_ylabel('Power', font = bold_font, fontsize = 14)
ax.set_xlim([0.5, 40])
# ax.set_ylim([-30, 60])

# Show the plot
fig.tight_layout()
# Save figure
outfile = os.path.join(fig_path, f"PSD_ME_subplots_Group_{chan_oi}.png")
fig.savefig(outfile, dpi=300)
print(f"Saved figure to {outfile}")


# %% MS Differences, Smoothed

smooth = 1

chan_oi = "Pz"

these_mindstates = mindstates[1:]
this_palette = [
    ["#FFC000", "#00B050"],
    ["#FFC000", "#0070C0"],
    ["#FFC000", "#7030A0"],
    ["#FFC000", "#000000"],
]

fig, axs = plt.subplots(
    nrows=1,
    ncols=len(these_mindstates),
    figsize=(15, 6),
    sharey=True,
    sharex=True
    )

for j, mindstate in enumerate(these_mindstates):
    ax = axs[j]

    for k, ms in enumerate(['ON', mindstate]):
        # Filter DataFrame for this mindstate, all subtypes & channels
        df_ms = this_df[
            (this_df['mindstate'] == ms) &
            (this_df['subtype'].isin(subtypes)) &
            (this_df['channel'] == chan_oi)
            ].copy().drop(columns="channel").groupby(
                ['sub_id', 'subtype', 'mindstate', 'freq_bin'],
                as_index=False
                ).mean()

        mean_power = df_ms[
            ['freq_bin', 'power_value']
            ].groupby('freq_bin').mean().power_value.values
        sem_power = df_ms[
            ['freq_bin', 'power_value']
            ].groupby('freq_bin').sem().power_value.values

        # LOWESS smoothing
        psd_db = gaussian_filter(mean_power, smooth)
        sem_db = gaussian_filter(sem_power, smooth)

        # Plot smoothed PSD
        ax.plot(
            freqs,
            psd_db,
            label=ms,
            color=this_palette[j][k],
            alpha=0.6,
            linewidth=2
            )
        # Fill ±SEM
        ax.fill_between(
            freqs,
            psd_db - sem_db,
            psd_db + sem_db,
            alpha=0.1,
            color=this_palette[j][k]
            )

    # Labels & title for this subplot
    ax.set_title(f"ON vs {mindstate}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize=16)

# Common y‐label
axs[0].set_ylabel('Power (dB)', font = bold_font, fontsize=16)

# Set limits on all subplots
for ax in axs:
    ax.set_xlim([0.5, 40])
    # ax.set_ylim([-130, -90])

sns.despine()
fig.tight_layout()

# Save figure
outfile = os.path.join(fig_path, f"PSD_ME_subplots_ONvsMS_{chan_oi}.png")
fig.savefig(outfile, dpi=300)
print(f"Saved figure to {outfile}")

# %% GRP Differences, smoothed

chan_oi = "Pz"
smooth = 1

fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(5, 6),
    sharey=True,
    sharex=True
    )

for j, subtype in enumerate(subtypes):

    df_ms = this_df[
        (this_df['subtype']== subtype) &
        (this_df['channel']== chan_oi)
        ].copy().drop(columns=['channel', 'mindstate']).groupby(
            ['sub_id', 'subtype', 'freq_bin'],
            as_index=False
            ).mean()

    mean_power = df_ms[
        ['freq_bin', 'power_value']
        ].groupby('freq_bin').mean().power_value.values
    sem_power = df_ms[
        ['freq_bin', 'power_value']
        ].groupby('freq_bin').sem().power_value.values

    # LOWESS smoothing
    psd_db = gaussian_filter(mean_power, smooth)
    sem_db = gaussian_filter(sem_power, smooth)

    # Plot smoothed PSD
    ax.plot(
        freqs,
        psd_db,
        color=subtype_palette[j],
        alpha=.9,
        linewidth=2
        )
    # Fill ±SEM
    ax.fill_between(
        freqs,
        psd_db - sem_db,
        psd_db + sem_db,
        alpha=0.2,
        color=subtype_palette[j]
        )

    # Labels & title for this subplot
    ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize=16)

# Common y‐label
ax.set_ylabel('Power (dB)', font = bold_font, fontsize=16)

ax.set_xlim([0.5, 40])

sns.despine()
fig.tight_layout()

# Save figure
outfile = os.path.join(fig_path, f"PSD_ME_subplots_GroupDiff_{chan_oi}.png")
fig.savefig(outfile, dpi=300)
print(f"Saved figure to {outfile}")

# %% 
            
# big_av_psd_savepath = os.path.join(
#     fig_dir, "fooof_averaged_psd_spectra_2.pickle"
#     )
# big_av_sem_savepath = os.path.join(
#     fig_dir, "fooof_sem_spectra_2.pickle"
#     )
                
# with open(big_av_psd_savepath, 'wb') as handle:
#     pickle.dump(big_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(big_av_sem_savepath, 'wb') as handle:
#     pickle.dump(dic_sem, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% AvChan | Average Group

fig, ax = plt.subplots(
    nrows=1, 
    ncols=1, 
    figsize=(5, 6)
    )

for j, subtype in enumerate(subtypes):
    # Convert power to dB
    psd_db = lowess(np.nanmean([dic_psd[subtype][ms][ch]
        for ch in channels for ms in mindstates
        ], axis = 0), freqs, 0.050)[:, 1]

    # Calculate the SEM
    sem_db = lowess(np.nanmean([dic_sem[subtype][ms][ch]
        for ch in channels for ms in mindstates
        ], axis = 0), freqs, 0.050)[:, 1]

    # Plot the PSD and SEM
    ax.plot(
        freqs, 
        psd_db, 
        label = subtype, 
        color = subtype_palette[j],
        alpha = .7,
        linewidth = 2,
        )
    ax.fill_between(
        freqs, 
        psd_db - sem_db, 
        psd_db + sem_db, 
        alpha= 0.2, 
        color = subtype_palette[j]
        )

ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize = 16)
ax.set_ylabel('Power (dB)', font = bold_font, fontsize = 16)

ax.set_xlim([0.5, 40])
ax.set_ylim([-130, -90])
sns.despine()
fig.tight_layout()
plt.savefig(os.path.join(
    fig_path, "PSD_ME_groupe.png"
    ), dpi=300)

# %% AvChan | Average Group

fig, ax = plt.subplots(
    nrows=1, 
    ncols=1, 
    figsize=(5, 6)
    )

for j, mindstate in enumerate(mindstates):
    # Convert power to dB
    psd_db = lowess(np.nanmean([dic_psd[st][mindstate][ch]
        for ch in channels for st in subtypes
        ], axis = 0), freqs, 0.075)[:, 1]

    # Calculate the SEM
    sem_db = lowess(np.nanmean([dic_sem[st][mindstate][ch]
        for ch in channels for st in subtypes
        ], axis = 0), freqs, 0.075)[:, 1]

    # Plot the PSD and SEM
    ax.plot(
        freqs, 
        psd_db, 
        label = subtype, 
        color = ms_palette[j],
        alpha = .6,
        linewidth = 2,
        )
    ax.fill_between(
        freqs, 
        psd_db - sem_db, 
        psd_db + sem_db, 
        alpha= 0.1, 
        color = ms_palette[j]
        )

ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize = 16)
ax.set_ylabel('Power (dB)', font = bold_font, fontsize = 16)

ax.set_xlim([0.5, 40])
ax.set_ylim([-130, -90])
sns.despine()
fig.tight_layout()
plt.savefig(os.path.join(
    fig_path, "PSD_ME_MS.png"
    ), dpi=300)

# %% AvChan | Average Group


these_mindstates = mindstates[1:]
this_palette = [
    ["#FFC000", "#00B050"],
    ["#FFC000", "#0070C0"],
    ["#FFC000", "#7030A0"],
    ["#FFC000", "#000000"],
    ]

fig, axs = plt.subplots(
    nrows=1, 
    ncols=4, 
    figsize=(15, 6),
    sharey=True,
    sharex=True
    )

for j, mindstate in enumerate(these_mindstates):

    for k, ms in enumerate(['ON', mindstate]) :
    
        psd_db = lowess(np.nanmean([dic_psd[st][ms][ch]
            for ch in channels for st in subtypes
            ], axis = 0), freqs, 0.075)[:, 1]
    
        sem_db = lowess(np.nanmean([dic_sem[st][ms][ch]
            for ch in channels for st in subtypes
            ], axis = 0), freqs, 0.075)[:, 1]
    
        axs[j].plot(
            freqs, 
            psd_db, 
            label = subtype, 
            color = this_palette[j][k],
            alpha = .6,
            linewidth = 2
            )
        axs[j].fill_between(
            freqs, 
            psd_db - sem_db, 
            psd_db + sem_db, 
            alpha= 0.1, 
            color = this_palette[j][k]
            )

    axs[j].set_xlabel('Frequency (Hz)', font = bold_font, fontsize = 16)
axs[0].set_ylabel('Power (dB)', font = bold_font, fontsize = 16)

ax.set_xlim([0.5, 40])
ax.set_ylim([-130, -90])
sns.despine()
fig.tight_layout()
plt.savefig(os.path.join(
    fig_path, "PSD_ME_subplots_ONvsMS.png"
    ), dpi=300)


# %% Midline | ST DIFFERENCES

fig, axs = plt.subplots(
    nrows=1, 
    ncols=len(midline), 
    figsize=(20, 6), 
    sharey=True, 
    layout = "constrained"
    )
for i, channel in enumerate(midline):
    ax = axs[i]

    # Loop through each population and plot its PSD and SEM
    for j, subtype in enumerate(subtypes):
        # Convert power to dB
        psd_db = lowess(np.nanmean([dic_psd[subtype][ms][channel]
            for ms in mindstates
            ], axis = 0), freqs, 0.075)[:, 1]

        # Calculate the SEM
        sem_db = lowess(np.nanmean([dic_sem[subtype][ms][channel]
            for ms in mindstates
            ], axis = 0), freqs, 0.075)[:, 1]

        # Plot the PSD and SEM
        ax.plot(
            freqs, 
            psd_db, 
            label = subtype, 
            color = subtype_palette[j],
            alpha = .7,
            linewidth = 2
            )
        ax.fill_between(
            freqs, 
            psd_db - sem_db, 
            psd_db + sem_db, 
            alpha= 0.2, 
            color = subtype_palette[j]
            )

    # Set the title and labels
    ax.set_title('Channel: ' + channel)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0.5, 40])
    # ax.set_ylim([-30, 60])
    ax.legend()

# Add the condition name as a title for the entire figure
fig.suptitle('Averaged MS - ST differences in Midline')

# Add a y-axis label to the first subplot
axs[0].set_ylabel('Power (dB)')
for i in range(len(midline)) :
    if i < len(midline)-1:
        axs[i].get_legend().set_visible(False)

# Adjust the layout of the subplots
# plt.constrained_layout()

# Show the plot
plt.show()

# %% Midline | MS DIFFERENCES

fig, axs = plt.subplots(
    nrows=1, 
    ncols=len(midline), 
    figsize=(20, 6), 
    sharey=True, 
    layout = "constrained"
    )
for i, channel in enumerate(midline):
    ax = axs[i]

    # Loop through each population and plot its PSD and SEM
    for j, ms in enumerate(mindstates):
        # Convert power to dB
        psd_db = lowess(np.nanmean([dic_psd[subtype][ms][channel]
            for subtype in subtypes
            ], axis = 0), freqs, 0.095)[:, 1]

        # Calculate the SEM
        sem_db = lowess(np.nanmean([dic_sem[subtype][ms][channel]
            for subtype in subtypes
            ], axis = 0), freqs, 0.095)[:, 1]

        # Plot the PSD and SEM
        ax.plot(
            freqs, 
            psd_db, 
            label = ms, 
            color = ms_palette[j],
            alpha = .7,
            linewidth = 2
            )
        ax.fill_between(
            freqs, 
            psd_db - sem_db, 
            psd_db + sem_db, 
            alpha= 0.2, 
            color = ms_palette[j]
            )

    # Set the title and labels
    ax.set_title('Channel: ' + channel)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0.5, 40])
    # ax.set_ylim([-30, 60])
    ax.legend()

# Add the condition name as a title for the entire figure
fig.suptitle('Averaged Subtypes - Mindstates differences in Midline')

# Add a y-axis label to the first subplot
axs[0].set_ylabel('Power (dB)')
for i in range(len(midline)) :
    if i < len(midline)-1:
        axs[i].get_legend().set_visible(False)

# Adjust the layout of the subplots
# plt.constrained_layout()

# Show the plot
plt.show()

# %% Midline | MS 

# palette = ["#5e6472", "#faa307"]

for ms in mindstates:
    # Create a new figure with three subplots
    fig, axs = plt.subplots(
        nrows=1, 
        ncols=len(midline), 
        figsize=(20, 6), 
        sharey=True, 
        layout = "constrained"
        )

    # Loop through each channel
    for i, channel in enumerate(midline):
        ax = axs[i]

        # Loop through each population and plot its PSD and SEM
        for j, subtype in enumerate(subtypes):
            # Convert power to dB
            psd_db = dic_psd[subtype][ms][channel]

            # Calculate the SEM
            sem_db = dic_sem[subtype][ms][channel]

            # Plot the PSD and SEM
            ax.plot(
                freqs, 
                psd_db, 
                label = subtype, 
                color = palette[j],
                alpha = .7,
                linewidth = 2
                )
            ax.fill_between(
                freqs, 
                psd_db - sem_db, 
                psd_db + sem_db, 
                alpha= 0.2, 
                color = palette[j]
                )

        # Set the title and labels
        ax.set_title('Channel: ' + channel)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xlim([0.5, 40])
        # ax.set_ylim([-30, 60])
        ax.legend()

    # Add the condition name as a title for the entire figure
    fig.suptitle('Condition: ' + ms)

    # Add a y-axis label to the first subplot
    axs[0].set_ylabel('Power (dB)')
    for i in range(len(midline)) :
        if i < len(midline)-1:
            axs[i].get_legend().set_visible(False)

    # Adjust the layout of the subplots
    # plt.constrained_layout()

    # Show the plot
    plt.show()
    fig_savename = (f"{fig_path}/PSD_plot_{ms}.png")
    # plt.savefig(fig_savename, dpi = 300)

# %% dics to DF

coi = ["sub_id", "subtype", "session",
       # "age", "gender", 
       "mindstate", "channel", 
       "freq_bin", "aperio_power"]

big_dic = {c : [] for c in coi}

for i, file in enumerate(aperiodic_files) :
    this_dic = pd.read_pickle(file)
    sub_id = file.split('aperiodic/')[-1].split('_aper')[0]
    subtype = sub_id[:2]
    session = sub_id[-2:]
    sub_id = sub_id[:-3]
    if len(subtypes)<3 :
        if subtype == "HI" : continue
    
    print(f"Processing : {sub_id}... [{i+1} / {len(aperiodic_files)}]")
    
    # this_demo = df_demographics.loc[df_demographics.code == sub_id]
    # this_age = this_demo.age.iloc[0]
    # this_gender = this_demo.sexe.iloc[0]
    
    for ms in mindstates:
        for channel in channels:
            for i_f, freq_bin in enumerate(freqs) :
                if len(this_dic[ms][channel]) < 1 :
                    big_dic["aperio_power"].append(np.nan)
                else : 
                    big_dic["aperio_power"].append(10 * np.log10(this_dic[ms][channel][0][i_f]))
                big_dic["sub_id"].append(sub_id)
                big_dic["subtype"].append(subtype)
                big_dic["session"].append(session)
                big_dic["mindstate"].append(ms)
                big_dic["channel"].append(channel)
                big_dic["freq_bin"].append(freq_bin)

df = pd.DataFrame.from_dict(big_dic) 

# %% LME + PLOTS

this_channels = "Cz"
subtypes_here = ["HS", "N1"]
palette_here = [subtype_palette[0], subtype_palette[1]]  

# Prepare the figure
num_channels = len(channels)
fig, axs = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(5, 6),
    sharey=True
    )

# Ensure axs is iterable
if num_channels == 1:
    axs = [axs]

for idx, channel in enumerate(this_channels):
    print(f"Processing channel: {channel}")

    # Filter the data for the specified mindstate and channel
    this_df = df.loc[
        (df.subtype.isin(subtypes_here)) &
        (df.channel == channel)]

    # Define the model formula
    model_formula = 'aperio_power ~ C(subtype, Treatment("HS"))'
    coef_name = f'C(subtype, Treatment("{subtypes_here[0]}"))[T.{subtypes_here[1]}]'

    t_values = []
    p_values = []

    # Extract t-values and p-values for each frequency bin
    for freq_bin in freqs:
        temp_df = this_df.loc[this_df.freq_bin == freq_bin]
        model = smf.mixedlm(model_formula, temp_df, groups=temp_df['sub_id'], missing='drop')
        model_result = model.fit()

        
        t_values.append(model_result.tvalues[coef_name])
        p_values.append(model_result.pvalues[coef_name])

    # Thresholding
    df_resid = model_result.df_resid
    t_thresh = t.ppf(1 - 0.025, df_resid)  # Two-tailed test
    significant_bins = np.array(t_values) > t_thresh

    # Identify clusters
    clusters, num_clusters = label(significant_bins)

    # Compute cluster-level statistics
    cluster_stats = []
    for i in range(1, num_clusters + 1):
        cluster_t_values = np.array(t_values)[clusters == i]
        cluster_stat = cluster_t_values.sum()
        cluster_stats.append(cluster_stat)

    # Permutation testing
    n_permutations = 10  # Increase this number for more accurate p-values
    max_cluster_stats = []

    for perm in range(n_permutations):
        # Permute subtype labels
        permuted_subtypes = this_df['subtype'].sample(frac=1, replace=False).values
        this_df_permuted = this_df.copy()
        this_df_permuted['subtype'] = permuted_subtypes

        perm_t_values = []
        for freq_bin in freqs:
            temp_df = this_df_permuted.loc[this_df_permuted.freq_bin == freq_bin]
            model = smf.mixedlm(model_formula, temp_df, groups=temp_df['sub_id'], missing='drop')
            model_result = model.fit()
            perm_t_values.append(model_result.tvalues[coef_name])

        # Threshold and identify clusters in permuted data
        perm_significant_bins = np.array(perm_t_values) > t_thresh
        perm_clusters, perm_num_clusters = label(perm_significant_bins)

        # Compute cluster statistics
        perm_cluster_stats = []
        for i in range(1, perm_num_clusters + 1):
            cluster_t_values = np.array(perm_t_values)[perm_clusters == i]
            perm_cluster_stat = cluster_t_values.sum()
            perm_cluster_stats.append(perm_cluster_stat)

        if perm_cluster_stats:
            max_cluster_stats.append(max(perm_cluster_stats))
        else:
            max_cluster_stats.append(0)

    # Compute corrected p-values
    corrected_p_values = []
    for observed_stat in cluster_stats:
        p_value = np.mean(np.array(max_cluster_stats) >= observed_stat)
        corrected_p_values.append(p_value)

    # Identify significant clusters
    alpha = 0.05
    significant_cluster_labels = [
        cluster_label for cluster_label, p_val in zip(range(1, num_clusters + 1), corrected_p_values)
        if p_val < alpha
    ]

    # Create the mask
    mask = np.zeros_like(clusters, dtype=bool)
    for cluster_label in significant_cluster_labels:
        mask[clusters == cluster_label] = True

    # Plotting
    ax = axs[idx]
    for j, subtype in enumerate(subtypes_here):
        # Get the PSD and SEM data
        psd_db = dic_psd[subtype][mindstate][channel]
        sem_db = dic_sem[subtype][mindstate][channel]

        # Plot the PSD and SEM
        ax.plot(
            freqs,
            psd_db,
            label=subtype,
            color=palette_here[j],
            alpha=0.9,
            linewidth=3
        )
        ax.fill_between(
            freqs,
            psd_db - sem_db,
            psd_db + sem_db,
            alpha=0.3,
            color=palette_here[j]
        )

    # Highlight significant clusters on the PSD plot
    if mask.any():
        # Adjust y-position for the significance marker as needed
        y_max = max(psd_db + sem_db)
        y_min = min(psd_db - sem_db)
        y_range = y_max - y_min
        significance_line_y = y_max + 0.05 * y_range

        # Plot significance line over significant frequency bins
        significant_freqs = np.array(freqs)[mask]
        ax.hlines(
            y=significance_line_y,
            xmin=significant_freqs[0],
            xmax=significant_freqs[-1],
            color='k',
            linewidth=4,
            label='Significant Cluster'
        )

    ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize = 18)
    if idx == 0:
        ax.set_ylabel('Power (dB)', font = bold_font, fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim([min(freqs), max(freqs)])

    sns.despine(ax=ax)

# Add legend and overall title
# axs[-1].legend(fontsize=12)
plt.show()
plt.savefig(os.path.join(fig_path, f"aperi_lme_hi_cns_{ms}.png"), dpi = 300)



