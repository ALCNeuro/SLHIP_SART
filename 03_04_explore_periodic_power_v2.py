#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:33:14 2024

@author: arthurlecoz

03_04_explore_periodic_power.py
"""
# %% Paths
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import SLHIP_config_ALC as config
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

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

periodicPath = os.path.join(powerPath, 'periodic')
reports_path = os.path.join(periodicPath, "reports")
fig_path = os.path.join(periodicPath, "figs")

periodic_files = glob(os.path.join(periodicPath, "*psd.pickle"))

subtypes = ["HS", "N1"]
channels = np.array(config.eeg_channels)
mindstates = ['ON', 'MW', 'MB', 'HALLU', 'FORGOT']
subtype_palette = ["#8d99ae", "#d00000"]
ms_palette = ["#FFC000", "#00B050", "#0070C0", "#7030A0", "#000000"]
freqs = np.linspace(0.5, 40, 159)
midline = ["Fz", "Cz", "Pz", "Oz"]

df = pd.read_csv(os.path.join(
    periodicPath, "all_periodic_psd.csv"
    ))
del df['Unnamed: 0']

# %% df manip

this_df = df.loc[df.subtype.isin(subtypes)]

# %% Pz ME Subtype

temp_df = this_df.loc[this_df.channel == "Fz"].copy().drop(
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
ax.set_xlim([0.5, 40])
# ax.set_ylim([-30, 60])

ax.set_ylabel('Power (dB)')

# Show the plot
fig.tight_layout()

# %% MS Differences, Smoothed

chan_oi = ["Cz"] # must be a list

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
            (this_df['channel'].isin(chan_oi))
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
        psd_db = lowess(mean_power, freqs, frac=0.095)[:, 1]
        sem_db = lowess(sem_power, freqs, frac=0.095)[:, 1]

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
outfile = os.path.join(fig_path, "PSD_ME_subplots_ONvsMS_Cz.png")
fig.savefig(outfile, dpi=300)
print(f"Saved figure to {outfile}")

# %% GRP Differences, smoothed

chan_oi = "Cz"

these_mindstates = mindstates[1:]
this_palette = [
    ["#FFC000", "#00B050"],
    ["#FFC000", "#0070C0"],
    ["#FFC000", "#7030A0"],
    ["#FFC000", "#000000"],
]

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
        (this_df['channel'].isin([chan_oi]))
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
    psd_db = lowess(mean_power, freqs, frac=0.08)[:, 1]
    sem_db = lowess(sem_power, freqs, frac=0.08)[:, 1]

    # Plot smoothed PSD
    ax.plot(
        freqs,
        psd_db,
        label=ms,
        color=subtype_palette[j],
        alpha=0.6,
        linewidth=2
        )
    # Fill ±SEM
    ax.fill_between(
        freqs,
        psd_db - sem_db,
        psd_db + sem_db,
        alpha=0.1,
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
outfile = os.path.join(fig_path, "PSD_ME_subplots_GroupDiff_Cz.png")
fig.savefig(outfile, dpi=300)
print(f"Saved figure to {outfile}")

# %% import numpy as np

fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(5, 6),
    sharey=True,
    sharex=True
    )

for j, subtype in enumerate(subtypes):
    # Filter DataFrame for this mindstate, all subtypes & channels
    df_ms = this_df[
        (this_df['subtype'].isin([subtype])) &
        (this_df['channel'].isin(channels))
        ]

    # Group by freq_bin and compute mean & SEM of power_value
    grp = df_ms.groupby('freq_bin')['power_value']
    mean_power = grp.mean().reindex(freqs).values
    sem_power  = grp.sem().reindex(freqs).values

    # LOWESS smoothing
    psd_db = lowess(mean_power, freqs, frac=0.095)[:, 1]
    sem_db = lowess(sem_power, freqs, frac=0.095)[:, 1]

    # Plot smoothed PSD
    ax.plot(
        freqs,
        psd_db,
        label=ms,
        color=subtype_palette[j],
        alpha=0.6,
        linewidth=2
        )
    # Fill ±SEM
    ax.fill_between(
        freqs,
        psd_db - sem_db,
        psd_db + sem_db,
        alpha=0.1,
        color=subtype_palette[j]
        )

# Labels & title for this subplot
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
# outfile = os.path.join(fig_path, "PSD_ME_subplots_ONvsMS.png")
# fig.savefig(outfile, dpi=300)
# print(f"Saved figure to {outfile}")

# %% Pz ME MS 

temp_df = this_df.loc[this_df.channel == "Pz"].copy().drop(
    columns=["n_probe", "n_block", "voluntary", "channel", "sleepiness"]
    ).groupby(
        ['sub_id', 'subtype', "mindstate", 'freq_bin'],
        as_index=False
        ).mean()
        
x = "freq_bin"
y = "power_value"
hue = "subtype"

fig, axs = plt.subplots(
    figsize = [12, 8],
    ncols = len(mindstates),
    nrows = 1,
    sharey = True,
    sharex = True
    )

for i, ms in enumerate(mindstates):
    ax = axs[i]
    sns.lineplot(
        data = temp_df.loc[temp_df.mindstate == ms],
        x = x,
        y = y, 
        hue = hue,
        hue_order = subtypes,
        palette = subtype_palette,
        legend=None,
        linewidth = 2,
        ax=ax
        )
    sns.despine()
    ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize = 14)
    ax.set_title(ms, font=bold_font, fontsize=14)
        
axs[0].set_ylabel('Power (dB)', font=bold_font, fontsize=14)
axs[0].set_xlim([0.5, 40])

fig.tight_layout()
    

# %% LME + PLOTS

def analyze_and_plot(mindstate, channels):
    """
    Performs cluster-based permutation testing and plots the results 
    for the specified mindstate and channels.

    Parameters
    ----------
    mindstate : str
        The mindstate to analyze (e.g., 'ON').
    channels : list of str
        List of channels to analyze (e.g., ['F3', 'C3', 'O1']).

    Returns
    -------
    Plots.

    """
    subtypes_here = ["C1", "HI"]
    palette_here = [palette[0], palette[1]]  # Ensure 'palette' is defined with at least two colors

    # Prepare the figure
    num_channels = len(channels)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=num_channels,
        figsize=(4 * num_channels, 5),
        sharey=True,
        constrained_layout=True
    )

    # Ensure axs is iterable
    if num_channels == 1:
        axs = [axs]

    for idx, channel in enumerate(channels):
        print(f"Processing channel: {channel}")

        # Filter the data for the specified stage and channel
        this_df = df.loc[
            (df.subtype.isin(subtypes_here)) &
            (df.channel == channel) &
            (df.mindstate == mindstate)
        ]

        # Define the model formula
        model_formula = 'perio_power ~ age + C(gender) + C(subtype, Treatment("C1"))'

        t_values = []
        p_values = []

        # Extract t-values and p-values for each frequency bin
        for freq_bin in freqs:
            temp_df = this_df.loc[this_df.freq_bin == freq_bin]
            model = smf.mixedlm(model_formula, temp_df, groups=temp_df['sub_id'], missing='drop')
            model_result = model.fit()

            coef_name = f'C(subtype, Treatment("{subtypes_here[0]}"))[T.{subtypes_here[1]}]'
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
        n_permutations = 100  # Increase this number for more accurate p-values
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
                color=palette_here[-1],
                linewidth=4,
                label='Significant Cluster'
            )

        ax.set_xlabel('Frequency (Hz)', font = bold_font, fontsize = 18)
        if idx == 0:
            ax.set_ylabel('Power', font = bold_font, fontsize = 18)
        # ax.set_title(f'Channel: {channel}', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim([min(freqs), max(freqs)])

        sns.despine(ax=ax)

    # Add legend and overall title
    # axs[-1].legend(fontsize=12)
    # plt.suptitle(f'Sleep mindstate: {mindstate}', fontsize=18)
    plt.savefig(os.path.join(fig_path, f"peri_lme_hi_cns_{mindstate}.png"), dpi = 300)
    plt.show()

# Example usage:
# Replace 'N2' with your desired sleep stage and ['F3', 'C3', 'O1'] with your desired channels
analyze_and_plot(mindstate='ON', channels=['F3', 'C3', 'O1'])

