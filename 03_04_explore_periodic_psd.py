#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:33:14 2024

@author: arthurlecoz

03_04_explore_periodic_psd.py
"""
# %% Paths
import os
import pickle
import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import SLHIP_config_ALC as config
import seaborn as sns

from mne.stats import permutation_cluster_test
from glob import glob
from scipy.stats import sem
from scipy.stats import t
from scipy.ndimage import label
from highlight_text import ax_text, fig_text
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

subtypes = ["HS", "N1", "HI"]
channels = np.array(config.eeg_channels)
mindstates = ['ON', 'MW', 'HALLU', 'MB']
palette = ["#8d99ae", "#d00000", "#ffb703"]
palette = ["#8d99ae", "#ffb703"]
freqs = np.linspace(0.5, 40, 159)

# %% Loop

big_dic = {
    subtype : {
        ms : {
            channel : [] for channel in channels
            } for ms in mindstates
        } for subtype in subtypes
    }

for i, file in enumerate(periodic_files) :
    this_dic = pd.read_pickle(file)
    sub_id = file.split('periodic/')[-1].split('_per')[0]
    subtype = sub_id[:2]
    if len(subtypes)<3 :
        if subtype == "N1" : continue
    
    print(f"Processing : {sub_id}... [{i+1} / {len(periodic_files)}]")
    
    for mindstate in mindstates:
        for channel in channels:
            if len(this_dic[mindstate][channel]) < 1 :
                big_dic[subtype][mindstate][channel].append(
                    np.nan * np.empty(159))
            else : 
                big_dic[subtype][mindstate][channel].append(
                    this_dic[mindstate][channel][0])
    
# %% 

# big_dic = ''
                
# with open(big_dic_psd_savepath, 'wb') as handle:
#     pickle.dump(big_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# %% 
dic_psd = {subtype : {mindstate : {chan : [] for chan in channels}
                      for mindstate in mindstates} for subtype in subtypes}
dic_sem = {subtype : {mindstate : {chan : [] for chan in channels}
                      for mindstate in mindstates} for subtype in subtypes}

for subtype in subtypes :
    for mindstate in big_dic[subtype].keys() :
        for channel in big_dic[subtype][mindstate].keys() :
            dic_psd[subtype][mindstate][channel] = np.nanmean(big_dic[subtype][mindstate][channel], axis = 0)
            dic_sem[subtype][mindstate][channel] = sem(big_dic[subtype][mindstate][channel], nan_policy = 'omit')
                
# %%  

# big_av_psd_savepath = os.path.join(
#     fig_dir, "fooof_averaged_psd_flat_spectra_2.pickle"
#     )
# big_av_sem_savepath = os.path.join(
#     fig_dir, "fooof_sem_flat_spectra_2.pickle"
#     )
           
# with open(big_av_psd_savepath, 'wb') as handle:
#     pickle.dump(big_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(big_av_sem_savepath, 'wb') as handle:
#     pickle.dump(dic_sem, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% 

# palette = ["#5e6472", "#faa307"]

for mindstate in mindstates:
    # Create a new figure with three subplots
    fig, axs = plt.subplots(
        nrows=1, ncols=3, figsize=(10, 16), sharey=True, layout = "constrained")

    # Loop through each channel
    for i, channel in enumerate(channels):
        ax = axs[i]

        # Loop through each population and plot its PSD and SEM
        for j, subtype in enumerate(subtypes):
            # Convert power to dB
            psd_db = dic_psd[subtype][mindstate][channel]

            # Calculate the SEM
            sem_db = dic_sem[subtype][mindstate][channel]

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
    fig.suptitle('Condition: ' + mindstate)

    # Add a y-axis label to the first subplot
    axs[0].set_ylabel('Power (dB)')
    axs[0].get_legend().set_visible(False)
    axs[1].get_legend().set_visible(False)

    # Adjust the layout of the subplots
    # plt.constrained_layout()

    # Show the plot
    plt.show()
    fig_savename = (f"{fig_path}/vhi_cns_flatPSD_plot_{mindstate}.png")
    plt.savefig(fig_savename, dpi = 300)

# %% dics to DF

coi = ["sub_id", "subtype", 
       # "age", "gender", 
       "mindstate", "channel", "freq_bin", "perio_power"]

big_dic = {c : [] for c in coi}

for i, file in enumerate(periodic_files) :
    this_dic = pd.read_pickle(file)
    sub_id = file.split('fooof/')[-1].split('_per')[0]
    subtype = sub_id[:2]
    if len(subtypes)<3 :
        if subtype == "N1" : continue
    
    print(f"Processing : {sub_id}... [{i+1} / {len(periodic_files)}]")
    
    # this_demo = df_demographics.loc[df_demographics.code == sub_id]
    # this_age = this_demo.age.iloc[0]
    # this_gender = this_demo.sexe.iloc[0]
    
    for mindstate in mindstates:
        for channel in channels:
            for i_f, freq_bin in enumerate(freqs) :
                if len(this_dic[mindstate][channel]) < 1 :
                    big_dic["perio_power"].append(np.nan)
                else : 
                    big_dic["perio_power"].append(this_dic[mindstate][channel][0][i_f])
                big_dic["sub_id"].append(sub_id)
                big_dic["subtype"].append(subtype)
                # big_dic["age"].append(this_age)
                # big_dic["gender"].append(this_gender)
                big_dic["mindstate"].append(mindstate)
                big_dic["channel"].append(channel)
                big_dic["freq_bin"].append(freq_bin)

df = pd.DataFrame.from_dict(big_dic) 

# %% LME + PLOTS

def analyze_and_plot(stage, channels):
    """
    Performs cluster-based permutation testing and plots the results for the specified sleep stage and channels.

    Parameters
    ----------
    stage : str
        The sleep stage to analyze (e.g., 'N2').
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
            (df.stage == stage)
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
            psd_db = dic_psd[subtype][stage][channel]
            sem_db = dic_sem[subtype][stage][channel]

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
    # plt.suptitle(f'Sleep Stage: {stage}', fontsize=18)
    plt.savefig(os.path.join(fig_dir, f"peri_lme_hi_cns_{stage}.png"), dpi = 300)
    plt.show()

# Example usage:
# Replace 'N2' with your desired sleep stage and ['F3', 'C3', 'O1'] with your desired channels
analyze_and_plot(stage='N2', channels=['F3', 'C3', 'O1'])

# %% 

chan_names = ["F3", "C3", "O1"]
subtypes_here = ["C1", "HI"]

alpha_cluster_forming = 0.05
n_conditions = 2
n_observations = 80
dfn = n_conditions - 1
dfd = n_observations - n_conditions

for i_st, stage in enumerate(stages) : 
    
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(12, 12), sharey=True, layout = "constrained")
    for i_ch, channel in enumerate(chan_names) :
        
        hsi_power = np.dstack(
            [i for i in big_dic[subtypes_here[0]][stage][channel]]
            ).transpose((2, 1, 0))
        ctl_power = np.dstack(
            [i for i in big_dic[subtypes_here[1]][stage][channel]]
            ).transpose((2, 1, 0))

        alpha_cluster_forming = 0.05
        n_conditions = 2
        n_observations = 159
        dfn = n_conditions - 1
        dfd = n_observations - n_conditions

        # Note: we calculate 1 - alpha_cluster_forming to get the critical value
        # on the right tail
        # f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

        F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            [ctl_power, hsi_power],
            out_type="mask",
            n_permutations=1000,
            # threshold=f_thresh,
            # tail=0,
            # adjacency = adjacency
            )
        
        # Loop through each population and plot its PSD and SEM
        for j, subtype in enumerate(subtypes_here):
            # Convert power to dB
            psd_db = dic_psd[subtype][stage][channel]
        
            # Calculate the SEM
            sem_db = dic_sem[subtype][stage][channel]
        
            # Plot the PSD and SEM
            ax[i_ch].plot(
                freqs, 
                psd_db, 
                label = subtype, 
                color = palette[j]
                )
            ax[i_ch].fill_between(
                freqs, 
                psd_db - sem_db, 
                psd_db + sem_db, 
                alpha= 0.3, 
                color = palette[j]
                )
        
        for i_c, c in enumerate(clusters):
            # c = c[:, i_ch]
            c = np.squeeze(c)
            if np.any(c) :
                if cluster_p_values[i_c] <= 0.05:
                    h = ax[i_ch].axvspan(freqs[c].min(), freqs[c].max(), color="r", alpha=0.1)
        
        # Set the title and labels
        ax[i_ch].set_title('Channel: ' + chan_names[i_ch])
        ax[i_ch].set_xlabel('Frequency (Hz)')
        ax[i_ch].set_xlim([0.5, 40])
        # ax.set_ylim([-30, 60])
        ax[i_ch].legend()
        
        # Add the condition name as a title for the entire figure
        fig.suptitle('Condition: ' + stage)
    axs[0].set_ylabel('Power', font = bold_font, fontsize = 18)
    
    plt.show(block = False)
    # fig_savename = (fig_dir + "/flatPSD_plot_clusterperm" 
    #                 + stage + ".png")
    # plt.savefig(fig_savename, dpi = 300)



