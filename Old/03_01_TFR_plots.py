#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/07/2023

@author: arthur.lecoz

03_01_TFR_plots.py
"""

# %%% Paths & Packages
import mne
import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy.ndimage import gaussian_filter
from glob import glob
from scipy.stats import sem
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.multitest import fdrcorrection

#### Paths
if 'arthur' in os.getcwd():
    path_root='/Volumes/DDE_ALC/PhD/SLHIP'
else:
    path_root='your_path'

path_data=os.path.join(path_root, '00_Raw')
path_preproc=os.path.join(path_root, '01_Preproc')

if os.path.exists(os.path.join(path_data, "reports"))==False:
    os.makedirs(os.path.join(path_data, "reports"))
if os.path.exists(os.path.join(path_data, "intermediary"))==False:
    os.makedirs(os.path.join(path_data,"intermediary"))

redo = 0;

epochs_files  = glob(os.path.join(path_preproc, "epochs_probes" , "*epo.fif"))

subtypes = ['N1', 'HS']
mindstates = ['ON', 'MW', 'HALLU', 'MB', 'FORGOT', "MISS"]

# channels = ["Fz", "Cz", "Pz", "Oz"]
freqs = np.linspace(0.5, 40, 159)
channels = [
    'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
    'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
    'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
    'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1',
    'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6',
    'AF8', 'AF4', 'F2', 'Iz'
    ]

# %%% Script

big_dic = {subtype : {mindstate : [] for mindstate in mindstates}
           for subtype in subtypes}

for i_file, file in enumerate(epochs_files) :
    sub_id = file.split('probes/')[-1].split('_epo')[0]
    
    if 'HI' in sub_id : continue
    
    print(f"\n...Processing {sub_id} : {i_file+1} / {len(epochs_files)}...")

    epochs = mne.read_epochs(file, preload = True)
    subtype = epochs.metadata.subtype.unique()[0]
    metadata = epochs.metadata
    
    for ms in mindstates :
        if ms not in metadata.mindstate.unique():
            big_dic[subtype][ms].append(
                np.nan * np.empty((len(channels), len(freqs)))
                )
        else : 
            big_dic[subtype][ms].append(
                np.mean((
                    epochs[epochs.metadata.mindstate == ms].compute_psd(
                        method = "welch",
                        fmin = .5, 
                        fmax = 40,
                        n_fft = 1024,
                        n_overlap = 256,
                        n_per_seg = 512,
                        window = "hamming",
                        # picks = channels
                        )),
                    axis = 0))
            
# %% mean

dic_psd = {"N1" : {}, "HS" : {}}
dic_sem = {"N1" : {}, "HS" : {}}

for subtype in subtypes :
    print(f"\n...Computing values for : {subtype}...")
    for mindstate in mindstates :
        print(f"\n...Computing session {mindstate}")
        dic_psd[subtype][mindstate] = 10 * np.log10(np.nanmean(
                big_dic[subtype][mindstate], axis = 0))
        dic_sem[subtype][mindstate] = sem(10 * np.log10(
                big_dic[subtype][mindstate]), axis = 0, nan_policy = "omit")
        
# %% plot psd

chan_id = [1, 23, 52, 12, 16, 63]
ms_toplot = ['ON', 'MW', 'MB', 'HALLU', 'FORGOT']

# Loop through each channel
for i, i_channel in enumerate(chan_id):
    fig, axs = plt.subplots(
        nrows=1, 
        ncols=len(ms_toplot), 
        figsize=(16, 16), 
        sharey=True, 
        sharex=True,
        layout = "constrained"
        )
    for s, subtype in enumerate(subtypes) :
        # Loop through each population and plot its PSD and SEM
        for j, ms in enumerate(ms_toplot):
            ax = axs[j]
            # Convert power to dB
            psd_db = dic_psd[subtype][ms][i_channel]
    
            # Calculate the SEM
            sem_db = dic_sem[subtype][ms][i_channel]
    
            # Plot the PSD and SEM
            ax.plot(
                freqs, 
                psd_db, 
                label = subtype, 
                #color = psd_palette[j]
                )
            ax.fill_between(
                freqs, 
                psd_db - sem_db, 
                psd_db + sem_db, 
                # color = psd_palette[j], 
                alpha = .2
                )
            ax.axvline(x=10, color='r', linestyle='--', linewidth=1)

            # Set the title and labels
            fig.suptitle('Channel: ' + channels[i_channel])
            ax.set_title(ms)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_xlim([0.5, 40])
            ax.legend()

# Add the condition name as a title for the entire figure

# Add a y-axis label to the first subplot
axs[0].set_ylabel('Power (dB)')

# Adjust the layout of the subplots
# plt.constrained_layout()

# Show the plot
plt.show()

# %% Smoothed plot psd

chan_id = [1, 23, 52, 12, 16, 63]
ms_toplot = ['ON', 'MW', 'MB', 'HALLU', 'FORGOT']
smooth = 3

# Loop through each channel
for i, i_channel in enumerate(chan_id):
    fig, axs = plt.subplots(
        nrows=1, 
        ncols=len(ms_toplot), 
        figsize=(16, 16), 
        sharey=True, 
        sharex=True,
        layout = "constrained"
        )
    for s, subtype in enumerate(subtypes) :
        # Loop through each population and plot its PSD and SEM
        for j, ms in enumerate(ms_toplot):
            ax = axs[j]
            # Convert power to dB
            psd_db = gaussian_filter(
                dic_psd[subtype][ms][i_channel],
                smooth
                )
    
            # Calculate the SEM
            sem_db = gaussian_filter(
                dic_sem[subtype][ms][i_channel],
                smooth
                )
    
            # Plot the PSD and SEM
            ax.plot(
                freqs, 
                psd_db, 
                label = subtype, 
                #color = psd_palette[j]
                )
            ax.fill_between(
                freqs, 
                psd_db - sem_db, 
                psd_db + sem_db, 
                # color = psd_palette[j], 
                alpha = .2
                )
            ax.axvline(x=10, color='r', linestyle='--', linewidth=1)

            # Set the title and labels
            fig.suptitle('Channel: ' + channels[i_channel])
            ax.set_title(ms)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_xlim([0.5, 40])
            ax.legend()

# Add the condition name as a title for the entire figure

# Add a y-axis label to the first subplot
axs[0].set_ylabel('Power (dB)')

# Adjust the layout of the subplots
# plt.constrained_layout()

# Show the plot
plt.show()

# %% Compute ≠

# diff_psd = [(
#     10 * np.log10(np.mean(big_dic["HARD"][n_sess], axis = 0))
#     - 10 * np.log10(np.mean(big_dic["EASY"][n_sess], axis = 0))
#     ) for n_sess in sessions]
# diff_sem = [sem(
#     10 * np.log10(np.mean(big_dic["HARD"][n_sess], axis = 0))
#     - 10 * np.log10(np.mean(big_dic["EASY"][n_sess], axis = 0))
#     ) for n_sess in sessions]
    
# %% ≠ Easy - Hard

# diff_palette = ["#FFBA08", "#F48C06", "#DC2F02", "#9D0208", "#370617"]
# # sem_palette = ['#FFEBB9', "#FDD6A4", "#FEAE9A", "#FD9C9F", "#C71654"]

# fig, axs = plt.subplots(
#     nrows=1, ncols=4, figsize=(16, 12), sharey=True, layout = "constrained")
# for i, channel in enumerate(channels) :
#     ax = axs[i]
#     # Loop through Sessions & Difficulty
#     for i_sess, n_sess in enumerate(sessions):
#         # if n_sess == 2:
#         #     continue

#         # Plot the PSD and SEM
#         ax.plot(
#             freqs, gaussian_filter(diff_psd[i_sess][i], 3), 
#             label = n_sess, 
#             color = diff_palette[i_sess])
#         ax.fill_between(
#             freqs, 
#             gaussian_filter(diff_psd[i_sess][i] - diff_sem[i_sess][i], 3), 
#             gaussian_filter(diff_psd[i_sess][i] + diff_sem[i_sess][i], 3), 
#             color = diff_palette[i_sess],
#             alpha = .2
#             )

#         # Set the title and labels
#         ax.set_title('Session: ' + channel)
#         ax.set_xlabel('Frequency (Hz)')
#         ax.set_xlim([.5, 40])
#         # ax.set_ylim([-30, 60])
#         ax.legend()

#     # Add the condition name as a title for the entire figure
#     fig.suptitle('Differences HARD - EASY')

#     # Add a y-axis label to the first subplot
#     axs[0].set_ylabel('dB')

#     # Adjust the layout of the subplots
#     # plt.constrained_layout()

# # Show the plot
# plt.show(block = False)
# fig_savename = (f"/{tfrDataPath}/Figures/Diff_PSD_plot_2008.png")
# plt.savefig(fig_savename, dpi = 300)

# %% Compute bandpowers 

# freqs = np.linspace(0.5, 40, 161)
# bands = {
#     "delta" : (1,  4),
#     "theta" : (4 ,  8),
#     "alpha" : (8 , 12),
#     "beta"  : (12, 30)
#     }

# basic_params = ["difficulty", "session", "sub_id", "channel",
#                 "delta", "theta", "alpha", "beta"]

# big_dic = {param : [] for param in basic_params}

# thisSavingPath = os.path.join(
#     tfrDataPath, "Figures", "bandpower_dataframe_d_1_4.csv"
#     )

# for i_subid, sub_id in enumerate(np.unique(subject_ids)) :
#     print(f"\n...Processing {sub_id} : {i_subid+1} / {len(np.unique(subject_ids))}...")
#     if sub_id in bad_ids or sub_id == "36_rc" :
#         print(f"\n/!\...Skipping {sub_id} : among subject to exclude...")
#         continue
    
#     if sub_id in subjects_easy :
#         subject_level = 'EASY'
#     elif sub_id in subjects_hard :
#         subject_level = 'HARD'
    
#     for i_sess, n_sess in enumerate(sessions):
#         if len(glob(f"{cleanDataPath}/*{sub_id}_sess_{n_sess}*-epo.fif")) < 1 :
#             print(f"\n/!!!\ ERROR No ICA Cleaned file found for {sub_id} session {n_sess} - Skipping...\n")
#             continue
        
#         thisEpochFile = glob(
#             f"{cleanDataPath}/*{sub_id}_sess_{n_sess}*-epo.fif"
#             )[0]
#         epochs = mne.read_epochs(thisEpochFile, preload = True)
        
#         temp_power = np.mean(
#                 (epochs.compute_psd(
#                     method = "welch",
#                     fmin = .5, 
#                     fmax = 40,
#                     n_fft = 1024,
#                     n_overlap = 256,
#                     n_per_seg = 512,
#                     window = "hamming",
#                     n_jobs = 4)
#                 ),
#                 axis = 0)

#         for i_ch, chan in enumerate(epochs.ch_names) : 
#             thischan_power = temp_power[i_ch, :]
#             big_dic["difficulty"].append(subject_level)
#             big_dic["session"].append(n_sess)
#             big_dic["sub_id"].append(sub_id)
#             big_dic["channel"].append(chan)
#             for band, borders in bands.items() :
#                 bandpower = np.mean(thischan_power[
#                     np.logical_and(freqs >= borders[0], freqs <= borders[1])
#                     ], axis = 0)
                    
#                 big_dic[band].append(bandpower)
     
# df = pd.DataFrame.from_dict(big_dic)
# for col in df.columns[-4:] :
#     df[col] = 10 * np.log10(df[col])
# df.to_csv(thisSavingPath)

# %% Compute normalized BP

# thisSavingPath = os.path.join(
#     tfrDataPath, "Figures", "df_normedbandpower_dataframe_d_1_4.csv"
#     )

# bands = ['delta', 'theta', 'alpha', 'beta']

# normBigDic = {
#     "sub_id" : [],
#     "channel" : [],
#     "session" : [],
#     "difficulty" : [],
#     "norm_delta" : [],
#     "norm_theta" : [],
#     "norm_alpha" : [],
#     "norm_beta" : [],
#     }

# for i, sub_id in enumerate(df.sub_id.unique()) : 
#     if sub_id in ["2_pf"] : 
#         continue
#     print(f"...Processing {sub_id} : {i+1}/{len(df.sub_id.unique())}")
#     for n_ch in df.channel.unique() :
#         if not sum(
#                 df.loc[df['sub_id'] == sub_id].channel == n_ch):
#             continue
#         for n_sess in [2, 3, 4, 5, 1] :
#             normBigDic['sub_id'].append(sub_id)
#             normBigDic['channel'].append(n_ch)
#             normBigDic['session'].append(n_sess)
#             normBigDic['difficulty'].append(
#                 df.difficulty.loc[
#                     df.sub_id == sub_id
#                     ].unique()[0]
#                 )
#             for band in bands :
#                 normBigDic[f'norm_{band}'].append(
#                     df[band].loc[
#                         (df['sub_id'] == sub_id)
#                         & (df['channel'] == n_ch)
#                         & (df['session'] == n_sess)
#                         ].iloc[0] - df[band].loc[
#                             (df['sub_id'] == sub_id)
#                             & (df['channel'] == n_ch)
#                             & (df['session'] == 1)
#                                 ].iloc[0]
#                         ) 

# bandnorm_df = pd.DataFrame.from_dict(normBigDic)
# bandnorm_df.to_csv(thisSavingPath)

# %% Plot bandpower

# palette_hardeasy = ["#0a9396", "#8e7dbe"]

# fig, ax = plt.subplots(
#     nrows = 1, ncols = 4, figsize = (16, 6),
#     sharey = True
#     )

# for i_band, band in enumerate(bandnorm_df.columns[-4:]) :
#     sns.pointplot(
#         data = bandnorm_df,
#         x = "session",
#         y = band,
#         hue = "difficulty",
#         errorbar = "se",
#         ax = ax[i_band],
#         palette = palette_hardeasy
#         )
#     ax[i_band].set_title(band)
# fig.suptitle(
#     "BANDPOWER x SESSIONS", fontsize = "xx-large", fontweight = "bold"
#     )    
# fig.tight_layout(pad = 1)

# %% 
# %% LME | ApeFeat ~ Session + Difficulty

# list_values = []
# for i_sess, n_sess in enumerate([1, 2, 3, 4, 5]) :
#     for difficulty in ["EASY", "HARD"] :   
#         for channel in df.channel.unique() :
#             list_values.append(df.delta.loc[
#                 (df["session"] == n_sess)
#                 & (df["difficulty"] == difficulty)
#                 & (df["channel"] == channel)
#                 ].mean())
# vmin = min(list_values)
# vmax = max(list_values)

# fig, ax = plt.subplots(
#     nrows = 2, 
#     ncols = 5,
#     figsize = (18,7),
#     layout = 'tight'
#     )
# for i_sess, n_sess in enumerate([1, 2, 3, 4, 5]) :
#     list_easy = []
#     list_hard = []        
#     for channel in epochs.ch_names :
#         list_easy.append(df.delta.loc[
#             (df["difficulty"] == "EASY")
#             & (df["session"] == n_sess)
#             & (df["channel"] == channel)
#             ].mean())
#         list_hard.append(df.delta.loc[
#             (df["difficulty"] == "HARD")
#             & (df["session"] == n_sess)
#             & (df["channel"] == channel)
#             ].mean())
    
#     if n_sess == 5 :
#         divider = make_axes_locatable(ax[0][4])
#         cax = divider.append_axes("right", size = "5%", pad=0.05)
#     im, cm = mne.viz.plot_topomap(
#         list_easy,
#         epochs.info,
#         axes = ax[0][i_sess],
#         size = 2,
#         # names = mean_df.channel.unique(),
#         show = False,
#         contours = 4,
#         vlim = (vmin, vmax),
#         cmap = "viridis"
#         )
#     if n_sess == 5 :
#         fig.colorbar(im, cax = cax, orientation = 'vertical')
#     # ax[0][i_sess].set_title(f"EASY - S{n_sess}")
#     ax[0][i_sess].set_title("")
#     fig.suptitle("")
#     # fig.suptitle("SW Density according to the session and difficulty")
    
#     if n_sess == 5 :
#         divider = make_axes_locatable(ax[1][i_sess])
#         cax = divider.append_axes("right", size = "5%", pad=0.05)
#     im, cm = mne.viz.plot_topomap(
#         list_hard,
#         epochs.info,
#         axes = ax[1][i_sess],
#         size = 2,
#         # names = mean_df.channel.unique(),
#         show = False,
#         contours = 4,
#         vlim = (vmin, vmax),
#         cmap = "viridis"
#         )
#     # ax[1][i_sess].set_title(f"HARD - S{n_sess}")
#     ax[1][i_sess].set_title("")
#     if n_sess == 5 :
#         fig.colorbar(im, cax = cax, orientation = 'vertical')
#     plt.show(block = False)

#     # figsavename = f"{swDataPath}{os.sep}Figs{os.sep}S1_topoplot_density_freq_05_4.png"
#     # plt.savefig(figsavename, dpi = 300)
    
# %% lme - power sessions difficulty

# indexes = [
#         'session',
#         'C(difficulty)[T.HARD]', 
#        'session:C(difficulty)[T.HARD]'
#        ]

# bands = ["delta", "theta", "alpha", "beta"]

# fig, ax = plt.subplots(
#     nrows = 4, ncols = 3, figsize = (12, 8), layout = 'tight')

# # Add text to the left of each row
# row_labels = ["Delta", "Theta", "Alpha", "Beta"]
# for i_bp, band in enumerate(bands) :
#     ax[i_bp][0].text(-0.5, 0.5, row_labels[i_bp], 
#                      verticalalignment='center', 
#                      horizontalalignment='center',
#                      transform=ax[i_bp][0].transAxes,
#                      fontsize='medium', fontweight='bold')
    
# for i_bp, band in enumerate(bands) :
#     models = [
#         f"{band} ~ session + C(difficulty)",
#         f"{band} ~ session + C(difficulty)",
#         f"{band} ~ session * C(difficulty)"
#         ]
#     for i, model in enumerate(models):
#         temp_tval = []; temp_pval = []; chan_l = []
#         for chan in df.channel.unique():
#             subdf = df[
#                 ['sub_id', 'difficulty', 'session', 'channel', 
#                  band]
#                 ].loc[
#                 (df.channel == chan)
#                 ].dropna()
#             md = smf.mixedlm(model, subdf, groups = subdf['sub_id'])
#             mdf = md.fit()
#             temp_tval.append(mdf.tvalues[indexes[i]])
#             temp_pval.append(mdf.pvalues[indexes[i]])
#             chan_l.append(chan)
             
#         _, corrected_pval = fdrcorrection(temp_pval)
        
#         divider = make_axes_locatable(ax[i_bp][i])
#         cax = divider.append_axes("right", size = "5%", pad=0.05)
#         im, cm = mne.viz.plot_topomap(
#             data = temp_tval,
#             pos = epochs.info,
#             axes = ax[i_bp][i],
#             contours = 2,
#             mask = np.asarray(corrected_pval) <= 0.05,
#             mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
#                         linewidth=0, markersize=4),
#             cmap = "viridis",
#             # vlim = (-2.5, 2.5)
#             )
#         fig.colorbar(im, cax = cax, orientation = 'vertical')
#     ax[0][0].set_title("Session Effect", fontweight = "bold")
#     ax[0][1].set_title("Difficulty Effect", fontweight = "bold")
#     ax[0][2].set_title("Interaction Effect", fontweight = "bold")
        
#     fig.suptitle("T-values, p-val FDR corrected", 
#                  fontsize = "xx-large", 
#                  fontweight = "bold")
#     # plt.savefig(
#     #     os.path.join(
#     #         swDataPath, "Figs", "topo_LME_swfeat.png"
#     #         ),
#     #     dpi = 200
#     #     )
