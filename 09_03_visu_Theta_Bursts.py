#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:43:00 2025

@author: arthurlecoz

09_03_visu_Theta_Bursts.py
"""
# %% Paths & Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import os

from matplotlib.font_manager import FontProperties
from matplotlib import font_manager 

# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')
prop = font_manager.FontProperties(fname=font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()


# %% Plot

file_path = "/Volumes/DDE_ALC/PhD/SLHIP/01_Preproc/raw_icaed/N1_017_AM_raw.fif"
burst_data_file = "/Volumes/DDE_ALC/PhD/SLHIP/09_Bursts/bursts_detected/Bursts_N1_017_AM_raw.csv"  
# channels = ['AFz', 'Fz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
channels = ['POz']
t_beg = 3050 * 256
t_end = 3070 * 256
n_time = []
offset = 40

raw = mne.io.read_raw(file_path, preload = True)
# Load the EEG epochs and select the channels of interest
raw.pick_channels(channels)

# Make a copy for filtering and resampling. The filtering steps here mimic those in your original snippet.
raw_filt = raw.copy().filter(1, 30)

# Load wave detection results for all waves and slow waves
df = pd.read_csv(burst_data_file)
df["actual_band"] = ["theta" if f < 8 else "alpha" for f in df.BurstFrequency.values]
df_theta = df.loc[
    (df.actual_band=="theta")
    & (df.ChannelIndexLabel.isin(channels))
    ]
df_theta['start_sec'] = df_theta.Start.values / 256
df_theta['closeness'] = np.concatenate((np.nan*np.empty((1)), np.diff(df_theta.start_sec)))

# Extract data for the specified epoch;
# data shape will be (n_channels, n_times)
data = raw_filt.get_data(units = 'uV')
this_data = data[:, t_beg:t_end]
n_times = this_data.shape[1]
time = np.arange(n_times) / 256  # Convert sample indices to seconds (256 Hz)

# Create the figure and a single axis for all channels
fig, ax = plt.subplots(figsize=(10, 1.4))
# Loop over channels and plot each with a vertical offset
for idx, channel in enumerate(channels):
    # Get the filtered signal for this channel and apply vertical offset
    signal = this_data[idx]
    shifted_signal = signal + idx * offset  # Vertical shift for clarity
    # Plot the filtered EEG data (light grey)
    ax.plot(time, shifted_signal, color='k', linewidth = .5, alpha = .8)
    # Get the burst detections for this channel
    this_df = df_theta[
        (df_theta['ChannelIndexLabel'] == channel) 
        & (df_theta["Start"] >= t_beg)
        & (df_theta["End"] <= t_end)
        ]
    # Overplot selected burst in red
    for i, wave in this_df.iterrows():
        start_idx = int(wave['Start'] - t_beg)
        end_idx = int(wave['End'] - t_beg)
        
        if start_idx < n_times and end_idx <= n_times:
            ax.plot(
                time[start_idx:end_idx], 
                shifted_signal[start_idx:end_idx],
                color='#7030A0',
                linewidth = 1
                )
    # Add a text label at the left edge for each channel
    ax.text(
        time[0] - 0.08 * (time[-1]-time[0]), 
        shifted_signal[0], 
        channel,
        verticalalignment='center', 
        font = bold_font,
        fontsize=8, 
        fontweight='bold'
        )
# Customize plot aesthetics
ax.set_xlabel(
    'Time (s)', 
    font = bold_font,
    fontsize=8
    )
ax.set_xticks(
    np.arange(0, 35, 5), 
    np.arange(0, 35, 5), 
    font = bold_font, 
    fontsize = 8
    )
ax.set_yticks(ticks = [])
ax.set_ylabel('', fontsize=8)
ax.set_ylim(-50, 50)
ax.set_xlim(0, 20)

# Optionally adjust y-axis limits to nicely frame all channels.
ax.set_ylim(-offset, offset * (len(channels)))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
fig.tight_layout()
plt.show()
this_savepath = os.path.join(
    "/Volumes/DDE_ALC/PhD/SLHIP/09_Bursts/figs/NT1_CTL", 
    "N1_017_exemple_burst.png"
    )
plt.savefig(this_savepath, dpi = 300)


