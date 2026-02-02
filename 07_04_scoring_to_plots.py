#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:36:35 2025

@author: arthurlecoz
"""
# %% Paths & Packages

import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager 
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')
prop = font_manager.FontProperties(fname=font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()

if "arthur" in os.getcwd():
    path_root = "/Volumes/DDE_ALC/PhD/SLHIP"
else:
    path_root = "your_path"

path_figures = os.path.join(path_root, "07_Scoring", "Figures")
path_raw_experiment = os.path.join(path_root, "00_Raw", "experiment")

this_savepath = os.path.join(path_figures, "table_percentage_recording_time_tradi.csv")
df_30s_classic = pd.read_csv(this_savepath)        

this_savepath = os.path.join(path_figures, "table_percentage_mindstate_time_tradi.csv")
df_ms_classic = pd.read_csv(this_savepath)

# %% Proportion stage per recording

x = "tradi_scoring"
order = ['W', 'N1', 'N2', 'R']
y = "percentage"
hue = "recording_type"
hue_order = ["RS1", "RS2", "Probe_before", "Probe_during"]
box_width = .2
data = df_30s_classic[['sub_id', 'recording_type', 'tradi_scoring', 'percentage']
    ].groupby(['sub_id', 'recording_type', 'tradi_scoring'], as_index = False).mean()

palette = "Purples"

fig, ax1 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, 
    figsize = (8, 6), layout = "constrained")

sns.barplot(
    x = x, 
    order = order,
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = data, 
    palette = palette,
    fill = False,
    linewidth = 3,
    errorbar = "se",
    capsize = .2
    )
sns.stripplot(
    x = x, 
    order = order,
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = data, 
    ax = ax1,
    palette = palette,
    size = 4,
    alpha = .2,
    legend = None,
    dodge = True,
    jitter = .2
    )

plt.xlabel("Percentage of Sleep Stage", font = bold_font, fontsize = 14)
plt.xticks(ticks = [0, 1, 2, 3], labels = order, font = font, fontsize = 12)
plt.yticks(ticks = np.arange(0, 120, 20), labels = np.arange(0, 120, 20), font = font, fontsize = 12)
plt.ylabel("Sleep Stage", font = bold_font, fontsize = 14)
plt.ylim(0, 100)
sns.despine()
fig.tight_layout()

ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{path_figures}/percentage_score_tradi_recordtype_NT1.png", 
    dpi = 300
    )

# %% Proportion stage per mindstate

x = "tradi_scoring"
order = ['W', 'N1', 'N2', 'R']
y = "percentage"
hue = "mindstate"
hue_order = ['ON', 'MW', 'MB', 'HALLU', 'FORGOT']
box_width = .2
data = df_ms_classic[['sub_id', 'mindstate', 'tradi_scoring', 'percentage']
    ].groupby(['sub_id', 'mindstate', 'tradi_scoring'], as_index = False).mean()

ms_palette = ["#FFC000", "#00B050", "#0070C0", "#7030A0", "#000000"]


fig, ax1 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, 
    figsize = (8, 6), layout = "constrained")

sns.barplot(
    x = x, 
    order = order,
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = data, 
    palette = ms_palette,
    fill = False,
    linewidth = 3,
    errorbar = "se",
    capsize = .2
    )
sns.stripplot(
    x = x, 
    order = order,
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = data, 
    ax = ax1,
    palette = ms_palette,
    size = 4,
    alpha = .2,
    legend = None,
    dodge = True,   
    jitter = .2
    )

plt.xlabel("Percentage of Sleep Stage", font = bold_font, fontsize = 14)
plt.xticks(ticks = [0, 1, 2, 3], labels = order, font = font, fontsize = 12)
plt.yticks(ticks = np.arange(0, 120, 20), labels = np.arange(0, 120, 20), font = font, fontsize = 12)
plt.ylabel("Mindstate", font = bold_font, fontsize = 14)
plt.ylim(0, 100)
sns.despine()
fig.tight_layout()

ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{path_figures}/percentage_score_tradi_mindstate_NT1.png", 
    dpi = 300
    )

# %% daytime nt1

x = "tradi_scoring"
order = ['W', 'N1', 'N2', 'R']
y = "percentage"
hue = "session_time"
hue_order = ['AM', 'PM']
box_width = .2
data = df_30s_classic[['sub_id', 'session_time', 'tradi_scoring', 'percentage']
    ].groupby(['sub_id', 'session_time', 'tradi_scoring'], as_index = False).mean()

# palette = "Purples"

fig, ax1 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, 
    figsize = (4, 6), layout = "constrained")

sns.barplot(
    x = x, 
    order = order,
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = data, 
    # palette = palette,
    fill = False,
    linewidth = 3,
    errorbar = "se",
    capsize = .2
    )
sns.stripplot(
    x = x, 
    order = order,
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = data, 
    ax = ax1,
    # palette = palette,
    size = 4,
    alpha = .2,
    legend = None,
    dodge = True,
    jitter = .2
    )

plt.xlabel("Sleep Stage", font = bold_font, fontsize = 14)
plt.xticks(ticks = [0, 1, 2, 3], labels = order, font = font, fontsize = 12)
plt.yticks(ticks = np.arange(0, 120, 20), labels = np.arange(0, 120, 20), font = font, fontsize = 12)
plt.ylabel("Percentage", font = bold_font, fontsize = 14)
plt.ylim(0, 100)
sns.despine()
fig.tight_layout()

ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{path_figures}/percentage_score_tradi_daytime_NT1.png", 
    dpi = 300
    )

# %% daytime nt1

x = "tradi_scoring"
order = ['W', 'N1', 'N2', 'R']
y = "percentage"
hue = "session_time"
hue_order = ['AM', 'PM']
box_width = .2
data = df_30s_classic.loc[df_30s_classic.recording_type=="Probe_during"][['sub_id', 'session_time', 'tradi_scoring', 'percentage']
    ].groupby(['sub_id', 'session_time', 'tradi_scoring'], as_index = False).mean()

# palette = "Purples"

fig, ax1 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, 
    figsize = (4, 6), layout = "constrained")

sns.barplot(
    x = x, 
    order = order,
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = data, 
    # palette = palette,
    fill = False,
    linewidth = 3,
    errorbar = "se",
    capsize = .2
    )
sns.stripplot(
    x = x, 
    order = order,
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = data, 
    ax = ax1,
    # palette = palette,
    size = 4,
    alpha = .2,
    legend = None,
    dodge = True,
    jitter = .2
    )

plt.xlabel("Sleep Stage", font = bold_font, fontsize = 14)
plt.xticks(ticks = [0, 1, 2, 3], labels = order, font = font, fontsize = 12)
plt.yticks(ticks = np.arange(0, 120, 20), labels = np.arange(0, 120, 20), font = font, fontsize = 12)
plt.ylabel("Percentage", font = bold_font, fontsize = 14)
plt.ylim(0, 100)
sns.despine()
fig.tight_layout()

ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{path_figures}/percentage_score_tradi_daytime_duringprobes_NT1.png", 
    dpi = 300
    )
