#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:19:47 2024

@author: arthurlecoz

05_01_explore_P300

"""
# %% Paths & Packages

import os
import numpy as np
import pandas as pd
import mne
from glob import glob

from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
 
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')
prop = font_manager.FontProperties(fname=font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()

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

evokeds_files = glob(os.path.join(path_data,"intermediary" ,'ERP_*.fif'))

palette = ["#8d99ae", "#ffb703", "#d00000"]
palette_dic = dict(
    HS = "#8d99ae", 
    HI = "#ffb703", 
    N1 = "#d00000"
    )

# %% Loading & Organizing

# Initialize a nested dictionary to store evoked data
group_cond_evokeds = {'HI': {'100': [], '101': []},
                      'HS': {'100': [], '101': []},
                      'N1': {'100': [], '101': []}}

for fname in evokeds_files:
    if 'HI_005' in fname : continue
    # Extract group from the filename
    basename = os.path.basename(fname)
    parts = basename.split('_')
    group = parts[1]  # 'HI', 'HS', or 'N1'

    # Read the evoked data from the file
    evokeds = mne.read_evokeds(fname, baseline=(None, 0), verbose='error')

    # Organize the evoked data by condition
    for evoked in evokeds:
        condition = evoked.comment  # '100' or '101'
        group_cond_evokeds[group][condition].append(evoked)

# %% Grand averages

grand_averages = {}

for group in group_cond_evokeds:
    grand_averages[group] = {}
    for condition in group_cond_evokeds[group]:
        evokeds = group_cond_evokeds[group][condition]
        if evokeds:  # Ensure there is data
            grand_avg = mne.grand_average(evokeds)
            grand_averages[group][condition] = grand_avg
            
# Plot NoGo condition ('100') across groups
mne.viz.plot_compare_evokeds(
    {group: grand_averages[group]['100'] for group in grand_averages},
    picks='Pz', title='NoGo Condition (100)', ci=True)

# Plot Go condition ('101') across groups
mne.viz.plot_compare_evokeds(
    {group: grand_averages[group]['101'] for group in grand_averages},
    picks='Pz', title='Go Condition (101)', ci=True)

# %% Visualization

# Plot NoGo condition ('100') across groups
mne.viz.plot_compare_evokeds(
    {group: grand_averages[group]['100'] for group in grand_averages},
    picks='Pz', title='NoGo Condition (100)', ci=True)

# Plot Go condition ('101') across groups
mne.viz.plot_compare_evokeds(
    {group: grand_averages[group]['101'] for group in grand_averages},
    picks='Pz', title='Go Condition (101)', ci=True)

# %% P300 group

difference_evokeds = {}

for group in grand_averages:
    evoked_nogo = grand_averages[group]['100']
    evoked_go = grand_averages[group]['101']
    # Compute the difference wave
    difference = mne.combine_evoked([evoked_nogo, evoked_go], weights=[1, -1])
    difference_evokeds[group] = difference

# Plot the difference waves across groups
mne.viz.plot_compare_evokeds(
    difference_evokeds, 
    picks='Pz', 
    title='Difference Waves (NoGo - Go)', 
    ci=True,
    colors = palette
    )

# %% Topographies
import matplotlib.pyplot as plt

times = [0.3, 0.4, 0.5]  # Times in seconds
for group in difference_evokeds:
    evoked = difference_evokeds[group]
    evoked.plot_topomap(times=times, vlim=(-6, 6))
    plt.suptitle(group, font = bold_font, fontsize = 16, color = palette_dic[group])
    
# %% With SEM

# Initialize a nested dictionary to store evoked data
group_cond_evokeds = {'HI': {'100': [], '101': []},
                      'HS': {'100': [], '101': []},
                      'N1': {'100': [], '101': []}}

for fname in evokeds_files:
    if 'HI_005' in fname : continue
    # Extract group from the filename
    basename = os.path.basename(fname)
    parts = basename.split('_')
    group = parts[1]  # 'HI', 'HS', or 'N1'

    # Read the evoked data from the file
    evokeds = mne.read_evokeds(fname, baseline=(None, 0), verbose='error')

    # Organize the evoked data by condition
    for evoked in evokeds:
        condition = evoked.comment  # '100' or '101'
        group_cond_evokeds[group][condition].append(evoked)

# Compute the difference waves for each subject
difference_evokeds_subjects = {}

for group in group_cond_evokeds:
    difference_evokeds_subjects[group] = []
    evokeds_nogo = group_cond_evokeds[group]['100']
    evokeds_go = group_cond_evokeds[group]['101']
    
    # if len(evokeds_nogo) != len(evokeds_go):
    #     print(f"Warning: Unequal number of NoGo and Go evokeds in group {group}")
    #     continue
    
    for evoked_nogo, evoked_go in zip(evokeds_nogo, evokeds_go):
        difference = mne.combine_evoked([evoked_nogo, evoked_go], weights=[1, -1])
        difference_evokeds_subjects[group].append(difference)

# Choose the channel of interest
channel = 'Pz'

# Initialize dictionaries to store mean and SEM for each group
mean_data_group = {}
sem_data_group = {}

# Time vector (assuming all Evoked objects have the same time points)
times = difference_evokeds_subjects[next(iter(difference_evokeds_subjects))][0].times

for group, evoked_list in difference_evokeds_subjects.items():
    # List to store data from each subject
    data_list = []
    for evoked in evoked_list:
        # Get index of the channel
        try:
            ch_idx = evoked.ch_names.index(channel)
        except ValueError:
            print(f"Channel {channel} not found in Evoked data.")
            continue
        data = evoked.data[ch_idx, :]  # Shape: (n_times,)
        data_list.append(data)
    # Stack data into array: (n_subjects, n_times)
    data_array = np.array(data_list)
    # Compute mean and SEM across subjects
    mean_data = np.mean(data_array, axis=0)
    sem_data = np.std(data_array, axis=0, ddof=1) / np.sqrt(data_array.shape[0])
    # Store in dictionaries
    mean_data_group[group] = mean_data
    sem_data_group[group] = sem_data        

# %% Plot

import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))

for group in ['HS', 'HI', 'N1']:
    mean_data = mean_data_group[group]
    sem_data = sem_data_group[group]
    ax.plot(
        times, 
        mean_data, 
        label=group, 
        color=palette_dic[group],
        linewidth = 2,
        alpha = .9
        )
    ax.fill_between(
        times, 
        mean_data - sem_data, 
        mean_data + sem_data, 
        color=palette_dic[group], 
        alpha=0.2
        )

ax.set_xlabel('Time (s)', font = font, fontsize = 14)
ax.set_ylabel('Amplitude (V)', font = font, fontsize = 14)
ax.set_title(
    f'Difference Waves (NoGo - Go) at {channel} with SEM', 
    font = bold_font, 
    fontsize = 18
    )
ax.legend(title='Groups')
ax.axvline(0, color='black', linestyle='--', alpha = .5)
ax.axhline(0, color='black', linestyle='--', alpha = .5)

ax.set_xlim(times[0], times[-1])
ax.set_ylim(-1.5e-6, 6e-6)
sns.despine()

plt.show()

# %% Statistical comparison

channel = 'Pz'
subject_evokeds = {}

# Process each evoked file
for fname in evokeds_files:
    basename = os.path.basename(fname)
    parts = basename.split('_')
    group = parts[1]  # 'HI', 'HS', or 'N1'
    sub_num = parts[2]  # '001', '002', etc.
    session = parts[3]  # 'AM', 'PM'

    sub_id = f"{group}_{sub_num}"

    evokeds = mne.read_evokeds(fname, verbose='error')

    # Initialize nested dictionaries if necessary
    if sub_id not in subject_evokeds:
        subject_evokeds[sub_id] = {}
    if session not in subject_evokeds[sub_id]:
        subject_evokeds[sub_id][session] = {}

    # Organize the evoked data by condition
    for evoked in evokeds:
        condition = evoked.comment  # '100' or '101'
        subject_evokeds[sub_id][session][condition] = evoked

coi = ['sub_id', 'group', 'session', 'time_point', 'amplitude', 'channel']
big_dic = {col: [] for col in coi}

for sub_id in subject_evokeds:
    group = sub_id.split('_')[0]  # 'HI', 'HS', or 'N1'
    for session in subject_evokeds[sub_id]:
        # Get evoked data for '100' and '101'
        evoked_nogo = subject_evokeds[sub_id][session].get('100', None)
        evoked_go = subject_evokeds[sub_id][session].get('101', None)
        if evoked_nogo is None or evoked_go is None:
            continue  # Skip if data missing
        # Compute the difference wave
        difference = mne.combine_evoked([evoked_nogo, evoked_go], weights=[1, -1])
        # Get the data for the channel of interest
        if channel not in difference.ch_names:
            print(f"Channel {channel} not found for subject {sub_id}, session {session}")
            continue
        ch_idx = difference.ch_names.index(channel)
        data = difference.data[ch_idx, :]  # Shape: (n_times,)
        times = difference.times  # Shape: (n_times,)
        # Add data to the dictionary
        for i_time, time_point in enumerate(times):
            amplitude = data[i_time]
            big_dic['sub_id'].append(sub_id)
            big_dic['group'].append(group)
            big_dic['session'].append(session)
            big_dic['time_point'].append(time_point)
            big_dic['amplitude'].append(amplitude)
            big_dic['channel'].append(channel)

# Create the DataFrame
df = pd.DataFrame.from_dict(big_dic)

