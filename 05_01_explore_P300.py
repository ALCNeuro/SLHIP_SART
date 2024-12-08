#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:19:47 2024

@author: arthurlecoz

05_01_explore_P300

"""
# %% Paths & Packages

import os
import mne

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager

from tqdm import tqdm
 
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
    if 'HI_005' in fname : continue

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

# %% Stats Uncorrected

# Function to perform statistical analysis and plotting using LMMs
def analyze_and_plot_erp_uncorrected(channel, groups_to_compare):
    """
    Performs statistical analysis using linear mixed models at each time point and plots the ERP data.
    No correction for multiple comparisons is applied.

    Parameters
    ----------
    channel : str
        The channel to analyze (e.g., 'Pz').
    groups_to_compare : list of str
        List of two group names to compare (e.g., ['HS', 'HI']).

    Returns
    -------
    Plots the ERP comparison with significant time points highlighted (uncorrected).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf
    from scipy.stats import t
    import seaborn as sns
    from scipy.ndimage import label

    group1, group2 = groups_to_compare

    # Filter the DataFrame for the specified groups and channel
    df_filtered = df[
        df['group'].isin([group1, group2]) &
        (df['channel'] == channel)
    ].copy()

    # Optionally, recode group as categorical with group1 as the reference
    df_filtered['group'] = pd.Categorical(df_filtered['group'], categories=[group1, group2])

    # Get the list of unique time points
    time_points = df_filtered['time_point'].unique()
    time_points.sort()

    # Prepare arrays to store t-values and p-values
    t_values = []
    p_values = []

    # Define the model formula
    # Adjust the formula if you have additional covariates (e.g., age, gender)
    # For example: 'amplitude ~ C(group) + age + C(gender)'
    model_formula = 'amplitude ~ C(group)'

    # Fit LMM at each time point
    print("Fitting LMMs at each time point...")
    for time_point in tqdm(time_points):
        df_time = df_filtered[df_filtered['time_point'] == time_point]
        # Fit the LMM
        model = smf.mixedlm(model_formula, df_time, groups=df_time['sub_id'], missing='drop')
        try:
            model_result = model.fit(reml=False)
            # Extract t-value and p-value for the group effect
            coef_name = 'C(group)[T.{0}]'.format(group2)
            t_value = model_result.tvalues[coef_name]
            p_value = model_result.pvalues[coef_name]
        except Exception as e:
            print(f"Error at time {time_point}: {e}")
            t_value = np.nan
            p_value = np.nan
        t_values.append(t_value)
        p_values.append(p_value)

    t_values = np.array(t_values)
    p_values = np.array(p_values)

    # Identify significant time points (uncorrected)
    alpha = 0.05  # Significance level
    significant_time_points = p_values < alpha
    clusters, num_clusters = label(significant_time_points)

    # Prepare data for plotting
    times = time_points
    data_group1 = []
    data_group2 = []
    for sub_id in df_filtered['sub_id'].unique():
        group = df_filtered[df_filtered['sub_id'] == sub_id]['group'].iloc[0]
        df_sub = df_filtered[df_filtered['sub_id'] == sub_id]
        # Average over sessions
        df_sub_grouped = df_sub.groupby('time_point')['amplitude'].mean()
        # Ensure amplitudes are aligned with times
        df_sub_grouped = df_sub_grouped.reindex(times)
        amplitudes = df_sub_grouped.values
        if group == group1:
            data_group1.append(amplitudes)
        elif group == group2:
            data_group2.append(amplitudes)

    data_group1 = np.array(data_group1)
    data_group2 = np.array(data_group2)

    # Compute mean and SEM
    mean_group1 = np.nanmean(data_group1, axis=0)
    sem_group1 = np.nanstd(data_group1, axis=0, ddof=1) / np.sqrt(data_group1.shape[0])

    mean_group2 = np.nanmean(data_group2, axis=0)
    sem_group2 = np.nanstd(data_group2, axis=0, ddof=1) / np.sqrt(data_group2.shape[0])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = dict(
        HS = "#8d99ae", 
        HI = "#ffb703", 
        N1 = "#d00000"
        )

    ax.plot(times, mean_group1, label=group1, color=colors[group1], linewidth=2)
    ax.fill_between(times, mean_group1 - sem_group1, mean_group1 + sem_group1, color=colors[group1], alpha=0.3)

    ax.plot(times, mean_group2, label=group2, color=colors[group2], linewidth=2)
    ax.fill_between(times, mean_group2 - sem_group2, mean_group2 + sem_group2, color=colors[group2], alpha=0.3)

    # Highlight significant time points
    y_max = np.max([mean_group1 + sem_group1, mean_group2 + sem_group2])
    y_min = np.min([mean_group1 - sem_group1, mean_group2 - sem_group2])
    y_range = y_max - y_min
    significance_line_y = y_max + 0.05 * y_range

    # For each cluster of significant time points, plot a horizontal line
    for i in range(1, num_clusters + 1):
        cluster_indices = np.where(clusters == i)[0]
        cluster_times = times[cluster_indices]
        if len(cluster_times) > 0:
            ax.hlines(y=significance_line_y, xmin=cluster_times[0], xmax=cluster_times[-1],
                      color='k', linewidth=2)
            # Optionally, add a small vertical line at each end
            ax.vlines(x=cluster_times[0], ymin=significance_line_y - 0.01 * y_range,
                      ymax=significance_line_y + 0.01 * y_range, color='k', linewidth=2)
            ax.vlines(x=cluster_times[-1], ymin=significance_line_y - 0.01 * y_range,
                      ymax=significance_line_y + 0.01 * y_range, color='k', linewidth=2)

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Amplitude (V)', fontsize=14)
    ax.set_title(f'ERP Comparison at {channel} between {group1} and {group2}\n(Significant time points uncorrected)', fontsize=16)
    ax.legend()
    ax.axvline(0, color='black', linestyle='--')
    ax.axhline(0, color='black', linestyle='--')
    sns.despine()
    plt.show()
    # Optionally, save the figure
    # fig.savefig(f"erp_comparison_uncorrected_{group1}_vs_{group2}_{channel}.png", dpi=300)

    # Optionally, plot t-values over time
    fig_t, ax_t = plt.subplots(figsize=(10, 6))
    ax_t.plot(times, t_values, color='purple', label='t-values')
    # Determine critical t-value for the significance level
    df_resid = model_result.df_resid
    critical_t = t.ppf(1 - alpha / 2, df_resid)
    ax_t.axhline(critical_t, color='red', linestyle='--', label='Critical t-value')
    ax_t.axhline(-critical_t, color='red', linestyle='--')
    ax_t.set_xlabel('Time (s)', fontsize=14)
    ax_t.set_ylabel('t-value', fontsize=14)
    ax_t.set_title(f't-values over time for {group1} vs {group2}', fontsize=16)
    ax_t.legend()
    sns.despine()
    plt.show()
    # Optionally, save the figure
    # fig_t.savefig(f"tvalues_{group1}_vs_{group2}_{channel}.png", dpi=300)
    
# %% 
# Example usage:
analyze_and_plot_erp_uncorrected(channel='Pz', groups_to_compare=['HS', 'HI'])
analyze_and_plot_erp_uncorrected(channel='Pz', groups_to_compare=['HS', 'N1'])
analyze_and_plot_erp_uncorrected(channel='Pz', groups_to_compare=['HI', 'N1'])
