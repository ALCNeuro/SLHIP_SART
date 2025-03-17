#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 18:27:26 2025

@author: arthurlecoz

06_03_usleep_explore.py
"""
# %% Paths & Packages

import mne 
import os
from glob import glob
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import statsmodels.formula.api as smf

# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

#### Paths
if 'arthur' in os.getcwd():
    path_root='/Volumes/DDE_ALC/PhD/SLHIP'
else:
    path_root='your_path'

path_data=os.path.join(path_root, '00_Raw')
path_usleep=os.path.join(path_root, '06_USleep')

# Paths to the EEG files, here brainvision files
files = glob(os.path.join(path_usleep, '*.npy'))

probe_col = [
    "nprobe","t_probe_th","t_probe_act","nblock","block_cond","ntrial",
    "PQ1_respkey","PQ2_respkey","PQ3_respkey",
    "PQ1_resptime","PQ2_resptime","PQ3_resptime",
    "PQ1_questime","PQ2_questime","PQ3_questime",
    "PQ1_respval","PQ2_respval","PQ3_respval"
    ]
ms_dic = {
    0 : "MISS",
    1 : 'ON',
    2 : 'MW',
    3 : 'DISTRACTED',
    4 : 'HALLU',
    5 : 'MB',
    6 : 'FORGOT'
    }

cols_oi = ["sub_id", "group", "session", "nprobe", 
           "mindstate", "voluntary", "sleepiness",
           "P_WAKE", "P_N1", "P_N2", "P_N3", "P_REM"]

time_window_oi = 5
subtype_palette = ["#8d99ae", "#d00000", "#ffb703"]

# %% Script

big_dict = {col : [] for col in cols_oi}

file = files[0]

for i, file in enumerate(files) :
    sub_id = file.split('USleep/')[-1].split('_hyp')[0]
    session = sub_id[-2:]
    if (sub_id == "N1_001_PM" or 'HS_008' in sub_id or "HS_007" in sub_id) : continue
    
    #### Extract EEG Infos
    raw_path = glob(os.path.join(path_data, 'experiment', f'sub_{sub_id[:-3]}', f'*SART*{session}*.vhdr'))[0]
    raw = mne.io.read_raw_brainvision(raw_path, preload=True)
    sf = raw.info['sfreq']
    events, event_id = mne.events_from_annotations(raw)
    ms_probes =  np.stack(
        [event for i, event in enumerate(events[events[:, 2] == 128]) 
         if not i%3])
    sec_ms_probes = np.round(ms_probes[:,0]/sf).astype(int)
    behav_paths = glob(os.path.join(
        path_data, "experiment", f"sub_{sub_id[:-3]}", "*.mat"
        ))
    
    #### Extract Behav Infos
    if len(behav_paths) < 1 :
        print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
        continue
    if session == "AM" :
        behav_path = behav_paths[0]
    else :
        behav_path = behav_paths[1]
    mat = loadmat(behav_path)
    df_probe = pd.DataFrame(
        mat['probe_res'], 
        columns = probe_col)
    if any(df_probe.PQ1_respval.isna()) :
        df_probe.PQ1_respval.replace(np.nan, 0, inplace = True)
        
    ms_answers = np.array(
        [ms_dic[value] for value in df_probe.PQ1_respval.values]
        )
    vol_answers = df_probe.PQ2_respval.values
    sleepi_answers = df_probe.PQ3_respval.values
    
    print(f"""\nIn {sub_id} file, were found :
        * {ms_probes.shape[0]} Probes (first question)
        -> {ms_answers.shape[0]} MS Answers
        -> {vol_answers.shape[0]} Voluntary Answers
        -> {sleepi_answers.shape[0]} Sleepiness answers""")
    
    if not len(ms_probes) == len(ms_answers) : 
        print(f"!!!\n{sub_id} : Careful, inconsistencies found between EEG and Behav\n!!!")
        continue
    
    #### Extract HD Infos
    av_confidence_scores = np.load(file, allow_pickle=True)
    # Divide score by the total number of chans (because it sums acros chans) and focus on the last 10s
    confidence_scores = (av_confidence_scores/22)
    # Take only the scores before the probes, remove the last probe :
    # U sleep might use a long time window and derive HD at the second scale.
    # probe_conf_scores = confidence_scores[oi_sec_ms_probes[:-time_window_oi]]
    
    for probe in range(ms_answers.shape[0]-1):
        this_conf_scores = np.mean(
            confidence_scores[
                sec_ms_probes[probe]-time_window_oi:sec_ms_probes[probe]
                ], axis=0
            )
        big_dict["sub_id"].append(sub_id)
        big_dict["group"].append(sub_id[:2])
        big_dict["session"].append(session)
        big_dict["nprobe"].append(probe)
        big_dict["mindstate"].append(ms_answers[probe])
        big_dict["voluntary"].append(vol_answers[probe])
        big_dict["sleepiness"].append(sleepi_answers[probe])
        big_dict["P_WAKE"].append(this_conf_scores[0])
        big_dict["P_N1"].append(this_conf_scores[1])
        big_dict["P_N2"].append(this_conf_scores[2])
        big_dict["P_N3"].append(this_conf_scores[3])
        big_dict["P_REM"].append(this_conf_scores[4])
     
df = pd.DataFrame.from_dict(big_dict)
df.to_csv(os.path.join(path_usleep, f'df_proba_probes_{time_window_oi}.csv'))

# %% 

df_oi = df.loc[df.mindstate.isin(["ON", "MW", "MB", "HALLU", "FORGOT"])]

mean_df = df_oi[
    ["sub_id", "group", "nprobe", "mindstate", "sleepiness",
    "P_WAKE", "P_N1", "P_N2", "P_N3", "P_REM"]
    ].groupby(["sub_id", "group", "mindstate"], 
              as_index=False
              ).mean()

# %% Plots ME Subtype

data = mean_df.copy()
y = 'P_REM'
x = "group"
order = ["HS", "N1", "HI"]   
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    errorbar = 'se',
    capsize = 0.05,
    linestyle = 'none',
    palette = subtype_palette
    )                
sns.stripplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    alpha = 0.15,
    dodge = True,
    legend = None,
    palette = subtype_palette
    )

ax.set_ylabel('Probability %', size = 18, font = bold_font)
ax.set_xlabel('Group', size = 18, font = bold_font)
ax.set_ylim(0, 1)
ax.set_xticks(
    ticks = np.arange(0, 2, 1), 
    labels = ["HS", "N1", "HI"]   ,
    font = font, 
    fontsize = 10)
ax.set_yticks(
    ticks = np.arange(0, 1.2, .2), 
    labels = np.arange(0, 120, 20), 
    font = font, fontsize = 12)
sns.despine()
# title = """<Mindstate Percentage> according to the <Mindstate>"""
# fig_text(
#    0.05, .93,
#    title,
#    fontsize=16,
#    ha='left', va='center',
#    color="k", font=font,
#    highlight_textprops=[
#       {'font': bold_font},
#       {'font': bold_font},
#    ],
#    fig=fig
# )

# plt.savefig(
#     os.path.join(path_usleep, "figs", "proba_probe_me_subtype.png"), 
#     dpi=200
#     )

# %% Stats ME Subtype

# model_formula = 'percentage ~ C(mindstate, Treatment("ON"))'
# model = smf.mixedlm(
#     model_formula, 
#     df_mindstate, 
#     groups=df_mindstate['sub_id'], 
#     missing = 'drop'
#     )
# model_result = model.fit()
# print(model_result.summary())

# %% Plots Subtype x Mindstate

# poi = ["P_WAKE", "P_N1", "P_N2", "P_N3", "P_REM"]
poi = ["P_WAKE", "P_REM"]
data = mean_df.copy()

hue = "group"
hue_order = ["HS", "N1", "HI"]   
x = "mindstate"
order = ["ON", "MW", "MB", "HALLU"]   

fig, axs = plt.subplots(
    nrows = len(poi),
    ncols = 1,
    figsize = (6,12),
    sharex=True,
    sharey=True
    )
for i_p, p in enumerate(poi):

    y = p
    
    sns.pointplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        hue = hue,
        hue_order = hue_order,
        errorbar = 'se',
        capsize = 0.05,
        linestyle = 'none',
        palette = subtype_palette,
        dodge=.55,
        ax = axs[i_p],
        legend= None
        )                
    sns.stripplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        hue = hue,
        hue_order = hue_order,
        alpha = 0.2,
        dodge = True,
        legend = None,
        palette = subtype_palette,
        ax = axs[i_p],
        )
    
    axs[i_p].set_ylabel(p, size = 18, font = bold_font)
    axs[i_p].set_xlabel('Group', size = 18, font = bold_font)
    axs[i_p].set_ylim(0, 1)
    # ax.set_xticks(
    #     ticks = np.arange(0, 2, 1), 
    #     labels = ["HS", "N1", "HI"]   ,
    #     font = font, 
    #     fontsize = 10)
    
    axs[i_p].set_yticks(
        ticks = np.arange(0, 1.2, .2), 
        labels = np.arange(0, 120, 20), 
        font = font, fontsize = 12)
    sns.despine()

fig.tight_layout(pad=1.5)

# plt.savefig(
#     os.path.join(path_usleep, "figs", "proba_probe_me_subtype.png"), 
#     dpi=200
#     )


# %% Stats Subtype x Mindstate

model_formula = 'P_WAKE ~ C(group, Treatment("HS")) * C(mindstate, Treatment("ON"))'
model = smf.mixedlm(
    model_formula, 
    mean_df, 
    groups=mean_df['sub_id'], 
    missing = 'drop'
    )
model_result = model.fit()
print(model_result.summary())

# %% Plots Subtype x Mindstate

this_df = df_oi[
    ["sub_id", "group", "sleepiness",
    "P_WAKE", "P_N1", "P_N2", "P_N3", "P_REM"]
    ].groupby(["sub_id", "group", "sleepiness"], 
              as_index=False
              ).mean()
              
this_df["sleepiness"] = this_df["sleepiness"].astype(int)
# poi = ["P_WAKE", "P_N1", "P_N2", "P_N3", "P_REM"]
poi = ["P_WAKE", "P_REM"]
data = this_df.copy()

hue = "group"
hue_order = ["HS", "N1", "HI"]   
x = "sleepiness"
order = np.arange(1, 10, 1)

fig, axs = plt.subplots(
    nrows = len(poi),
    ncols = 1,
    figsize = (6,12),
    sharex=True,
    sharey=True
    )
for i_p, p in enumerate(poi):

    y = p
    
    sns.pointplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        hue = hue,
        hue_order = hue_order,
        errorbar = 'se',
        capsize = 0.05,
        linestyle = 'none',
        palette = subtype_palette,
        dodge=.55,
        ax = axs[i_p],
        legend= None
        )                
    sns.stripplot(
        data = data, 
        x = x,
        y = y,
        order = order,
        hue = hue,
        hue_order = hue_order,
        alpha = 0.2,
        dodge = True,
        legend = None,
        palette = subtype_palette,
        ax = axs[i_p],
        )
    
    axs[i_p].set_ylabel(p, size = 18, font = bold_font)
    axs[i_p].set_xlabel('Sleepiness (KSS)', size = 18, font = bold_font)
    axs[i_p].set_ylim(0, 1)
    # ax.set_xticks(
    #     ticks = np.arange(0, 2, 1), 
    #     labels = ["HS", "N1", "HI"]   ,
    #     font = font, 
    #     fontsize = 10)
    
    axs[i_p].set_yticks(
        ticks = np.arange(0, 1.2, .2), 
        labels = np.arange(0, 120, 20), 
        font = font, fontsize = 12)
    sns.despine()

fig.tight_layout(pad=1.5)

# plt.savefig(
#     os.path.join(path_usleep, "figs", "proba_probe_me_subtype.png"), 
#     dpi=200
#     )
