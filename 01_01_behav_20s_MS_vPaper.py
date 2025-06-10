    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:10:05 2023

@author: arthurlecoz

01_01_behav_20s_MS.py
"""
# %% Paths
import os
import SLHIP_config_ALC as cfg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
# from highlight_text import fig_text
from matplotlib.font_manager import FontProperties
from scipy.io import loadmat
from glob import glob
from yasa import transition_matrix

# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

rootpath = cfg.rootpath
rawpath = os.path.join(rootpath, "00_Raw")
preprocpath = os.path.join(rootpath, "01_Preproc")
behavpath = os.path.join(rootpath, "02_BehavResults")

taskepochspath = os.path.join(preprocpath, "epochs_epochs")
probespath = os.path.join(preprocpath, "raws_probes")
restingspath = os.path.join(preprocpath, "raws_restings")
autorejectpath = os.path.join(preprocpath, "autoreject")
icapath = os.path.join(preprocpath, "ica_files")
# figpath = os.path.join(behavpath)


test_col = ["nblock","block_cond","image","ntrial","digit","nogo_digit",
           "resp_key","stim_onset","dur_pres","resp_time","corr_nogo",
           "corr_go"]

probe_col = [
    "nprobe","t_probe_th","t_probe_act","nblock","block_cond","ntrial",
    "PQ1_respkey","PQ2_respkey","PQ3_respkey",
    "PQ1_resptime","PQ2_resptime","PQ3_resptime",
    "PQ1_questime","PQ2_questime","PQ3_questime",
    "PQ1_respval","PQ2_respval","PQ3_respval"
    ]

probe_int_str = {
    1 : "ON",
    2 : "MW_I",
    3 : "MW_E",
    4 : "MW_H",
    5 : "MB",
    6 : "FORGOT",
    0 : "MISS"
    }
sesstype = {0:'AM', 1:'PM'}

sub_ids = np.unique(np.array(
    [file.split('experiment/')[1].split('/')[0] for file 
     in glob(os.path.join(rawpath, "experiment", "**", "*.mat"))]
    ))

ms_keydic = {}

# subtype_palette = ["#4a5759", "#bf0603", "#ffc300"]
subtype_palette = ["#8d99ae", "#d00000"]
ms_palette = ["#FFC000", "#00B050", "#0070C0", "#7030A0", "#7F7F7F"]

# subtype_palette = ["#8d99ae", "#d00000"]

# %% functions

def filter_behav(mat, nblock):
    """ Filter data by nblock and further by 'rt' conditions. """
    return mat.loc[(mat["nblock"] == nblock) & ((mat['rt'] > 0.15) | (mat['rt'].isna()))]
def filter_probe(mat, nblock):
    """ Filter data by nblock and further by 'rt' conditions. """
    return mat.loc[(mat["nblock"] == nblock)]

def append_to_lists(list_dict, **kwargs):
    """ Append multiple values to respective lists efficiently. """
    for key, value in kwargs.items():
        list_dict[key].append(value)

def calculate_behavioral_metrics(interprobe_mat):
    """ Calculate behavioral metrics such as hits, misses, correct rejections, and false alarms. """
    interprobe_mat_20s = pd.concat([
        interprobe_mat[interprobe_mat['digit'] != 3].iloc[-8:],
        interprobe_mat[interprobe_mat['digit'] == 3].iloc[-2:]
    ])
    go_trials = interprobe_mat_20s[interprobe_mat_20s['digit'] != 3]
    go_corr = interprobe_mat_20s[
        (interprobe_mat_20s['digit'] != 3) 
        & (interprobe_mat_20s['corr_go'] == 1)
        ]
    nogo_trials = interprobe_mat_20s[interprobe_mat_20s['digit'] == 3]
    hits = 100 * len(go_trials[go_trials['corr_go'] == 1]) / len(go_trials)
    miss = 100 * (1 - hits / 100)
    if not len(nogo_trials) :
        cr = np.nan
        fa = np.nan
    else : 
        cr = 100 * len(nogo_trials[nogo_trials['corr_nogo'] == 1]) / len(nogo_trials)
        fa = 100 * (1 - cr / 100)
    rtgo = go_corr['rt'].mean()
    std_rtgo = go_corr['rt'].std()
    rtnogo = nogo_trials['rt'].mean()
    std_rtnogo = nogo_trials['rt'].std()
    return hits, miss, cr, fa, rtgo, rtnogo, std_rtgo, std_rtnogo


# %% Loop

columns = ['sub_id', 'subtype', 'nblock', 'probe', 
           'rt_go', 'rt_nogo', 'std_rtgo', 'std_rtnogo', 
           'hits', 'miss', 'correct_rejections', 'false_alarms', 
           'mindstate', 'voluntary', 'sleepiness', 'daytime']

data_dict = {col: [] for col in columns}

for sub_id in sub_ids :
    if sub_id == 'sub_HS_008' : continue
    this_files = glob(os.path.join(rawpath, "experiment", sub_id, "*.mat"))
    subtype = sub_id.split('_')[1]
    testlist = []
    probelist = []
    for i_file, file in enumerate(this_files) :
        temp_results = loadmat(file)
        
        df_test = pd.DataFrame(
                temp_results['test_res'], 
                columns = test_col)
        df_test['rt'] = df_test["resp_time"] - df_test["stim_onset"]
        
        df_probe = pd.DataFrame(
            temp_results['probe_res'], 
            columns = probe_col)
        
        if sub_id == 'sub_N1_015' and sesstype[i_file] == 'PM' :
            df_probe = df_probe.loc[df_probe.nblock!=4]
            df_test = df_test.loc[df_test.nblock!=4]
        
        if any(df_probe.PQ1_respval.isna()) :
            df_probe.PQ1_respval.replace(np.nan, 0, inplace = True)
        
        session = sesstype[i_file]
        df_test['session_type'] = [session for i in range(len(df_test))]
        df_probe['session_type'] = [session for i in range(len(df_probe))]
        df_probe["mindstate"] = [probe_int_str[value] for value in df_probe.PQ1_respval]
     
        # Predefined lists for data collection
        for nblock in df_test['nblock'].unique():
            
            block_mat_test = filter_behav(df_test, nblock)
            block_mat_probe = filter_probe(df_probe, nblock)
            for i, pause_trial in enumerate(block_mat_probe["ntrial"]):
                if i == 0 :
                    interprobe_mat = block_mat_test.loc[
                        block_mat_test['ntrial'] < pause_trial
                            ]
                else :
                    interprobe_mat = block_mat_test.loc[
                            (block_mat_test['ntrial'] > block_mat_probe["ntrial"].iloc[i-1])
                            & (block_mat_test['ntrial'] < pause_trial)
                                ]
                probe_submat = block_mat_probe.iloc[i]
    
                # Behavioral metrics
                hits, miss, cr, fa, rtgo, rtnogo, std_rtgo, std_rtnogo = calculate_behavioral_metrics(interprobe_mat)
                append_to_lists(
                    data_dict, 
                    sub_id=sub_id, 
                    subtype=subtype, 
                    nblock=nblock, 
                    probe=i+1,
                    rt_go=rtgo, 
                    rt_nogo=rtnogo,
                    std_rtgo=std_rtgo,
                    std_rtnogo=std_rtnogo,
                    hits=hits, 
                    miss=miss, 
                    correct_rejections=cr, 
                    false_alarms=fa, 
                    mindstate=probe_submat['mindstate'], 
                    voluntary=probe_submat['PQ2_respval'], 
                    sleepiness=probe_submat['PQ3_respval'], 
                    daytime=session
                    )

# Convert lists to DataFrame and save to CSV
df = pd.DataFrame(data_dict)
df.to_csv(os.path.join(behavpath, "VDF_dfBEHAV_SLHIP_20sbProbe.csv"))

# %% DF Manip

sub_df = df.loc[(df.subtype != 'HI') & (df.mindstate != 'MISS') & (df.mindstate != "MW_E")]

total_block = []
total_probe = []
for sub_id in sub_df.sub_id.unique() :
    totblock = 0
    totprobe = 0
    subject_df = sub_df.loc[sub_df.sub_id == sub_id]
    for daytime in subject_df.daytime.unique() :
        daytime_df = subject_df.loc[sub_df.daytime == daytime]
        for block in daytime_df.nblock.unique():
            totblock +=1 
            block_df = daytime_df.loc[daytime_df.nblock == block]
            for probe in block_df.probe.unique():
                totprobe += 1
                
                total_block.append(totblock)
                total_probe.append(totprobe)

sub_df['total_block'] = total_block
sub_df['total_probe'] = total_probe

sub_df.to_csv(os.path.join(behavpath, "NT1_CTL", "df_20s_behav.csv"))

block_df = sub_df[['sub_id', 'subtype', 'nblock', 'rt_go', 'rt_nogo','std_rtgo',
       'std_rtnogo', 'hits', 'miss', 'correct_rejections', 'false_alarms', 
       'mindstate', 'voluntary', 'sleepiness', 'daytime']].groupby(
           ['sub_id', 'subtype', 'mindstate', 'nblock', 'daytime'], 
           as_index = False).mean()

this_df = sub_df[['sub_id', 'subtype','rt_go', 'rt_nogo','std_rtgo',
       'std_rtnogo','hits', 'miss','correct_rejections', 'false_alarms', 
       'mindstate', 'sleepiness', 'daytime']].groupby(
           ['sub_id', 'subtype', 'mindstate', 'daytime'], 
           as_index = False).mean()
           
hs_df = this_df.loc[
    (this_df.subtype == 'HS') 
    & (this_df.mindstate != 'MW_E')
    & (this_df.mindstate != 'FORGOT')
    & (this_df.mindstate != 'MW_H')
    ]

# %% RadarPlot % MS / Subtype

# Calculate percentages
mindstate_counts = sub_df.groupby(['subtype', 'mindstate']).size().unstack(fill_value=0)
mindstate_percentages = mindstate_counts.div(mindstate_counts.sum(axis=1), axis=0) * 100

# Create radar plot for each subtype
# subtypes = ['HS', 'N1', 'HI']
subtypes = ['HS', 'N1']
kind_thought = list(sub_df.mindstate.unique())
full_thought = ["ON", "MW", "Distracted", "MB", "Forgot", "Hallucination"]
# kind_thought.remove('MISS')  # Remove 'MISS' from the radar plot

max_value = mindstate_percentages.max().max()

# Radar plot settings
colors = {'HS': '#8d99ae', 'N1': '#d00000', 'HI' : '#539F41'}

fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4, 4), dpi=150)
for subtype in subtypes:
    values = mindstate_percentages.loc[subtype, kind_thought].values.tolist()
    values += values[:1]  # to close the radar plot

    angles = np.linspace(0, 2 * np.pi, len(kind_thought), endpoint=False).tolist()
    angles += angles[:1]  # to close the radar plot

    ax.fill(angles, values, color=colors[subtype], alpha=0.25)
    ax.plot(angles, values, color=colors[subtype], linewidth=2, label=subtype)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(full_thought, font = bold_font, fontsize = 14)
ax.set_ylim(0, 60)
ax.set_yticks(np.linspace(0, 60, 4))
ax.set_yticklabels(
    [f'{int(tick)}%' for tick in np.linspace(0, 60, 4)], 
    color='grey',
    font = font)
ax.yaxis.set_ticks_position('left')

# Make grid lines more discrete
ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)

# Remove the outer circle
ax.spines['polar'].set_visible(False)

fig.tight_layout()
plt.savefig(
    os.path.join(behavpath, "NT1_CTL", "radar_plot_mindstates.png"), 
    dpi=200
    )
plt.show()

# %% RadarPlot | Behavioral Diff MS

df_rplot = sub_df[[
    'rt_go', 'std_rtgo', 'miss','false_alarms','mindstate', 'sleepiness']
    ].copy()
mean_rplot = df_rplot.groupby('mindstate').mean()

dic_label = {
    "ON" : "ON",
    "MW_I": "MW",
    "MB": "MB",
    "MW_H": "HALLU", 
    "FORGOT": "FORGOT"
    }

# (2) Choose a padding p = q = 0.1 (so min→0.1, max→0.9). If you want ON to be at ~0.2 instead,
#     you could set p=0.2 and q=0.1 (or some other pair). Here we’ll do p=q=0.1 for illustration:

p = 0.1
q = 0.1
span = 1.0 - p - q   # = 0.8

# Create a new DataFrame “df_padded” that linearly rescales each column from [raw_min,raw_max] → [p,1−q]:
df_padded = mean_rplot.copy()

for col in df_padded.columns:
    col_min = mean_rplot[col].min()
    col_max = mean_rplot[col].max()
    # Avoid any division‐by‐zero if col_max == col_min (not the case here, but good practice)
    if col_max == col_min:
        df_padded[col] = p + span / 2.0   # collapse to midpoint of [p,1−q]
    else:
        df_padded[col] = (
            (mean_rplot[col] - col_min) 
            / (col_max - col_min) 
            * span 
            + p
        )

# Now, df_padded holds values ∈ [0.1, 0.9] (ON rows are ≈0.1, FORGOT rows ≈0.9).
# Let’s check quickly:
print("Original min/max of each column:")
print(mean_rplot.agg(['min','max']))
print("\nAfter padding, min/max of each column:")
print(df_padded.agg(['min','max']))

# (3) Build angles for a 5‐axis radar (and close the loop).
categories = list(df_padded.columns)
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # duplicate first angle at end to close the polygon

# (4) Split into five small subplots (1 row × 5 cols), each with its own mindstate in blue,
#     but also draw the ON‐ and FORGOT‐curves (if you like) in the background as reference.

fig, axes = plt.subplots(1, 5, figsize=(14, 3), subplot_kw={'projection': 'polar'})

# Grab the padded‐norm vectors for ON and FORGOT, so we can overlay them on every subplot.
on_padded     = df_padded.loc['ON'].tolist() + [df_padded.loc['ON'].tolist()[0]]
forgot_padded = df_padded.loc['FORGOT'].tolist() + [df_padded.loc['FORGOT'].tolist()[0]]

for ax, mindstate, i in zip(axes, (["ON", "MW_I", "MB", "MW_H", "FORGOT"]), range(5)):
    
    this_color = ms_palette[i]
    # (a) Plot ON (thin dashed gray) and FORGOT (thin dashed gray) as reference
    ax.plot(angles, on_padded,      linestyle='--', color='gray', linewidth=1)#, label='ON   (≈0.1)')
    ax.plot(angles, forgot_padded,  linestyle='--', color='gray', linewidth=1)#, label='FORGOT   (≈0.9)')
    
    # (b) Plot this mindstate’s padded values (solid blue):
    padded_vals = df_padded.loc[mindstate].tolist()
    padded_vals += padded_vals[:1]
    ax.plot(angles, padded_vals, linewidth=2, label=mindstate, color = this_color)
    ax.fill(angles, padded_vals, alpha=0.3, color = this_color)
    
    # (c) Annotate each spoke with its raw (un‐normalized) number:
    raw_vals = mean_rplot.loc[mindstate].tolist()
    raw_vals += raw_vals[:1]
    # for angle, pad_val, raw in zip(angles, padded_vals, raw_vals):
    #     if mindstate in ["MB", "MW_H", "FORGOT"] :
    #         ax.text(
    #             angle, 
    #             pad_val - 0.15, 
    #             f"{raw:.2f}",
    #             ha='center', 
    #             va='center', 
    #             fontsize=7,
    #             font = font,
    #             color='black'
    #             )
    #     else : 
    #         ax.text(
    #             angle, 
    #             pad_val + 0.10, 
    #             f"{raw:.2f}",
    #             ha='center', 
    #             va='center', 
    #             fontsize=7,
    #             font = font,
    #             color='black'
    #             )
    
    # (d) Tidy up labels & limits:
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ["RT Go", "STD RT Go", "Misses", "False Alarms", "Sleepiness"], 
        font = font, fontsize=10)
    # Give a bit of radial margin so ON=0.1 and FORGOT=0.9 don't hug the exact center/edge
    ax.set_ylim(0.0, 1.0)  
    # If you want any radial ticks visible, you can do something like
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])  # keep whichever rings you like
    ax.set_yticklabels([])      # hide the numeric labels if you prefer
    ax.set_yticks([])
    ax.set_title(dic_label[mindstate], font = bold_font, fontsize=14)
    # Hide the circular border
    ax.spines['polar'].set_visible(False)
    
    # (Optional) also turn off all radial/grid lines:
    ax.grid(True)
    ax.set_ylim(0, .9)

# (5) One shared legend for the ON/FORGOT reference lines + each mindstate’s label:
# handles, labels = axes[-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9)

plt.suptitle("Five Mindstates (with ON≈0.1 & FORGOT≈0.9 as Thresholds)", y=1.05)
# plt.tight_layout()
plt.show()
plt.savefig(os.path.join(behavpath, "NT1_CTL", "RadarPlots_MS_Behav.png"), dpi=300)

# %% Separate Radarplots

for i, mindstate in enumerate(["ON", "MW_I", "MB", "MW_H", "FORGOT"]) :
    fig, ax = plt.subplots(
        1, 1, figsize=(5, 5), subplot_kw={'projection': 'polar'}
        )
    
    # Grab the padded‐norm vectors for ON and FORGOT, so we can overlay them on every subplot.
    on_padded     = df_padded.loc['ON'].tolist() + [df_padded.loc['ON'].tolist()[0]]
    forgot_padded = df_padded.loc['FORGOT'].tolist() + [df_padded.loc['FORGOT'].tolist()[0]]
        
    this_color = ms_palette[i]
    # (a) Plot ON (thin dashed gray) and FORGOT (thin dashed gray) as reference
    # ax.plot(angles, on_padded,      linestyle='--', color='gray', linewidth=1)#, label='ON   (≈0.1)')
    # ax.plot(angles, forgot_padded,  linestyle='--', color='gray', linewidth=1)#, label='FORGOT   (≈0.9)')
    
    # (b) Plot this mindstate’s padded values (solid blue):
    padded_vals = df_padded.loc[mindstate].tolist()
    padded_vals += padded_vals[:1]
    ax.plot(angles, padded_vals, linewidth=2, label=mindstate, color = this_color)
    ax.fill(angles, padded_vals, alpha=0.3, color = this_color)
    
    # (c) Annotate each spoke with its raw (un‐normalized) number:
    raw_vals = mean_rplot.loc[mindstate].tolist()
    raw_vals += raw_vals[:1]
    # for angle, pad_val, raw in zip(angles, padded_vals, raw_vals):
    #     if mindstate in ["MB", "MW_H", "FORGOT"] :
    #         ax.text(
    #             angle, 
    #             pad_val - 0.15, 
    #             f"{raw:.2f}",
    #             ha='center', 
    #             va='center', 
    #             fontsize=7,
    #             font = font,
    #             color='black'
    #             )
    #     else : 
    #         ax.text(
    #             angle, 
    #             pad_val + 0.10, 
    #             f"{raw:.2f}",
    #             ha='center', 
    #             va='center', 
    #             fontsize=7,
    #             font = font,
    #             color='black'
    #             )
    
    # (d) Tidy up labels & limits:
    ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(
    #     ["RT Go", "STD RT Go", "Misses", "False Alarms", "Sleepiness"], 
    #     font = font, fontsize=10)
    ax.set_xticklabels([])
    # Give a bit of radial margin so ON=0.1 and FORGOT=0.9 don't hug the exact center/edge
    ax.set_ylim(0.0, 1.0)  
    # If you want any radial ticks visible, you can do something like
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])  # keep whichever rings you like
    ax.set_yticklabels([])      # hide the numeric labels if you prefer
    ax.set_yticks([])
    # ax.set_title(dic_label[mindstate], font = bold_font, fontsize=14)
    # Hide the circular border
    ax.spines['polar'].set_visible(False)
    
    # (Optional) also turn off all radial/grid lines:
    ax.grid(False)
    ax.set_ylim(0, .9)
    
    # (5) One shared legend for the ON/FORGOT reference lines + each mindstate’s label:
    # handles, labels = axes[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9)
    
    # plt.suptitle("Five Mindstates (with ON≈0.1 & FORGOT≈0.9 as Thresholds)", y=1.05)
    # plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(behavpath, "NT1_CTL", f"{mindstate}_Radar.png"), dpi=300)

# %% RadarPlot | Behavioral Diff MS

df_rplot = sub_df[[
    'rt_go', 'std_rtgo', 'miss','false_alarms','mindstate', 'sleepiness']
    ].copy()
mean_rplot = df_rplot.groupby('mindstate').mean()

dic_label = {
    "ON" : "ON",
    "MW_I": "MW",
    "MB": "MB",
    "MW_H": "HALLU", 
    "FORGOT": "FORGOT"
    }

# (2) Choose a padding p = q = 0.1 (so min→0.1, max→0.9). If you want ON to be at ~0.2 instead,
#     you could set p=0.2 and q=0.1 (or some other pair). Here we’ll do p=q=0.1 for illustration:

p = 0.1
q = 0.1
span = 1.0 - p - q   # = 0.8

# Create a new DataFrame “df_padded” that linearly rescales each column from [raw_min,raw_max] → [p,1−q]:
df_padded = mean_rplot.copy()

for col in df_padded.columns:
    col_min = mean_rplot[col].min()
    col_max = mean_rplot[col].max()
    # Avoid any division‐by‐zero if col_max == col_min (not the case here, but good practice)
    if col_max == col_min:
        df_padded[col] = p + span / 2.0   # collapse to midpoint of [p,1−q]
    else:
        df_padded[col] = (
            (mean_rplot[col] - col_min) 
            / (col_max - col_min) 
            * span 
            + p
        )

# Now, df_padded holds values ∈ [0.1, 0.9] (ON rows are ≈0.1, FORGOT rows ≈0.9).
# Let’s check quickly:
print("Original min/max of each column:")
print(mean_rplot.agg(['min','max']))
print("\nAfter padding, min/max of each column:")
print(df_padded.agg(['min','max']))

# (3) Build angles for a 5‐axis radar (and close the loop).
categories = list(df_padded.columns)
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # duplicate first angle at end to close the polygon

# (4) Split into five small subplots (1 row × 5 cols), each with its own mindstate in blue,
#     but also draw the ON‐ and FORGOT‐curves (if you like) in the background as reference.

fig, axes = plt.subplots(1, 5, figsize=(14, 3), subplot_kw={'projection': 'polar'})

# Grab the padded‐norm vectors for ON and FORGOT, so we can overlay them on every subplot.
on_padded     = df_padded.loc['ON'].tolist() + [df_padded.loc['ON'].tolist()[0]]
forgot_padded = df_padded.loc['FORGOT'].tolist() + [df_padded.loc['FORGOT'].tolist()[0]]

for ax, mindstate, i in zip(axes, (["ON", "MW_I", "MB", "MW_H", "FORGOT"]), range(5)):
    
    this_color = ms_palette[i]
    # (a) Plot ON (thin dashed gray) and FORGOT (thin dashed gray) as reference
    ax.plot(angles, on_padded,      linestyle='--', color='gray', linewidth=1)#, label='ON   (≈0.1)')
    ax.plot(angles, forgot_padded,  linestyle='--', color='gray', linewidth=1)#, label='FORGOT   (≈0.9)')
    
    # (b) Plot this mindstate’s padded values (solid blue):
    padded_vals = df_padded.loc[mindstate].tolist()
    padded_vals += padded_vals[:1]
    ax.plot(angles, padded_vals, linewidth=2, label=mindstate, color = this_color)
    ax.fill(angles, padded_vals, alpha=0.3, color = this_color)
    
    # (c) Annotate each spoke with its raw (un‐normalized) number:
    raw_vals = mean_rplot.loc[mindstate].tolist()
    raw_vals += raw_vals[:1]
    # for angle, pad_val, raw in zip(angles, padded_vals, raw_vals):
    #     if mindstate in ["MB", "MW_H", "FORGOT"] :
    #         ax.text(
    #             angle, 
    #             pad_val - 0.15, 
    #             f"{raw:.2f}",
    #             ha='center', 
    #             va='center', 
    #             fontsize=7,
    #             font = font,
    #             color='black'
    #             )
    #     else : 
    #         ax.text(
    #             angle, 
    #             pad_val + 0.10, 
    #             f"{raw:.2f}",
    #             ha='center', 
    #             va='center', 
    #             fontsize=7,
    #             font = font,
    #             color='black'
    #             )
    
    # (d) Tidy up labels & limits:
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ["RT Go", "STD RT Go", "Misses", "False Alarms", "Sleepiness"], 
        font = font, fontsize=10)
    # Give a bit of radial margin so ON=0.1 and FORGOT=0.9 don't hug the exact center/edge
    ax.set_ylim(0.0, 1.0)  
    # If you want any radial ticks visible, you can do something like
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])  # keep whichever rings you like
    ax.set_yticklabels([])      # hide the numeric labels if you prefer
    ax.set_yticks([])
    ax.set_title(dic_label[mindstate], font = bold_font, fontsize=14)
    # Hide the circular border
    ax.spines['polar'].set_visible(False)
    
    # (Optional) also turn off all radial/grid lines:
    ax.grid(True)
    ax.set_ylim(0, .9)

# (5) One shared legend for the ON/FORGOT reference lines + each mindstate’s label:
# handles, labels = axes[-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9)

plt.suptitle("Five Mindstates (with ON≈0.1 & FORGOT≈0.9 as Thresholds)", y=1.05)
# plt.tight_layout()
plt.show()
plt.savefig(os.path.join(behavpath, "NT1_CTL", "RadarPlots_MS_Behav.png"), dpi=300)

    # %% ready fig ms %

# coi = ['sub_id', 'subtype', 'mindstate', 'percentage']
coi = ['sub_id', 'subtype', 'daytime', 'mindstate', 'percentage']
dic = {c : [] for c in coi}

for sub_id in sub_df.sub_id.unique() :    
    this_df = sub_df.loc[sub_df['sub_id'] == sub_id]
    for dt in this_df.daytime.unique() :
        df_dt = this_df.loc[this_df['daytime'] == dt]
        for mindstate in ['ON', 'MW_I', 'MW_E', 'MW_H', 'MB', 'FORGOT'] :
            dic['sub_id'].append(sub_id)
            dic['subtype'].append(sub_id.split('_')[1])
            dic['daytime'].append(dt)
            dic['mindstate'].append(mindstate)
            dic['percentage'].append(
                len(df_dt.mindstate.loc[
                    (df_dt['mindstate'] == mindstate)]
                    )/len(df_dt.mindstate))
            # dic['sleepiness'].append(
            #     this_df.sleepiness.loc[
            #         (this_df['mindstate'] == mindstate)].mean()
            #         )

df_mindstate = pd.DataFrame.from_dict(dic)
df_mindstate.to_csv(os.path.join(
    behavpath, "NT1_CTL", "per_ms.csv"
    ))

# %% fig ms % Subtype

data = df_mindstate[['sub_id', 'subtype', 'mindstate', 'percentage']].groupby(
    ['sub_id', 'subtype', 'mindstate'], as_index = False
    ).mean()
y = 'percentage'
x = "mindstate"
# order = ['ON', 'MW_I', 'MB', 'MW_H']
order = ['ON', 'MW_H', 'MW_I', 'MB', 'FORGOT']
# order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT', 'MW_E']
hue = "subtype"
hue_order = ['HS', 'N1']    
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 4))
     
sns.barplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    errorbar='se', 
    n_boot=1000, 
    orient=None, 
    palette=subtype_palette, 
    fill=False,
    hue_norm=None, 
    width=0.8, 
    dodge='auto', 
    gap=0, 
    capsize=0.05, 
    ax=ax,
    legend=None
    )
# sns.stripplot(
#     data = data, 
#     x = x,
#     y = y,
#     hue = hue,
#     order = order,
#     hue_order = hue_order,
#     alpha = 0.1,
#     dodge = True,
#     legend = None,
#     palette = subtype_palette
#     )

ax.set_ylabel('Pourcentage %', size = 24, font = bold_font)
ax.set_xlabel('Mindstate', size = 24, font = bold_font)
ax.set_ylim(0, 1)
ax.set_xticks(
    ticks = np.arange(0, 5, 1), 
    # labels = ["ON", "MW", "MB", "HALLU", "FGT", 'DTD'],
    labels = ["ON", "HALLU", "MW", "MB", "FGT"],
    # labels = ["ON", "MW", "MB", "HALLU"],
    font = font, 
    fontsize = 16)
ax.set_yticks(
    ticks = np.arange(0, 1.2, .2), 
    labels = np.arange(0, 120, 20), 
    font = font, 
    fontsize = 16)
sns.despine()
fig.tight_layout()

plt.savefig(f"{behavpath}/point_strip_per_mindstates_by_subtype.png", dpi=200)

# %% Stats

temp_df = df_mindstate[['sub_id', 'subtype', 'mindstate', 'percentage']].groupby(
    ['sub_id', 'subtype', 'mindstate'], as_index = False
    ).mean()
temp_df = temp_df.loc[temp_df.mindstate.isin(order)]

model_formula = 'percentage ~ C(mindstate, Treatment("FORGOT")) * C(subtype, Treatment("HS"))'
model = smf.mixedlm(
    model_formula, 
    df_mindstate, 
    groups=df_mindstate['sub_id'], 
    missing = 'omit'
    )
model_result = model.fit()
print(model_result.summary())

# %% Ready figure % Sleepi

coi = ['sub_id', 'subtype', 'daytime', 'sleepiness', 'percentage']
dic = {c : [] for c in coi}

for sub_id in df.sub_id.unique() :    
    this_df = df.loc[df['sub_id'] == sub_id]
    for dt in this_df.daytime.unique() :
        df_dt = this_df.loc[this_df['daytime'] == dt]
        for sleepi_level in np.linspace(1, 9, 9) :
            dic['sub_id'].append(sub_id)
            dic['subtype'].append(sub_id.split('_')[1])
            dic['daytime'].append(dt)
            dic['sleepiness'].append(sleepi_level)
            dic['percentage'].append(
                len(df_dt.sleepiness.loc[
                    (df_dt['sleepiness'] == sleepi_level)]
                    )/len(df_dt.sleepiness))

df_sleepi = pd.DataFrame.from_dict(dic)

# %% Sleepiness  (full_scale, use df) [MS x ST Diff]
 
data = df_sleepi.copy()
data = data.loc[data.subtype != "HI"]

x = 'sleepiness'
order = np.linspace(1, 9, 9)
y = 'percentage'
hue = 'subtype'
hue_order = ['HS', 'N1']

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))

sns.barplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    errorbar='se', 
    orient=None, 
    palette=subtype_palette, 
    fill=False,
    hue_norm=None, 
    width=0.8, 
    dodge='auto', 
    gap=0, 
    capsize=0.05, 
    ax=ax
    )

ax.set_yticks(
    ticks = np.linspace(0, .4, 5), 
    labels = np.linspace(0, 40, 5).astype(int), 
    font = font, 
    size = 10)
ax.set_ylim(0, .4)
ax.set_ylabel("Percentage %", font = bold_font, size = 18)
ax.set_xticks(
    ticks = np.linspace(0, 8, 9), 
    labels = np.linspace(1, 9, 9).astype(int), 
    size = 14,
    font = font
    )
ax.set_xlabel("Sleepiness", font = bold_font, size = 18)
ax.tick_params(axis='both', labelsize=16)
sns.despine()

fig.tight_layout(pad = 2)
plt.savefig(
    os.path.join(behavpath, 
                 "NT1_CTL", 
                 "point_strip_sleepiness_ms_subtype.png"), 
    dpi=200 
    )

# %% LME Sleepiness

"""I should loop around 1 to 9 and compute the corrected p_vals then print df"""

data['sleepiness'] = pd.Categorical(data['sleepiness'], ordered=True)

model_formula = 'percentage ~ C(sleepiness, Treatment(9)) * C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, data=data, groups = data['sub_id'])
result = model.fit()

print(result.summary())

# %% Behav Joint [MSxST diff]

this_df = sub_df[
    ['sub_id', 'subtype','rt_go', 'rt_nogo', 'std_rtgo', 'hits', 'miss',
     'correct_rejections', 'false_alarms', 'mindstate', 'sleepiness']
    ].groupby(['sub_id', 'subtype', 'mindstate'], as_index = False).mean()

data = this_df
x = 'mindstate'
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']
hue = 'subtype'
hue_order = ['HS', 'N1']

fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize = (3, 8), sharex = True)

y = 'miss'
sns.boxplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    palette=subtype_palette, 
    fill=False,
    width=0.8, 
    dodge='auto', 
    gap=0, 
    ax=ax[0]
    )

ax[0].set_yticks(
    ticks = np.linspace(0, 80, 5), 
    labels =  np.linspace(0, 80, 5).astype(int), 
    font = font, 
    size = 10)
ax[0].set_ylim(0, 80)
ax[0].set_ylabel("Misses (%)", font = bold_font, size = 15)


y = "false_alarms"
sns.boxplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    palette=subtype_palette, 
    fill=False, 
    width=0.8, 
    dodge='auto', 
    gap=0, 
    ax=ax[1]
    )

ax[1].set_ylabel("False Alarms (%)", font = bold_font, size = 15)
ax[1].set_yticks(
    ticks = np.linspace(0, 100, 5), 
    labels = np.linspace(0, 100, 5).astype(int), 
    font = font, 
    size = 10)
ax[1].set_ylim(0, 100)


y = 'rt_go'
sns.boxplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    palette=subtype_palette, 
    fill=False,
    width=0.8, 
    dodge='auto', 
    gap=0, 
    ax=ax[2]
    )

ax[2].set_yticks(
    ticks = np.linspace(0.3, 0.8, 6), 
    labels = np.round(np.linspace(0.3, 0.8, 6), 1), 
    font = font, 
    size = 10)
ax[2].set_ylabel("RT Go (ms)", font = bold_font, size = 15)


y = 'std_rtgo'

sns.boxplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    palette=subtype_palette, 
    fill=False,
    width=0.8, 
    dodge='auto', 
    gap=0, 
    ax=ax[3]
    )

ax[3].set_yticks(
    ticks = np.linspace(0.05, 0.25, 5), 
    labels = np.round(np.linspace(0.05, 0.25, 5),2), 
    font = font, 
    size = 10)
ax[3].set_ylim(0.03, 0.26)
ax[3].set_ylabel("STD RT Go (ms)", font = bold_font, size = 15)
ax[3].set_xticks(
    ticks = np.arange(0, 5, 1), 
    labels = ["ON", "MW", "MB", "Hallu", "Forgot"], 
    size = 20,
    font = font
    )
ax[3].set_xlabel("Mindstate", font = bold_font, size = 15)
sns.despine(fig, bottom = True)

for i in range(3):
    ax[i].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        )
for i in range(4):
    ax[i].get_legend().remove()
    
fig.tight_layout(pad=1)

plt.savefig(os.path.join(
    behavpath, "NT1_CTL", "miss_fa_rt_subtype_n1.png"), 
    dpi=200
    )

# %% Separate plots [MSxST diff]

feats = ["sleepiness", "miss", "false_alarms", "rt_go", "std_rtgo"]
this_df = sub_df[
    ['sub_id', 'subtype','rt_go', 'rt_nogo', 'std_rtgo', 'hits', 'miss',
     'correct_rejections', 'false_alarms', 'mindstate', 'sleepiness']
    ].groupby(['sub_id', 'subtype', 'mindstate'], as_index = False).mean()

minmax = {
    "sleepiness" : [1, 9],
    "miss" : [0, 100],
    "false_alarms" : [0, 100],
    "rt_go" : [.2, .8],
    "std_rtgo" : [0, .3]
    }
labels = {
    "sleepiness" : "Sleepiness",
    "miss" : "Misses (%)",
    "false_alarms" : "False Alarms (%)",
    "rt_go" : "Reaction Time (ms)",
    "std_rtgo" : "Standard Deviation RT"
    }
ticks = {
    "sleepiness" : [np.linspace(1,9,9), np.linspace(1,9,9).astype(int)],
    "miss" : [np.linspace(0,100,6), np.linspace(0,100,6).astype(int)],
    "false_alarms" : [np.linspace(0,100,6), np.linspace(0,100,6).astype(int)],
    "rt_go" : [np.linspace(.2,.8, 7), np.round(np.linspace(.2,.8,7),1)],
    "std_rtgo" : [np.linspace(0,.3, 4), np.round(np.linspace(0,.3,4),1)]
    }

data = this_df
x = 'mindstate'
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']
hue = 'subtype'
hue_order = ['HS', 'N1']

for feat in feats :
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (3, 3))

    y = feat
    sns.boxplot(
        data = data, 
        x = x,
        y = y,
        hue = hue,
        order = order,
        hue_order = hue_order,
        palette=subtype_palette, 
        fill=False,
        width=0.8, 
        dodge='auto', 
        gap=0, 
        ax=ax
        )
    sns.despine()
    fig.tight_layout()
    
    ax.set_xticks(
        np.linspace(0, len(order)-1, len(order)),
        ['ON', 'MW', 'MB', 'HALLU', 'FGT'],
        font = font, 
        fontsize = 12
        )
    ax.set_xlabel("Mindstates", font=bold_font, fontsize=14)
    ax.get_legend().remove()
    ax.set_yticks(
        ticks = ticks[feat][0],
        labels = ticks[feat][1],
        font = font,
        fontsize = 12
        )
    ax.set_ylim(minmax[feat][0], minmax[feat][1])
    ax.set_ylabel(labels[feat], font = bold_font, size = 14)
    
    plt.savefig(os.path.join(
        behavpath, "NT1_CTL", f"{feat}_boxplot_groupdiff.png"
        ), dpi=300)
    
# %% Separate plots [MS diff]

ms_palette = ["#FFC000", "#00B050", "#0070C0", "#7030A0", "#7F7F7F"]

feats = ["sleepiness", "miss", "false_alarms", "rt_go", "std_rtgo"]
this_df = sub_df[
    ['sub_id', 'rt_go', 'rt_nogo', 'std_rtgo', 'hits', 'miss',
     'correct_rejections', 'false_alarms', 'mindstate', 'sleepiness']
    ].groupby(['sub_id', 'mindstate'], as_index = False).mean()

minmax = {
    "sleepiness" : [1, 9],
    "miss" : [0, 100],
    "false_alarms" : [0, 100],
    "rt_go" : [.2, .8],
    "std_rtgo" : [0, .3]
    }
labels = {
    "sleepiness" : "Sleepiness",
    "miss" : "Misses (%)",
    "false_alarms" : "False Alarms (%)",
    "rt_go" : "Reaction Time (ms)",
    "std_rtgo" : "Standard Deviation RT"
    }
ticks = {
    "sleepiness" : [np.linspace(0,8,9), np.linspace(1,9,9).astype(int)],
    "miss" : [np.linspace(0,100,6), np.linspace(0,100,6).astype(int)],
    "false_alarms" : [np.linspace(0,100,6), np.linspace(0,100,6).astype(int)],
    "rt_go" : [np.linspace(.2,.8, 7), np.round(np.linspace(.2,.8,7),1)],
    "std_rtgo" : [np.linspace(0,.3, 4), np.round(np.linspace(0,.3,4),1)]
    }

data = this_df
hue = 'mindstate'
hue_order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']

for feat in feats :
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (3, 3))

    y = feat
    sns.boxplot(
        data = data, 
        # x = x,
        y = y,
        hue = hue,
        # order = order,
        hue_order = hue_order,
        palette=ms_palette, 
        fill=False,
        width=0.8, 
        dodge='auto', 
        gap=.2, 
        ax=ax
        )
    sns.despine()
    
    ax.set_xticks(
        np.linspace(-0.33, 0.33, len(order)),
        ['ON', 'MW', 'MB', 'HA', 'FGT'],
        font = font, 
        fontsize = 12
        )
    ax.set_xlabel("Mindstates", font=bold_font, fontsize=14)
    ax.set_yticks(
        ticks = ticks[feat][0],
        labels = ticks[feat][1],
        font = font,
        fontsize = 12
        )
    ax.set_ylim(minmax[feat][0], minmax[feat][1])
    ax.set_ylabel(labels[feat], font = bold_font, size = 14)
    
    ax.get_legend().remove()
    
    fig.tight_layout()
    
    plt.savefig(os.path.join(
        behavpath, "NT1_CTL", f"{feat}_boxplot_msonly.png"
        ), dpi=300)
    
# %% Stats

comparison_oi = ["ON", "MW_I"]

feat_oi = "std_rtgo"
ms_oi = comparison_oi[0]

df_stats = this_df.loc[this_df.mindstate.isin(comparison_oi)]

model_formula = f'{feat_oi} ~ C(mindstate)'
model = smf.mixedlm(
    model_formula, 
    df_stats, 
    groups=df_stats['sub_id'], 
    missing = 'drop'
    )
model_result = model.fit()
print(model_result.summary())

# %% Compute TransMat MS

dic_stoi = {
    "ON" : 1,
    "MW_I" : 2,
    "MB" : 3,
    "MW_H" : 4,
    "FORGOT" : 5
    }
dic_itos = {
    1 : "ON", 
    2 : "MW", 
    3 : "MB", 
    4 : "HALLU", 
    5 : "FORGOT"
    }

coi = ["sub_id", "subtype", "daytime", "mindstate", 
       "proba_ON", "proba_MW", "proba_MB", "proba_HALLU", "proba_FORGOT"]
thisdic = {c : [] for c in coi}

for sub_id in sub_df.sub_id.unique() :
    subid_df = sub_df.loc[sub_df.sub_id==sub_id]
    subtype = subid_df.subtype.unique()[0]
    
    for daytime in ["AM", "PM"]:
        subid_df_am = subid_df.loc[subid_df.daytime == daytime]
        
        if subid_df_am.empty : continue
        actual_ms = np.asarray(
            [dic_stoi[ms] for ms in subid_df_am.mindstate.values]
            )
        unique_ms = np.sort(np.unique(actual_ms))
        _, probs = transition_matrix(actual_ms)
        np_probs = probs.to_numpy()
        
        temp_transmat = np.nan * np.empty((5, 5))
        
        for i in probs.index.values :
            for k in probs.columns.values :
                temp_transmat[i-1, k-1] = probs.loc[i][k]
                
        for i, j in enumerate(temp_transmat) :
            thisdic["sub_id"].append(sub_id)
            thisdic["subtype"].append(subtype)
            thisdic["daytime"].append(daytime)
            thisdic["mindstate"].append(dic_itos[i+1])
            thisdic["proba_ON"].append(j[0])
            thisdic["proba_MW"].append(j[1])
            thisdic["proba_MB"].append(j[2])
            thisdic["proba_HALLU"].append(j[3])
            thisdic["proba_FORGOT"].append(j[4])
            
df_transi = pd.DataFrame.from_dict(thisdic)
av_transi = df_transi[[
    'subtype', 'mindstate', 'proba_ON', 
    "proba_MW", "proba_MB",'proba_HALLU', 'proba_FORGOT']].groupby(
        by=['subtype', 'mindstate'], as_index=False).mean()
order = ["ON", "MW", "MB", "HALLU", "FORGOT"]
av_transi['mindstate'] = pd.Categorical(
    av_transi['mindstate'],
    categories=order,
    ordered=True
    )
av_transi = av_transi.sort_values('mindstate').reset_index(drop=True)
av_transi.set_index('mindstate', inplace=True)

transi_n1 = av_transi.loc[av_transi.subtype=="N1"]
transi_hs = av_transi.loc[av_transi.subtype=="HS"]

transi_n1.drop(columns="subtype", inplace=True)
transi_hs.drop(columns="subtype", inplace=True)

# %% Plot TransMat MS

grid_kws = {"height_ratios": (.9, .05), "hspace": .1}

f, (ax, cbar_ax) = plt.subplots(
    2, 
    gridspec_kw=grid_kws,
    figsize=(6, 6)
    )
sns.heatmap(
    transi_n1, 
    ax=ax, 
    square=False, 
    vmin=0, 
    vmax=1, 
    cbar=True,
    cbar_ax=cbar_ax, 
    cmap='Purples', 
    annot=True, 
    annot_kws={"size": 14},
    fmt='.2f',
    cbar_kws={
        "orientation": "horizontal", 
        "fraction": 0.1,
        "label": "Transition probability"}
    )
ax.set_xlabel("To Mental State", font=bold_font, fontsize=12)
ax.xaxis.tick_top()
ax.set_ylabel("From Mental State", font=bold_font, fontsize=12)
ax.xaxis.set_label_position('top')


ax.set_xticks(
    np.linspace(0.5,4.5,5),
    ['ON', 'MW', 'MB', 'HALLU', 'FORGOT'],
    font = font, 
    fontsize = 12
    )
ax.set_yticks(
    np.linspace(0.5,4.5,5),
    ['ON', 'MW', 'MB', 'HALLU', 'FORGOT'],
    font = font, 
    fontsize = 12
    )
ax.set_title("Narcolepsy Type 1", font = bold_font, fontsize = 14)
f.tight_layout()

plt.savefig(os.path.join(
    behavpath, "NT1_CTL", "transimat_mwmb_ms_nt1.png"
    ), dpi=300)

f, (ax, cbar_ax) = plt.subplots(
    2, 
    gridspec_kw=grid_kws,
    figsize=(6, 6)
    )
sns.heatmap(
    transi_hs, 
    ax=ax, 
    square=False, 
    vmin=0, 
    vmax=1, 
    cbar=True,
    cbar_ax=cbar_ax, 
    cmap='Purples', 
    annot=True, 
    annot_kws={"size": 14},
    fmt='.2f',
    cbar_kws={
        "orientation": "horizontal", 
        "fraction": 0.1,
        "label": "Transition probability"}
    )
ax.set_xlabel("To Mental State", font=bold_font, fontsize=12)
ax.xaxis.tick_top()
ax.set_ylabel("From Mental State", font=bold_font, fontsize=12)
ax.xaxis.set_label_position('top')

ax.set_xticks(
    np.linspace(0.5,4.5,5),
    ['ON', 'MW', 'MB', 'HALLU', 'FORGOT'],
    font = font, 
    fontsize = 12
    )
ax.set_yticks(
    np.linspace(0.5,4.5,5),
    ['ON', 'MW', 'MB', 'HALLU', 'FORGOT'],
    font = font, 
    fontsize = 12
    )
ax.set_title("Controls", font = bold_font, fontsize = 14)
f.tight_layout()
plt.savefig(os.path.join(
    behavpath, "NT1_CTL", "transimat_mwmb_ctl.png"
    ), dpi=300)


# %% Explore Dynamics (DAY)

this_df = sub_df.copy()

foi = 'false_alarms'

fig, ax = plt.subplots(nrows = 1, ncols = 1)
sns.pointplot(
    data = sub_df,
    x = 'total_block',
    y = foi,
    hue = 'subtype',
    hue_order = ['HS', 'N1', 'HI'],
    errorbar = "se",
    alpha = .8,
    palette = subtype_palette,
    linewidth = 3.5
    )
ax.set_ylabel(foi, font = bold_font, fontsize = 16)
ax.set_xlabel("Blocks throughout the day", font = bold_font, fontsize = 16)
ax.set_xticks(
    np.linspace(0,7,8), 
    ["1-AM", "2-AM", "3-AM", "4-AM",
     "1-PM", "2-PM", "3-PM", "4-PM"], 
    font = font, 
    fontsize = 12
    )
sns.despine()
fig.tight_layout()

# %% Explore Dynamics (SESSION)

this_df = sub_df.copy()
doubled_palette = dict(
    HS=["#8d99ae", "#d2d7df"],
    N1=["#d00000", "#ffbcbc"],
    HI=["#ffb703", "#ffe4a0"],
    )

foi = 'sleepiness'

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (12,5), sharex=True, sharey=True)
for i_st, dis_subtype in enumerate(subtypes) :
    ax = axs[i_st]
    ax.set_title(dis_subtype, font=bold_font, fontsize=14, color=subtype_palette[i_st])
    plot_df = sub_df.loc[sub_df.subtype == dis_subtype] 
   
    sns.pointplot(
        data = plot_df,
        x = 'nblock',
        y = foi,
        hue = 'daytime',
        hue_order = ['AM', 'PM'],
        errorbar = "se",
        alpha = .8,
        palette = doubled_palette[dis_subtype],
        linewidth = 3,
        ax = ax,
        capsize = .02
        )
    ax.set_ylabel(foi, font = bold_font, fontsize = 16)
    ax.set_xlabel("Blocks throughout the day", font = bold_font, fontsize = 16)
    ax.set_xticks(
        np.linspace(0,3,4), 
        ["1", "2", "3", "4"], 
        font = font, 
        fontsize = 12
        )
    sns.despine()
    fig.tight_layout()

figname = os.path.join(
    behavpath, 
    f"betweenblocks_ampm_{foi}.png"
    )
plt.savefig(figname, dpi=100)

# %% Dynamics Mindstates

coi = ['sub_id', 'subtype', 'sleepiness', 'mindstate', 
       'percentage', 'daytime', 'nblock']

dic = {c : [] for c in coi}

for sub_id in sub_df.sub_id.unique() :    
    this_df = sub_df.loc[sub_df['sub_id'] == sub_id]
    for dt in this_df.daytime.unique() :
        df_dt = this_df.loc[sub_df['daytime'] == dt]
        for bloc in df_dt.nblock.unique() :
            df_bloc = df_dt.loc[df_dt['nblock'] == bloc]
            for mindstate in ['ON', 'MW_I', 'MW_E', 'MW_H', 'MB', 'FORGOT'] :
                dic['sub_id'].append(sub_id)
                dic['subtype'].append(sub_id.split('_')[1])
                dic['nblock'].append(bloc)
                dic['daytime'].append(dt)
                dic['mindstate'].append(mindstate)
                dic['percentage'].append(
                    len(df_bloc.mindstate.loc[
                        (df_bloc['mindstate'] == mindstate)]
                        )/len(df_bloc.mindstate))
                dic['sleepiness'].append(
                    df_bloc.sleepiness.loc[
                        (df_bloc['mindstate'] == mindstate)].mean()
                        )

df_mindstate_bloc = pd.DataFrame.from_dict(dic)

this_df = df_mindstate_bloc.loc[
    df_mindstate_bloc.mindstate.isin(['ON', 'MW_I','MW_H', 'MB'])
    ]

doubled_ms_palette = dict(
    AM=["#46964D", "#E78B42", "#70A1C9", "#D50000"],
    PM=["#9ED1A2", "#F1BD93", "#C4D9E9", "#FF9999"]
    )

ms_palette = ["#46964D", "#E78B42", "#70A1C9", "#D50000"]

foi = 'percentage'

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (12,5), sharex=True, sharey=True)
for i_st, dis_subtype in enumerate(subtypes) :
    ax = axs[i_st]
    ax.set_title(dis_subtype, font=bold_font, fontsize=14, color=subtype_palette[i_st])
    for session in ['AM', 'PM'] :
    
        plot_df = this_df.loc[
            (this_df.subtype == dis_subtype)
            & (this_df.daytime == session)
            ] 
       
        sns.pointplot(
            data = plot_df,
            x = 'nblock',
            y = foi,
            hue = 'mindstate',
            hue_order = ['ON', 'MW_I','MB', 'MW_H'],
            errorbar = "se",
            alpha = .8,
            palette = doubled_ms_palette[session],
            linewidth = 3,
            ax = ax,
            capsize = .02,
            legend = None
            )
        ax.set_ylabel(foi, font = bold_font, fontsize = 16)
        ax.set_xlabel("Blocks throughout the day", font = bold_font, fontsize = 16)
        ax.set_xticks(
            np.linspace(0,3,4), 
            ["1", "2", "3", "4"], 
            font = font, 
            fontsize = 12
            )
        sns.despine()
        
fig.tight_layout()

figname = os.path.join(
    behavpath, 
    "betweenblocks_ampm_mindstate.png"
    )
plt.savefig(figname, dpi=100)

# %% ON, OFF, HALLU, FORGOT

onoff_df = df.loc[(df.subtype != 'HI') & (df.mindstate != 'MISS') & (df.mindstate != 'MW_E')]
onoff_df['mindstate'] = onoff_df['mindstate'].replace({'MW_I': 'OFF', 'MB': 'OFF'}) 

onoff_df.to_csv(os.path.join(
    behavpath, "NT1_CTL", "onoff_df.csv"
    ))

# %% Ready DF Per MS

coi = ['sub_id', 'subtype', 'daytime', 'mindstate', 'percentage']
dic = {c : [] for c in coi}

for sub_id in onoff_df.sub_id.unique() :    
    this_df = onoff_df.loc[onoff_df['sub_id'] == sub_id]
    for dt in this_df.daytime.unique() :
        df_dt = this_df.loc[this_df['daytime'] == dt]
        for mindstate in ['ON', 'OFF', 'MW_H', 'FORGOT'] :
            dic['sub_id'].append(sub_id)
            dic['subtype'].append(sub_id.split('_')[1])
            dic['daytime'].append(dt)
            dic['mindstate'].append(mindstate)
            dic['percentage'].append(
                len(df_dt.mindstate.loc[
                    (df_dt['mindstate'] == mindstate)]
                    )/len(df_dt.mindstate))

df_mindstate = pd.DataFrame.from_dict(dic)
df_mindstate.to_csv(os.path.join(
    behavpath, "NT1_CTL", "per_ms_onoff.csv"
    ))

# %% Figure Per MS

data = df_mindstate[['sub_id', 'subtype', 'mindstate', 'percentage']].groupby(
    ['sub_id', 'subtype', 'mindstate'], as_index = False
    ).mean()
y = 'percentage'
x = "mindstate"
order = ['ON', 'MW_H', 'FORGOT', 'OFF']
hue = "subtype"
hue_order = ['HS', 'N1']    
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 4))
     
sns.barplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    errorbar='se', 
    orient=None, 
    palette=subtype_palette, 
    fill=False,
    hue_norm=None, 
    width=0.8, 
    dodge='auto', 
    gap=0.1, 
    capsize=0.05, 
    ax=ax
    )

ax.get_legend().remove()

ax.set_ylabel('Pourcentage %', size = 16, font = bold_font)
ax.set_xlabel('Mindstate', size = 16, font = bold_font)
ax.set_ylim(0, 1)
ax.set_xticks(
    ticks = np.arange(0, 4, 1), 
    labels = ["ON", "HALLU", "FGT", "OFF"],
    font = font, 
    fontsize = 12)
ax.set_yticks(
    ticks = np.arange(0, 1.2, .2), 
    labels = np.arange(0, 120, 20), 
    font = font, 
    fontsize = 12)
sns.despine()
fig.tight_layout()

plt.savefig(os.path.join(
    behavpath, "NT1_CTL", "per_ms_group_onoff.png"
    ), dpi=200)

# %% Stats Per MS

temp_df = data.copy()

model_formula = 'percentage ~ C(mindstate, Treatment("ON")) * C(subtype, Treatment("HS"))'
model = smf.mixedlm(
    model_formula, 
    df_mindstate, 
    groups=df_mindstate['sub_id'], 
    missing = 'omit'
    )
model_result = model.fit(method="powell", maxiter=1000, tol=1e-6)
print(model_result.summary())

# %% Ready figure % Sleepi

coi = ['sub_id', 'subtype', 'daytime', 'sleepiness', 'percentage']
dic = {c : [] for c in coi}

for sub_id in df.sub_id.unique() :    
    this_df = onoff_df.loc[onoff_df['sub_id'] == sub_id]
    for dt in this_df.daytime.unique() :
        df_dt = this_df.loc[this_df['daytime'] == dt]
        for sleepi_level in np.linspace(1, 9, 9) :
            dic['sub_id'].append(sub_id)
            dic['subtype'].append(sub_id.split('_')[1])
            dic['daytime'].append(dt)
            dic['sleepiness'].append(sleepi_level)
            dic['percentage'].append(
                len(df_dt.sleepiness.loc[
                    (df_dt['sleepiness'] == sleepi_level)]
                    )/len(df_dt.sleepiness))

df_sleepi = pd.DataFrame.from_dict(dic)
df_sleepi.to_csv(os.path.join(
    behavpath, "NT1_CTL", "sleepiness_df.csv"
    ))

# %% Sleepiness  (full_scale, use df) [MS x ST Diff]
 
# See stats on R scripts !

data = df_sleepi.copy()

x = 'sleepiness'
order = np.linspace(1, 9, 9)
y = 'percentage'
hue = 'subtype'
hue_order = ['HS', 'N1']

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))

sns.barplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    errorbar='se', 
    orient=None, 
    palette=subtype_palette, 
    fill=False,
    hue_norm=None, 
    width=0.8, 
    dodge='auto', 
    gap=0, 
    capsize=0.05, 
    ax=ax
    )
ax.get_legend().remove()
ax.set_yticks(
    ticks = np.linspace(0, .3, 4), 
    labels = np.linspace(0, 30, 4).astype(int), 
    font = font, 
    size = 10)
ax.set_ylim(0, .3)
ax.set_ylabel("Percentage %", font = bold_font, size = 18)
ax.set_xticks(
    ticks = np.linspace(0, 8, 9), 
    labels = np.linspace(1, 9, 9).astype(int), 
    size = 14,
    font = font
    )
ax.set_xlabel("Sleepiness", font = bold_font, size = 18)
ax.tick_params(axis='both', labelsize=16)
sns.despine()

fig.tight_layout(pad = 2)
plt.savefig(
    os.path.join(behavpath, 
                 "NT1_CTL", 
                 "point_strip_sleepiness_onoff_ms_subtype.png"), 
    dpi=200 
    )

# %% Caracterize Hallu 

ms_palette = ["#FFC000", "#7030A0", "#7F7F7F", "#0070C0"]

feats = ["sleepiness", "miss", "false_alarms", "rt_go", "std_rtgo"]

minmax = {
    "sleepiness" : [1, 9],
    "miss" : [0, 100],
    "false_alarms" : [0, 100],
    "rt_go" : [.2, .8],
    "std_rtgo" : [0, .3]
    }
labels = {
    "sleepiness" : "Sleepiness",
    "miss" : "Misses (%)",
    "false_alarms" : "False Alarms (%)",
    "rt_go" : "Reaction Time (ms)",
    "std_rtgo" : "Standard Deviation RT"
    }
ticks = {
    "sleepiness" : [np.linspace(1,9,9), np.linspace(1,9,9).astype(int)],
    "miss" : [np.linspace(0,100,6), np.linspace(0,100,6).astype(int)],
    "false_alarms" : [np.linspace(0,100,6), np.linspace(0,100,6).astype(int)],
    "rt_go" : [np.linspace(.2,.8, 7), np.round(np.linspace(.2,.8,7),1)],
    "std_rtgo" : [np.linspace(0,.3, 4), np.round(np.linspace(0,.3,4),1)]
    }

data = onoff_df.copy()[[
    'sub_id', 'subtype', 'rt_go', 'rt_nogo', 'std_rtgo',
    'std_rtnogo', 'hits', 'miss', 'correct_rejections', 'false_alarms',
    'mindstate', 'sleepiness', 
    ]].groupby(
        by=['sub_id', 'subtype', 'mindstate'], 
        as_index=False
        ).mean()

hue = 'mindstate'
hue_order = ['ON', 'MW_H', 'FORGOT', 'OFF']

for feat in feats :
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (3, 3))

    y = feat
    sns.boxplot(
        data = data, 
        # x = x,
        y = y,
        hue = hue,
        # order = order,
        hue_order = hue_order,
        palette=ms_palette, 
        fill=False,
        width=0.8, 
        dodge='auto', 
        gap=.2, 
        ax=ax
        )
    sns.despine()
    
    ax.set_xticks(
        np.linspace(-0.33, 0.33, len(hue_order)),
        ['ON', 'HA', 'FGT', 'OFF'],
        font = font, 
        fontsize = 12
        )
    ax.set_xlabel("Mindstates", font=bold_font, fontsize=14)
    ax.set_yticks(
        ticks = ticks[feat][0],
        labels = ticks[feat][1],
        font = font,
        fontsize = 12
        )
    ax.set_ylim(minmax[feat][0], minmax[feat][1])
    ax.set_ylabel(labels[feat], font = bold_font, size = 14)
    
    ax.get_legend().remove()
    
    fig.tight_layout()
    
    plt.savefig(os.path.join(
        behavpath, "NT1_CTL", f"{feat}_boxplot_msonly_onoff.png"
        ), dpi=300)

# %% Compute TransMat MS

dic_stoi = {
    "ON" : 1,
    "OFF" : 2,
    "MW_H" : 3,
    "FORGOT" : 4
    }
dic_itos = {
    1 : "ON", 
    2 : "OFF", 
    3 : "HALLU", 
    4 : "FORGOT"
    }

coi = ["sub_id", "subtype", "daytime", "mindstate", 
       "proba_ON", "proba_OFF", "proba_HALLU", "proba_FORGOT"]
thisdic = {c : [] for c in coi}

for sub_id in onoff_df.sub_id.unique() :
    subid_df = onoff_df.loc[onoff_df.sub_id==sub_id]
    subtype = subid_df.subtype.unique()[0]
    
    for daytime in ["AM", "PM"]:
        subid_df_am = subid_df.loc[subid_df.daytime == daytime]
        
        if subid_df_am.empty : continue
        actual_ms = np.asarray(
            [dic_stoi[ms] for ms in subid_df_am.mindstate.values]
            )
        unique_ms = np.sort(np.unique(actual_ms))
        _, probs = transition_matrix(actual_ms)
        np_probs = probs.to_numpy()
        
        temp_transmat = np.nan * np.empty((4, 4))
        
        for i in probs.index.values :
            for k in probs.columns.values :
                temp_transmat[i-1, k-1] = probs.loc[i][k]
                
        for i, j in enumerate(temp_transmat) :
            thisdic["sub_id"].append(sub_id)
            thisdic["subtype"].append(subtype)
            thisdic["daytime"].append(daytime)
            thisdic["mindstate"].append(dic_itos[i+1])
            thisdic["proba_ON"].append(j[0])
            thisdic["proba_OFF"].append(j[1])
            thisdic["proba_HALLU"].append(j[2])
            thisdic["proba_FORGOT"].append(j[3])
            
df_transi = pd.DataFrame.from_dict(thisdic)
av_transi = df_transi[[
    'subtype', 'mindstate', 'proba_ON', 
    'proba_OFF','proba_HALLU', 'proba_FORGOT']].groupby(
        by=['subtype', 'mindstate'], as_index=False).mean()
order = ["ON", "OFF", "HALLU", "FORGOT"]
av_transi['mindstate'] = pd.Categorical(
    av_transi['mindstate'],
    categories=order,
    ordered=True
    )
av_transi = av_transi.sort_values('mindstate').reset_index(drop=True)
av_transi.set_index('mindstate', inplace=True)

transi_n1 = av_transi.loc[av_transi.subtype=="N1"]
transi_hs = av_transi.loc[av_transi.subtype=="HS"]

transi_n1.drop(columns="subtype", inplace=True)
transi_hs.drop(columns="subtype", inplace=True)

# %% Plot TransMat MS

grid_kws = {"height_ratios": (.9, .05), "hspace": .1}

f, (ax, cbar_ax) = plt.subplots(
    2, 
    gridspec_kw=grid_kws,
    figsize=(6, 6)
    )
sns.heatmap(
    transi_n1, 
    ax=ax, 
    square=False, 
    vmin=0, 
    vmax=1, 
    cbar=True,
    cbar_ax=cbar_ax, 
    cmap='Purples', 
    annot=True, 
    annot_kws={"size": 14},
    fmt='.2f',
    cbar_kws={
        "orientation": "horizontal", 
        "fraction": 0.1,
        "label": "Transition probability"}
    )
ax.set_xlabel("To Mental State", font=bold_font, fontsize=12)
ax.xaxis.tick_top()
ax.set_ylabel("From Mental State", font=bold_font, fontsize=12)
ax.xaxis.set_label_position('top')


ax.set_xticks(
    np.linspace(0.5,3.5,4),
    ['ON', 'OFF', 'HALLU', 'FORGOT'],
    font = font, 
    fontsize = 12
    )
ax.set_yticks(
    np.linspace(0.5,3.5,4),
    ['ON', 'OFF', 'HALLU', 'FORGOT'],
    font = font, 
    fontsize = 12
    )
ax.set_title("Narcolepsy Type 1", font = bold_font, fontsize = 14)
f.tight_layout()

plt.savefig(os.path.join(
    behavpath, "NT1_CTL", "transimat_ms_nt1.png"
    ), dpi=300)

f, (ax, cbar_ax) = plt.subplots(
    2, 
    gridspec_kw=grid_kws,
    figsize=(6, 6)
    )
sns.heatmap(
    transi_hs, 
    ax=ax, 
    square=False, 
    vmin=0, 
    vmax=1, 
    cbar=True,
    cbar_ax=cbar_ax, 
    cmap='Purples', 
    annot=True, 
    annot_kws={"size": 14},
    fmt='.2f',
    cbar_kws={
        "orientation": "horizontal", 
        "fraction": 0.1,
        "label": "Transition probability"}
    )
ax.set_xlabel("To Mental State", font=bold_font, fontsize=12)
ax.xaxis.tick_top()
ax.set_ylabel("From Mental State", font=bold_font, fontsize=12)
ax.xaxis.set_label_position('top')

ax.set_xticks(
    np.linspace(0.5,3.5,4),
    ['ON', 'OFF', 'HALLU', 'FORGOT'],
    font = font, 
    fontsize = 12
    )
ax.set_yticks(
    np.linspace(0.5,3.5,4),
    ['ON', 'OFF', 'HALLU', 'FORGOT'],
    font = font, 
    fontsize = 12
    )
ax.set_title("Controls", font = bold_font, fontsize = 14)
f.tight_layout()
plt.savefig(os.path.join(
    behavpath, "NT1_CTL", "transimat_ms_ctl.png"
    ), dpi=300)


# %% Compute TransMat Sleepi

coi = ["sub_id", "subtype", "daytime", "sleepiness", 
       "proba_1", "proba_2", "proba_3", "proba_4", "proba_5", 
       "proba_6", "proba_7", "proba_8", "proba_9"
       ]
thisdic = {c : [] for c in coi}

for sub_id in onoff_df.sub_id.unique() :
    subid_df = onoff_df.loc[onoff_df.sub_id==sub_id]
    subtype = subid_df.subtype.unique()[0]
    
    for daytime in ["AM", "PM"]:
        subid_df_am = subid_df.loc[subid_df.daytime == daytime]
        
        if subid_df_am.empty : continue
        
        actual_sleepi = subid_df_am.sleepiness.values
        sleepi_values = np.unique(actual_sleepi)
        
        _, probs = transition_matrix(actual_sleepi)
        np_probs = probs.to_numpy()
        
        temp_transmat = np.nan * np.empty((9, 9))
        for i in probs.index.values :
            for k in probs.columns.values :
                temp_transmat[i-1, k-1] = probs.loc[i][k]
                
        for i, j in enumerate(temp_transmat) :
            thisdic["sub_id"].append(sub_id)
            thisdic["subtype"].append(subtype)
            thisdic["daytime"].append(daytime)
            thisdic["sleepiness"].append(i+1)
            for k, sleepi in enumerate(range(1, 10)) :
                thisdic[f"proba_{sleepi}"].append(j[k])
            
df_transi_sleepi = pd.DataFrame.from_dict(thisdic)
av_transi_sleepi = df_transi_sleepi[[
    'subtype', "sleepiness", "proba_1", "proba_2", "proba_3", "proba_4",
    "proba_5",  "proba_6", "proba_7", "proba_8", "proba_9"]].groupby(
        by=['subtype', 'sleepiness'], as_index=False).mean()
        
# order = ["ON", "OFF", "HALLU", "FORGOT"]
av_transi_sleepi.sort_values(by="sleepiness").reset_index(drop=True)
av_transi_sleepi = av_transi_sleepi.sort_values('sleepiness').reset_index(drop=True)
av_transi_sleepi.set_index('sleepiness', inplace=True)

transi_sleepi_n1 = av_transi_sleepi.loc[av_transi_sleepi.subtype=="N1"]
transi_sleepi_hs = av_transi_sleepi.loc[av_transi_sleepi.subtype=="HS"]

transi_sleepi_n1.drop(columns="subtype", inplace=True)
transi_sleepi_hs.drop(columns="subtype", inplace=True)

# %% Plot TransMat Sleepi

grid_kws = {"height_ratios": (.9, .05), "hspace": .1}

f, (ax, cbar_ax) = plt.subplots(
    2, 
    gridspec_kw=grid_kws,
    figsize=(6, 6)
    )
sns.heatmap(
    transi_sleepi_n1, 
    ax=ax, 
    square=False, 
    vmin=0, 
    vmax=1, 
    cbar=True,
    cbar_ax=cbar_ax, 
    cmap='Purples', 
    annot=True, 
    annot_kws={"size": 10},
    fmt='.2f',
    cbar_kws={
        "orientation": "horizontal", 
        "fraction": 0.1,
        "label": "Transition probability"}
    )
ax.set_xlabel("To Sleepiness", font=bold_font, fontsize=12)
ax.xaxis.tick_top()
ax.set_ylabel("From Sleepiness", font=bold_font, fontsize=12)
ax.xaxis.set_label_position('top')

ax.set_xticks(
    np.linspace(0.5,8.5,9),
    np.linspace(1, 9, 9).astype(int),
    font = font, 
    fontsize = 10
    )
ax.set_yticks(
    np.linspace(0.5,8.5,9),
    np.linspace(1, 9, 9).astype(int),
    font = font, 
    fontsize = 10
    )
ax.set_title("Narcolepsy Type 1", font = bold_font, fontsize = 14)
f.tight_layout()
plt.savefig(os.path.join(
    behavpath, "NT1_CTL", "transimat_sleepiness_nt1.png"
    ), dpi=300)

f, (ax, cbar_ax) = plt.subplots(
    2, 
    gridspec_kw=grid_kws,
    figsize=(6, 6)
    )
sns.heatmap(
    transi_sleepi_hs, 
    ax=ax, 
    square=False, 
    vmin=0, 
    vmax=1, 
    cbar=True,
    cbar_ax=cbar_ax, 
    cmap='Purples', 
    annot=True, 
    annot_kws={"size": 10},
    fmt='.2f',
    cbar_kws={
        "orientation": "horizontal", 
        "fraction": 0.1,
        "label": "Transition probability"}
    )
ax.set_xlabel("To Sleepiness", font=bold_font, fontsize=12)
ax.xaxis.tick_top()
ax.set_ylabel("From Sleepiness", font=bold_font, fontsize=12)
ax.xaxis.set_label_position('top')

ax.set_xticks(
    np.linspace(0.5,8.5,9),
    np.linspace(1, 9, 9).astype(int),
    font = font, 
    fontsize = 10
    )
ax.set_yticks(
    np.linspace(0.5,8.5,9),
    np.linspace(1, 9, 9).astype(int),
    font = font, 
    fontsize = 10
    )
ax.set_title("Controls", font = bold_font, fontsize = 14)
f.tight_layout()
plt.savefig(os.path.join(
    behavpath, "NT1_CTL", "transimat_sleepiness_ctl.png"
    ), dpi=300)

# %% Compute Entropy and dKL on Transition Probabilities

# Entropy
# this_hd_entropy = [-sum(p * np.log2(p) 
#                         for p in probas if p != 0) 
#                    for probas in hypnodensity]

# dKL
# log_ratio = np.log2(hypnodensity[:-1] / hypnodensity[1:])
# aux_td_dkl = np.sum(hypnodensity[:-1] * log_ratio, axis=1)

# long_aux_td_dkl = np.append(aux_td_dkl, np.nan)

 