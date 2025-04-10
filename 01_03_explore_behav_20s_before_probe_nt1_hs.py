    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:10:05 2023

@author: arthurlecoz

01_03_explore_behav_20s_before_probe_nt1_hs.py
"""

# %% Paths
import os
import SLHIP_config_ALC as cfg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
# from highlight_text import fig_text
from matplotlib.font_manager import FontProperties

# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

rootpath = cfg.rootpath
rawpath = os.path.join(rootpath, "00_Raw")
preprocpath = os.path.join(rootpath, "01_Preproc")
behavpath = os.path.join(rootpath, "02_BehavResults")

# subtype_palette = ["#4a5759", "#bf0603", "#ffc300"]
subtype_palette = ["#8d99ae", "#d00000"]

df = pd.read_csv(f"{behavpath}/VDF_dfBEHAV_SLHIP_20sbProbe.csv")

# %% DF Manip

# sub_df = df.loc[(df.subtype != 'HI') & (df.mindstate != 'MISS')]
df = df.loc[df.subtype != 'HI']

sub_df = df.loc[(df.mindstate != 'MISS')]

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

# %% Plots no mindstates
# %% Joint

this_df = sub_df[
    ['sub_id', 'subtype','rt_go', 'rt_nogo', 'std_rtgo', 'std_rtnogo', 'hits', 'miss',
     'correct_rejections', 'false_alarms', 'sleepiness']
    ].groupby(['sub_id', 'subtype'], as_index = False).mean()

data = this_df
hue = 'subtype'
hue_order = ['HS', 'N1']

fig, ax = plt.subplots(nrows = 5, ncols = 1, figsize = (2.5, 8), sharex = True)

y = 'sleepiness'
sns.pointplot(
    data = data, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .4,
    palette = subtype_palette,
    legend = None,
    ax = ax[0],
    errorbar = "se",
    capsize = .02,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order,  
    dodge = True,
    ax = ax[0],
    palette = subtype_palette,
    legend = None,
    alpha = .3,
    size = 3
    )

ax[0].set_yticks(
    ticks = np.linspace(1, 9, 5), 
    labels = np.linspace(1, 9, 5).astype(int), 
    font = font, 
    size = 12)
ax[0].set_ylim(1, 9)
ax[0].set_ylabel("Sleepiness (KSS)", font = bold_font, size = 15)


y = 'miss'
sns.pointplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .4,
    palette = subtype_palette,
    legend = None,
    ax = ax[1],
    errorbar = "se",
    capsize = .02,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order,  
    dodge = True,
    ax = ax[1],
    palette = subtype_palette,
    legend = None,
    alpha = .3,
    size = 3
    )

ax[1].set_yticks(
    ticks = np.linspace(0, 40, 5), 
    labels =  np.linspace(0, 40, 5).astype(int), 
    font = font, 
    size = 12)
ax[1].set_ylim(0, 40)
ax[1].set_ylabel("Misses (%)", font = bold_font, size = 15)


y = "false_alarms"
sns.pointplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .4,
    palette = subtype_palette,
    legend = None,
    ax = ax[2],
    errorbar = "se",
    capsize = .02,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax[2],
    palette = subtype_palette,
    legend = None,
    alpha = .3,
    size = 3
    )
ax[2].set_ylabel("False Alarms (%)", font = bold_font, size = 15)
ax[2].set_yticks(
    ticks = np.linspace(0, 60, 4), 
    labels = np.linspace(0, 60, 4).astype(int), 
    font = font, 
    size = 12)
ax[2].set_ylim(0, 60)

y = 'rt_go'
sns.pointplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .4,
    palette = subtype_palette,
    legend = None,
    ax = ax[3],
    errorbar = "se",
    capsize = .02,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax[3],
    palette = subtype_palette,
    legend = None,
    alpha = .3,
    size = 3
    )

ax[3].set_yticks(
    ticks = np.linspace(.3, .6, 4), 
    labels = np.round(np.linspace(.3, .6, 4), 1), 
    font = font, 
    size = 12)
ax[3].set_ylim(.3, .6)
ax[3].set_ylabel("RT Go (ms)", font = bold_font, size = 15)
ax[3].set_xlabel("Subtype", font = bold_font, size = 15)
sns.despine(fig, bottom = True)

y = 'std_rtgo'
sns.pointplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .4,
    palette = subtype_palette,
    legend = None,
    ax = ax[4],
    errorbar = "se",
    capsize = .02,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax[4],
    palette = subtype_palette,
    legend = None,
    alpha = .3,
    size = 3
    )

ax[4].set_yticks(
    ticks = np.linspace(.05, .15, 3), 
    labels = np.linspace(.05, .15, 3), 
    font = font, 
    size = 12)
ax[4].set_ylim(.03, .16)
ax[4].set_ylabel("std RT Go (ms)", font = bold_font, size = 15)
ax[4].set_xlabel("Group", font = bold_font, size = 15)
sns.despine(fig, bottom = True)

for i in range(5):
    ax[i].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        )

fig.tight_layout()
plt.savefig(os.path.join(behavpath, "nt1_hs_kss_miss_fa_rt_stdrt.png"), dpi = 300)

# %% Check stats individually 

ys = ["miss", "false_alarms", "rt_go", "std_rtgo"]

for y in ys : 

    model_formula = f'{y} ~ sleepiness + C(subtype, Treatment("HS"))'
    model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'], missing = 'drop')
    model_result = model.fit()
    print(f"Statistics for {y}:\n{model_result.summary()}")

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
# plt.title('Percentage of Mindstates by Subtype', size=24, y=1.05, weight='bold', loc='left')
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
# ax.set_rlabel_position(0)  # Move radial labels to the top

# Make grid lines more discrete
ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)

# Add legend
# ax.legend(
#     title = "Subtype", 
#     frameon = False, 
#     bbox_to_anchor=(.65, .65, 0.5, 0.5), 
#     title_fontsize = 14, 
#     fontsize = 12
#     ) 

# Remove the outer circle
ax.spines['polar'].set_visible(False)

# title = 'Percentage of <Mindstates> by <Subtype>'
# fig_text(
#    0.1, .95,
#    title,
#    fontsize=15,
#    ha='left', va='center',
#    color="k", font=font,
#    highlight_textprops=[
#       {'font': bold_font},
#       {'font': bold_font},
#    ],
#    fig=fig
# )
fig.tight_layout()
# plt.title('Percentage of Mindstates by Subtype', size=24, y=1.05, weight='bold')
# plt.savefig(f"{behavpath}/radar_plot_mindstates_by_subtype.png", dpi=200)
plt.savefig(f"{behavpath}/radar_plot_mindstates_nt1_cns.png", dpi=200)
plt.show()

# %% ready fig ms %

# coi = ['sub_id', 'subtype', 'mindstate', 'percentage']
coi = ['sub_id', 'subtype', 'daytime', 'mindstate', 'percentage', 'sleepiness']
dic = {c : [] for c in coi}

for sub_id in sub_df.sub_id.unique() :    
    this_df = sub_df.loc[sub_df['sub_id'] == sub_id]
    for dt in this_df.daytime.unique() :
        df_dt = this_df.loc[sub_df['daytime'] == dt]
        for mindstate in ['ON', 'MW_I', 'MW_E', 'MW_H', 'MB', 'FORGOT'] :
            dic['sub_id'].append(sub_id)
            dic['subtype'].append(sub_id.split('_')[1])
            dic['daytime'].append(dt)
            dic['mindstate'].append(mindstate)
            dic['percentage'].append(
                len(df_dt.mindstate.loc[
                    (df_dt['mindstate'] == mindstate)]
                    )/len(df_dt.mindstate))
            dic['sleepiness'].append(
                this_df.sleepiness.loc[
                    (this_df['mindstate'] == mindstate)].mean()
                    )

df_mindstate = pd.DataFrame.from_dict(dic)

# %% fig ms % Subtype

#### Plot

# palette = ['#51b7ff','#a4abff']
# palette = ['#565B69','#0070C0']
data = df_mindstate[['sub_id', 'subtype', 'mindstate', 'percentage']].groupby(
    ['sub_id', 'subtype', 'mindstate'], as_index = False
    ).mean()
y = 'percentage'
x = "mindstate"
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']#, 'MW_E']
hue = "subtype"
hue_order = ['HS', 'N1']    
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 4))
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    errorbar = 'se',
    capsize = 0.05,
    dodge = .4,
    linestyle = 'none',
    alpha = .8,
    palette = subtype_palette,
    legend = None
    )                 
sns.stripplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    alpha = 0.2,
    dodge = True,
    legend = None,
    palette = subtype_palette
    )

# plt.legend(
#     title = "Subtype", 
#     frameon = False, 
#     bbox_to_anchor=(.5, .5, 0.5, 0.5), 
#     title_fontsize = 14, 
#     fontsize = 12
#     ) 

ax.set_ylabel('Pourcentage %', size = 24, font = bold_font)
ax.set_xlabel('Mindstate', size = 24, font = bold_font)
ax.set_ylim(0, 1)
ax.set_xticks(
    ticks = np.arange(0, 5, 1), 
    labels = ["ON", "MW", "MB", "HALLU", "FORGOT"],#, 'DISTRACTED'],
    font = font, 
    fontsize = 14)
ax.set_yticks(
    ticks = np.arange(0, 1.2, .2), 
    labels = np.arange(0, 120, 20), 
    font = font, 
    fontsize = 16)
# ax.tick_params(axis='both', labelsize=16)
sns.despine()
# title = """
# <Mindstate Percentage> by <Subtype>"""
# fig_text(
#    0.1, .93,
#    title,
#    fontsize=20,
#    ha='left', va='center',
#    color="k", font=font,
#    highlight_textprops=[
#       {'font': bold_font},
#       {'font': bold_font},
#    ],
#    fig=fig
# )
fig.tight_layout()

plt.savefig(f"{behavpath}/point_strip_per_mindstates_nt1_hs.png", dpi=200)

# %% Stats

temp_df = df_mindstate[['sub_id', 'subtype', 'mindstate', 'percentage', 'sleepiness']].groupby(
    ['sub_id', 'subtype', 'mindstate'], as_index = False
    ).mean()
temp_df = temp_df.loc[temp_df.mindstate.isin(order)]

model_formula = 'percentage ~ sleepiness + C(mindstate, Treatment("MW_H")) * C(subtype, Treatment("HS"))'
model = smf.mixedlm(
    model_formula, 
    temp_df, 
    groups=temp_df['sub_id'], 
    missing = 'drop'
    )
model_result = model.fit()
print(model_result.summary())

# %% Sleepiness [MS x ST Diff]

#### Plot 
this_df = sub_df[
    ['sub_id', 'subtype', 'mindstate', 'sleepiness']
    ].groupby(['sub_id', 'subtype', 'mindstate'], as_index = False).mean()

subtypes = ["HS", "N1"]
# colors = ['#51b7ff','#a4abff']
# colors = ['#565B69','#0070C0']
data = this_df
x = 'mindstate'
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']
y = 'sleepiness'
hue = 'subtype'
hue_order = ['HS', 'N1']

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 4))

sns.violinplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    fill = True, 
    inner = None,
    cut = .1,
    ax = ax,
    palette = subtype_palette,
    legend = None,
    # split = True,
    gap = .05,
    alpha = .2,
    linecolor = "white"
    )

sns.pointplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .4,
    palette = subtype_palette,
    errorbar = 'se',
    legend = None,
    capsize = .02,
    linestyle = 'none'
    )

sns.stripplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax,
    palette = subtype_palette,
    legend = None,
    jitter = True,
    alpha = .2
    )
ax.set_yticks(
    ticks = np.arange(1, 10, 1), 
    labels = ["Very Awake : 1", "2", "3", "4", "5", "6", "7", "8", "Falling Asleep : 9"], 
    font = font, 
    size = 12)
ax.set_ylabel("Sleepiness", font = bold_font, size = 14)
ax.set_xticks(
    ticks = np.arange(0, 5, 1), 
    labels = ["ON", "MW", "MB", 
              "HALLU", "FORGOT"], 
    size = 12,
    font = font
    )
ax.set_ylim(1, 9)
ax.tick_params(axis='both', labelsize=12)
sns.despine()
ax.set_xlabel("Mindstate", font = bold_font, size = 14)

fig.tight_layout(pad = 1)
plt.savefig(f"{behavpath}/point_strip_sleepiness_ms_nt1_hs.png", dpi=200)

# %% LME Sleepiness x MS

model_formula = 'sleepiness ~ C(mindstate, Treatment("ON")) * C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, data=this_df, groups = this_df['sub_id'])
result = model.fit()

print(result.summary())

# %% Misses & False Alarms, RT, STD RT [MSxST diff]

this_df = sub_df[
    ['sub_id', 'subtype','rt_go', 'rt_nogo', 'std_rtgo', 'hits', 'miss',
     'correct_rejections', 'false_alarms', 'mindstate', 'sleepiness']
    ].groupby(['sub_id', 'subtype', 'mindstate'], as_index = False).mean()

data = this_df
x = 'mindstate'
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']
hue = 'subtype'
hue_order = ['HS', 'N1']

fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize = (5, 8), sharex = True)

y = 'miss'
sns.pointplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .55,
    palette = subtype_palette,
    legend = None,
    ax = ax[0],
    errorbar = "se",
    capsize = .1,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order,  
    dodge = True,
    ax = ax[0],
    palette = subtype_palette,
    legend = None,
    alpha = .5,
    size = 3
    )

ax[0].set_yticks(
    ticks = np.linspace(0, 80, 5), 
    labels =  np.linspace(0, 80, 5).astype(int), 
    font = font, 
    size = 10)
ax[0].set_ylim(0, 80)
ax[0].set_ylabel("Misses (%)", font = bold_font, size = 15)


y = "false_alarms"
sns.pointplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .55,
    palette = subtype_palette,
    # fill = False,
    legend = None,
    # gap = .15,
    # width = .2,
    # showfliers = False,
    ax = ax[1],
    errorbar = "se",
    capsize = .1,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax[1],
    palette = subtype_palette,
    legend = None,
    alpha = .5,
    size = 3
    )
ax[1].set_ylabel("False Alarms (%)", font = bold_font, size = 15)
ax[1].set_yticks(
    ticks = np.linspace(0, 100, 5), 
    labels = np.linspace(0, 100, 5).astype(int), 
    font = font, 
    size = 10)
ax[1].set_ylim(0, 100)

y = 'rt_go'
sns.pointplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .55,
    palette = subtype_palette,
    # fill = False,
    legend = None,
    # gap = .15,
    # width = .2,
    # showfliers = False,
    ax = ax[2],
    errorbar = "se",
    capsize = .1,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax[2],
    palette = subtype_palette,
    legend = None,
    alpha = .5,
    size = 3
    )

ax[2].set_yticks(
    ticks = np.linspace(0.3, 0.8, 6), 
    labels = np.round(np.linspace(0.3, 0.8, 6), 1), 
    font = font, 
    size = 10)
ax[2].set_ylabel("RT Go (ms)", font = bold_font, size = 15)


y = 'std_rtgo'
sns.pointplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .55,
    palette = subtype_palette,
    # fill = False,
    legend = None,
    # gap = .15,
    # width = .2,
    # showfliers = False,
    ax = ax[3],
    errorbar = "se",
    capsize = .1,
    ls = "none",
    alpha = .8
    )

sns.stripplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax[3],
    palette = subtype_palette,
    legend = None,
    alpha = .5,
    size = 3
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
    labels = ["ON", "MW", "MB", "Hallu/Illu", "Forgot"], 
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
    
fig.tight_layout(pad=1)

plt.savefig(f"{behavpath}/miss_fa_rt_subtype_n1.png", dpi=200)

# %% Stats

feat_oi = "miss"
ms_oi = "MW_I"

df_stats = this_df.loc[this_df.subtype != 'HI']

model_formula = f'{feat_oi} ~ sleepiness + C(mindstate, Treatment("{ms_oi}")) * C(subtype, Treatment("HS"))'
model = smf.mixedlm(
    model_formula, 
    df_stats, 
    groups=df_stats['sub_id'], 
    missing = 'drop'
    )
model_result = model.fit()
print(model_result.summary())

# %% 

this_df = sub_df[
    ['sub_id', 'subtype','rt_go', 'rt_nogo', 'hits', 'miss',
     'correct_rejections', 'false_alarms', 'mindstate', 'sleepiness']
    ].groupby(['sub_id', 'subtype', 'mindstate'], as_index = False).mean()

model_formula = 'miss ~ sleepiness + C(mindstate, Treatment("MB")) * C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, data=this_df, groups = this_df['sub_id'])
result = model.fit(method='bfgs')

print(result.summary())

# %% 

model_formula = 'percentage ~ C(mindstate, Treatment("MW_H")) * C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, data=df_mindstate, groups = df_mindstate['sub_id'])
result = model.fit(method='bfgs')
print(result.summary())

# %% Explore Dynamics (DAY)

this_df = sub_df.copy()

foi = 'false_alarms'

fig, ax = plt.subplots(nrows = 1, ncols = 1)
sns.pointplot(
    data = sub_df,
    x = 'total_block',
    y = foi,
    hue = 'subtype',
    hue_order = ['HS', 'N1'],
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
    # HI=["#ffb703", "#ffe4a0"],
    )

foi = 'sleepiness'

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5), sharex=True, sharey=True)
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


# %% Inspection HI (court/long)

plt.figure(); 
sns.pointplot(
    data=sub_df.loc[sub_df.subtype=='HI'], 
    y='sleepiness', 
    x='sub_id', 
    linestyle='none', 
    errorbar='se', 
    capsize=.05
    )
plt.figure(); 
sns.pointplot(
    data=sub_df.loc[sub_df.subtype=='HI'], 
    y='false_alarms', 
    x='sub_id', 
    linestyle='none', 
    errorbar='se', 
    capsize=.05
    )
plt.figure(); 
sns.pointplot(
    data=sub_df.loc[sub_df.subtype=='HI'], 
    y='miss', 
    x='sub_id', 
    linestyle='none', 
    errorbar='se', 
    capsize=.05
    )
plt.figure(); 
sns.pointplot(
    data=sub_df.loc[sub_df.subtype=='HI'], 
    y='rt_go', 
    x='sub_id', 
    linestyle='none', 
    errorbar='se', 
    capsize=.05
    )
