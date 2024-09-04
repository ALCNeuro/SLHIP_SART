    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:10:05 2023

@author: arthurlecoz

behav_explore.py
"""
# %% Description
"""
- test_res
Col 1: block number
Col 2: block condition (always 2)
Col 3: image set (always 3)
Col 4: trial number
Col 5: digit displayed
Col 6: nogo digit
Col 7: response key
Col 8: stimulus onset (in seconds, PTB time)
Col 9: duration presentation (in seconds)
Col 10: response time (in seconds, PTB time)
Col 11: correctness on nogo trial
Col 12: correctness on go trial
    
- probe_res (résultats des probes, 1 ligne = 1 probe)

Col 1: probe number
Col 2: probe time  (in seconds, PTB time, theoretical)
Col 3: probe time  (in seconds, PTB time, actual)
Col 4: block number
Col 5: block condition
Col 6: trial number
Col 7: Probe Question 1 - Response key
Col 8: Probe Question 2 - Response key 
Col 9: Probe Question 3 - Response key 
Col 10: Probe Question 1 - Response time
Col 11: Probe Question 2 - Response time 
Col 12: Probe Question 3 - Response time 
Col 13: Probe Question 1 - Question time
Col 14: Probe Question 2 - Question time
Col 15: Probe Question 3 - Question time
Col 16: Probe Question 1 - Response value
Col 17: Probe Question 2 - Response value
Col 18: Probe Question 3 - Response value

Probe Q1 : Etat d''esprit juste avant l'interruption.
Ans :   1 - J'étais concentré-e sur la tâche 
        2 - J'étais distrait·e par quelque chose d''interne (pensée, image, etc.)
        3 - J'étais distrait·e par quelque chose d''externe (environnement)
        4 - J'étais distrait·e par quelque chose de fictif (illusion, hallucination)
        5 - Je ne pensais à rien 
        6 - Je ne me souviens pas

Probe Q2 : Avez-vous volontairement contrôlé sur quoi se portait votre attention ?
Ans :   1 - Oui
        2 - Non

Probe Q3 : Notez votre vigilance
Ans : 1 - 9 

---

Informations to use to clean the data :
    1. Reaction Time
    2. Amount of miss in a row
    3. Eye tracker eyes closed

The amount of miss and eye tracker eyes closed should communicate

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
from highlight_text import fig_text
from matplotlib.font_manager import FontProperties
from scipy.io import loadmat
from glob import glob

# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
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

subtype_palette = ["#4a5759", "#bf0603", "#ffc300"]

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
    cr = 100 * len(nogo_trials[nogo_trials['corr_nogo'] == 1]) / len(nogo_trials)
    fa = 100 * (1 - cr / 100)
    rtgo = go_corr['rt'].mean()
    rtnogo = nogo_trials['rt'].mean()
    return hits, miss, cr, fa, rtgo, rtnogo


# %% 

columns = ['sub_id', 'subtype', 'nblock', 'rt_go', 'rt_nogo', 'hits', 
           'miss', 'correct_rejections', 'false_alarms', 'mindstate', 
           'voluntary', 'sleepiness', 'daytime']
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
                hits, miss, cr, fa, rtgo, rtnogo = calculate_behavioral_metrics(interprobe_mat)
                append_to_lists(
                    data_dict, 
                    sub_id=sub_id, 
                    subtype=subtype, 
                    nblock=nblock, 
                    rt_go=rtgo, 
                    rt_nogo=rtnogo, 
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
df.to_csv(f"{behavpath}/VDF_dfBEHAV_SLHIP_20sbProbe.csv")

# %% DF Manip

# sub_df = df.loc[(df.subtype != 'HI') & (df.mindstate != 'MISS')]
sub_df = df.loc[(df.mindstate != 'MISS')]

this_df = sub_df[['sub_id', 'subtype','rt_go', 'rt_nogo', 'hits', 'miss',
       'correct_rejections', 'false_alarms', 'mindstate', 
       'sleepiness', 'daytime']].groupby(
           ['sub_id', 'subtype', 'mindstate', 'daytime'], 
           as_index = False).mean()
           
hs_df = this_df.loc[
    (this_df.subtype == 'HS') 
    & (this_df.mindstate != 'MW_E')
    & (this_df.mindstate != 'FORGOT')
    & (this_df.mindstate != 'MW_H')
    ]

# %% Plots no subtypes
#### FA

data = this_df
y = 'false_alarms'
x = "mindstate"
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT', 'MW_E']
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))

sns.stripplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    alpha = 0.5,
    dodge = True,
    legend = None,
    color = 'lightgrey'
    )
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    errorbar = 'se',
    capsize = 0.05,
    linestyle = 'none',
    color = 'grey'
    )         
sns.violinplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    fill = True,
    alpha = 0.2,
    dodge = True,
    linecolor = 'white',
    inner = None,
    legend = None,
    color = 'lightgrey',
    cut = .2
    )         

ax.set_ylabel(y, font = bold_font, fontsize = 16)
ax.set_xlabel(x, font = bold_font, fontsize = 16)
ax.set_xticks(
    np.linspace(0,5,6), 
    ["ON", "MW", "MB", "HALLU", "FORGOT", "DISTRACTED"], 
    font = font, 
    fontsize = 10
    )
ax.set_ylim(0, 100)
sns.despine(bottom=True)
fig.tight_layout()
plt.savefig(f"{behavpath}/{y}_MS.png", dpi=200)

model_formula = 'false_alarms ~ C(mindstate, Treatment("ON"))'
model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'], missing = 'drop')
model_result = model.fit()
print(f"Statistics for {y}:\n{model_result.summary()}")

#### SLEEPINESS

y = 'sleepiness'
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))

sns.stripplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    alpha = 0.5,
    dodge = True,
    legend = None,
    color = 'lightgrey'
    )
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    errorbar = 'se',
    capsize = 0.05,
    linestyle = 'none',
    color = 'grey'
    )             
sns.violinplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    fill = True,
    alpha = 0.2,
    dodge = True,
    linecolor = 'white',
    inner = None,
    legend = None,
    color = 'lightgrey',
    cut = .2
    )    

ax.set_ylabel(y, font = bold_font, fontsize = 16)
ax.set_xlabel(x, font = bold_font, fontsize = 16)
ax.set_xticks(
    np.linspace(0,5,6), 
    ["ON", "MW", "MB", "HALLU", "FORGOT", "DISTRACTED"], 
    font = font, 
    fontsize = 10
    )
ax.set_ylim(1, 9)
sns.despine(bottom=True)
fig.tight_layout()
plt.savefig(f"{behavpath}/{y}_MS.png", dpi=200)

model_formula = f'{y} ~ C(mindstate, Treatment("ON"))'
model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'], missing = 'drop')
model_result = model.fit()
print(f"Statistics for {y}:\n{model_result.summary()}")

#### MISS

y = 'miss'
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))

sns.stripplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    alpha = 0.5,
    dodge = True,
    legend = None,
    color = 'lightgrey'
    )
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    errorbar = 'se',
    capsize = 0.05,
    linestyle = 'none',
    color = 'grey'
    )             

ax.set_ylabel(y, font = bold_font, fontsize = 16)
ax.set_xlabel(x, font = bold_font, fontsize = 16)
ax.set_xticks(
    np.linspace(0,5,6), 
    ["ON", "MW", "MB", "HALLU", "FORGOT", "DISTRACTED"], 
    font = font, 
    fontsize = 10
    )
ax.set_ylim(-1, 70)
sns.despine(bottom=True)
fig.tight_layout()
plt.savefig(f"{behavpath}/{y}_MS.png", dpi=200)

model_formula = f'{y} ~ C(mindstate, Treatment("ON"))'
model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'], missing = 'drop')
model_result = model.fit()
print(f"Statistics for {y}:\n{model_result.summary()}")

# %% Plots no mindstates
#### FA

st_df = sub_df[['sub_id', 'subtype','rt_go', 'rt_nogo', 'hits', 'miss',
       'correct_rejections', 'false_alarms', 
       'sleepiness', ]].groupby(
           ['sub_id', 'subtype'], 
           as_index = False).mean()

data = st_df
y = 'false_alarms'
x = "subtype"
order = ['HS', 'N1', 'HI']
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 5))

sns.stripplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    alpha = 0.5,
    dodge = True,
    legend = None,
    palette = subtype_palette
    )
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    dodge = .55,
    errorbar = 'se',
    capsize = 0.05,
    linestyle = 'none',
    palette = subtype_palette
    )         
sns.violinplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    fill = True,
    alpha = 0.2,
    dodge = True,
    linecolor = 'white',
    inner = None,
    legend = None,
    palette = subtype_palette,
    )         

ax.set_ylabel(y, font = bold_font, fontsize = 16)
ax.set_xlabel(x, font = bold_font, fontsize = 16)
ax.set_xticks(
    np.linspace(0,2,3), 
    ["HS", "N1", "HI"], 
    font = font, 
    fontsize = 16
    )
ax.set_yticks(
    np.linspace(0,100,5), 
    np.linspace(0,100,5), 
    font = font, 
    fontsize = 12
    )
ax.set_ylim(0, 100)
sns.despine(bottom=True)
fig.tight_layout()
plt.savefig(f"{behavpath}/{y}_subtype.png", dpi=200)

model_formula = f'{y} ~ C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, st_df, groups=st_df['sub_id'], missing = 'drop')
model_result = model.fit()
print(f"Statistics for {y}:\n{model_result.summary()}")

#### SLEEPINESS

y = 'sleepiness'

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 5))

sns.stripplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    alpha = 0.5,
    dodge = True,
    legend = None,
    palette = subtype_palette,
    )
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    dodge = .55,
    errorbar = 'se',
    capsize = 0.05,
    linestyle = 'none',
    palette = subtype_palette
    )             
sns.violinplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    fill = True,
    alpha = 0.2,
    dodge = True,
    linecolor = 'white',
    inner = None,
    legend = None,
    palette = subtype_palette,
    cut = .2
    )    

ax.set_ylabel(y, font = bold_font, fontsize = 16)
ax.set_xlabel(x, font = bold_font, fontsize = 16)
ax.set_xticks(
    np.linspace(0,2,3), 
    ["HS", "N1", "HI"],
    font = font, 
    fontsize = 14
    )
ax.set_yticks(
    np.linspace(1,9,9), 
    np.linspace(1,9,9).astype(int), 
    font = font, 
    fontsize = 12
    )
ax.set_ylim(1, 9)
sns.despine(bottom=True)
fig.tight_layout()
plt.savefig(f"{behavpath}/{y}_subtype.png", dpi=200)

model_formula = f'{y} ~ C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, st_df, groups=st_df['sub_id'], missing = 'drop')
model_result = model.fit()
print(f"Statistics for {y}:\n{model_result.summary()}")

#### MISS

y = 'miss'
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (3, 5))

sns.stripplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    alpha = 0.5,
    dodge = True,
    legend = None,
    palette = subtype_palette,
    )
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    errorbar = 'se',
    capsize = 0.05,
    dodge = .55,
    linestyle = 'none',
    palette = subtype_palette
    )             

ax.set_ylabel(y, font = bold_font, fontsize = 16)
ax.set_xlabel(x, font = bold_font, fontsize = 16)
ax.set_xticks(
    np.linspace(0,2,3), 
    ["HS", "N1", "HI"],
    font = font, 
    fontsize = 10
    )
ax.set_ylim(-1, 75)
sns.despine(bottom=True)
fig.tight_layout()
plt.savefig(f"{behavpath}/{y}_subtype.png", dpi=200)

model_formula = f'{y} ~ C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, st_df, groups=st_df['sub_id'], missing = 'drop')
model_result = model.fit()
print(f"Statistics for {y}:\n{model_result.summary()}")

# %% 
# %% 

model_formula = 'sleepiness ~ C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'], missing = 'drop')
model_result = model.fit()
print(model_result.summary())

# %% RadarPlot % MS / Subtype

# Calculate percentages
mindstate_counts = sub_df.groupby(['subtype', 'mindstate']).size().unstack(fill_value=0)
mindstate_percentages = mindstate_counts.div(mindstate_counts.sum(axis=1), axis=0) * 100

# Create radar plot for each subtype
subtypes = ['HS', 'N1', 'HI']
kind_thought = list(sub_df.mindstate.unique())
full_thought = ["ON", "MW Internal", "Distracted", "MB", "Forgor", "Hallucination/Illusion"]
# kind_thought.remove('MISS')  # Remove 'MISS' from the radar plot

max_value = mindstate_percentages.max().max()

# Radar plot settings
colors = {'HS': '#417AB2', 'N1': '#EF8E3D', 'HI' : '#539F41'}

fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(7, 7), dpi=150)
# plt.title('Percentage of Mindstates by Subtype', size=24, y=1.05, weight='bold', loc='left')
for subtype in subtypes:
    values = mindstate_percentages.loc[subtype, kind_thought].values.tolist()
    values += values[:1]  # to close the radar plot

    angles = np.linspace(0, 2 * np.pi, len(kind_thought), endpoint=False).tolist()
    angles += angles[:1]  # to close the radar plot

    ax.fill(angles, values, color=colors[subtype], alpha=0.25)
    ax.plot(angles, values, color=colors[subtype], linewidth=2, label=subtype)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(full_thought, font = font)
ax.set_ylim(0, 100)
ax.set_yticks(np.linspace(0, 100, 5))
ax.set_yticklabels(
    [f'{int(tick)}%' for tick in np.linspace(0, 100, 5)], 
    color='grey',
    font = font)
ax.yaxis.set_ticks_position('left')
# ax.set_rlabel_position(0)  # Move radial labels to the top

# Make grid lines more discrete
ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)

# Add legend
ax.legend(
    title = "Subtype", 
    frameon = False, 
    bbox_to_anchor=(.65, .65, 0.5, 0.5), 
    title_fontsize = 14, 
    fontsize = 12
    ) 

# Remove the outer circle
ax.spines['polar'].set_visible(False)

title = 'Percentage of <Mindstates> by <Subtype>'
fig_text(
   0.1, .95,
   title,
   fontsize=15,
   ha='left', va='center',
   color="k", font=font,
   highlight_textprops=[
      {'font': bold_font},
      {'font': bold_font},
   ],
   fig=fig
)
fig.tight_layout()
# plt.title('Percentage of Mindstates by Subtype', size=24, y=1.05, weight='bold')
plt.savefig(f"{behavpath}/radar_plot_mindstates_by_subtype.png", dpi=200)
plt.show()

# %% ready fig ms %

coi = ['sub_id', 'subtype', 'mindstate', 'percentage']
# coi = ['sub_id', 'subtype', 'daytime', 'mindstate', 'percentage']
dic = {c : [] for c in coi}

for sub_id in sub_df.sub_id.unique() :    
    this_df = sub_df.loc[sub_df['sub_id'] == sub_id]
    # for dt in this_df.daytime.unique() :
    #     df_dt = this_df.loc[sub_df['daytime'] == dt]
    for mindstate in ['ON', 'MW_I', 'MW_E', 'MW_H', 'MB', 'FORGOT'] :
        dic['sub_id'].append(sub_id)
        dic['subtype'].append(sub_id.split('_')[1])
        # dic['daytime'].append(dt)
        dic['mindstate'].append(mindstate)
        dic['percentage'].append(
            len(this_df.mindstate.loc[
                (this_df['mindstate'] == mindstate)]
                )/len(this_df.mindstate))

df_mindstate = pd.DataFrame.from_dict(dic)

# %% fig ms % No Subtype

# palette = ['#51b7ff','#a4abff']
# palette = ['#565B69','#0070C0']
data = df_mindstate
y = 'percentage'
x = "mindstate"
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT', 'MW_E']
# hue = "subtype"
# hue_order = ['HS', 'N1']    
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    errorbar = 'se',
    capsize = 0.05,
    linestyle = 'none',
    color = 'grey'
    )                
sns.stripplot(
    data = data, 
    x = x,
    y = y,
    order = order,
    alpha = 0.5,
    dodge = True,
    legend = None,
    color = 'lightgrey'
    )

# plt.legend(
#     title = "Subtype", 
#     frameon = False, 
#     bbox_to_anchor=(.5, .5, 0.5, 0.5), 
#     title_fontsize = 14, 
#     fontsize = 12
#     ) 

ax.set_ylabel('Percentage %', size = 18, font = bold_font)
ax.set_xlabel('Mindstate', size = 18, font = bold_font)
ax.set_ylim(0, 1)
ax.set_xticks(
    ticks = np.arange(0, 6, 1), 
    labels = ["ON", "MW", "MB", "HALLU", "FORGOT", 'DISTRACTED'],
    font = font, 
    fontsize = 10)
ax.set_yticks(
    ticks = np.arange(0, 1.2, .2), 
    labels = np.arange(0, 120, 20), 
    font = font, fontsize = 12)
sns.despine()
title = """<Mindstate Percentage> according to the <Mindstate>"""
fig_text(
   0.05, .93,
   title,
   fontsize=16,
   ha='left', va='center',
   color="k", font=font,
   highlight_textprops=[
      {'font': bold_font},
      {'font': bold_font},
   ],
   fig=fig
)

plt.savefig(f"{behavpath}/point_strip_per_mindstates.png", dpi=200)

#### Stats

model_formula = 'percentage ~ C(mindstate, Treatment("ON"))'
model = smf.mixedlm(
    model_formula, 
    df_mindstate, 
    groups=df_mindstate['sub_id'], 
    missing = 'drop'
    )
model_result = model.fit()
print(model_result.summary())

# %% fig ms % Subtype

#### Plot

# palette = ['#51b7ff','#a4abff']
# palette = ['#565B69','#0070C0']
data = df_mindstate
y = 'percentage'
x = "mindstate"
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT', 'MW_E']
hue = "subtype"
hue_order = ['HS', 'N1', 'HI']    
         
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 12))
sns.pointplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    errorbar = 'se',
    capsize = 0.05,
    dodge = .55,
    linestyle = 'none',
    alpha = .8
    # palette = palette,
    )         
# sns.violinplot(
#     data = data, 
#     x = x,
#     y = y,
#     hue = hue,
#     order = order,
#     hue_order = hue_order,
#     fill = True,
#     alpha = 0.2,
#     dodge = True,
#     linecolor = 'white',
#     inner = None,
#     legend = None,
#     palette = palette,
#     cut = .5
#     )         
sns.stripplot(
    data = data, 
    x = x,
    y = y,
    hue = hue,
    order = order,
    hue_order = hue_order,
    alpha = 0.5,
    dodge = True,
    legend = None,
    # palette = palette
    )

plt.legend(
    title = "Subtype", 
    frameon = False, 
    bbox_to_anchor=(.5, .5, 0.5, 0.5), 
    title_fontsize = 14, 
    fontsize = 12
    ) 

ax.set_ylabel('Pourcentage %', size = 18, font = bold_font)
ax.set_xlabel('Mindstate', size = 18, font = bold_font)
ax.set_ylim(0, 1)
ax.set_xticks(
    ticks = np.arange(0, 6, 1), 
    labels = ["ON", "MW", "MB", "HALLU", "FORGOT", 'DISTRACTED'],
    font = font, 
    fontsize = 14)
ax.set_yticks(
    ticks = np.arange(0, 1.2, .2), 
    labels = np.arange(0, 120, 20), 
    font = font, 
    fontsize = 16)
# ax.tick_params(axis='both', labelsize=16)
sns.despine()
title = """
<Mindstate Percentage> according to the <Subtype>"""
fig_text(
   0.1, .93,
   title,
   fontsize=20,
   ha='left', va='center',
   color="k", font=font,
   highlight_textprops=[
      {'font': bold_font},
      {'font': bold_font},
   ],
   fig=fig
)

plt.savefig(f"{behavpath}/point_strip_per_mindstates_by_subtype.png", dpi=200)

#### Stats

model_formula = 'percentage ~ C(mindstate, Treatment("MW_E"))*C(subtype, Treatment("HS"))'
model = smf.mixedlm(
    model_formula, 
    df_mindstate, 
    groups=df_mindstate['sub_id'], 
    missing = 'drop'
    )
model_result = model.fit()
print(model_result.summary())

# %% Sleepiness

#### Plot 
this_df = sub_df[
    ['sub_id', 'subtype', 'mindstate', 'sleepiness']
    ].groupby(['sub_id', 'subtype', 'mindstate'], as_index = False).mean()

subtypes = ["HS", "N1", "HI"]
# colors = ['#51b7ff','#a4abff']
# colors = ['#565B69','#0070C0']
data = this_df
x = 'mindstate'
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']
y = 'sleepiness'
hue = 'subtype'
hue_order = ['HS', 'N1', 'HI']

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8))

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
    # palette = colors,
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
    dodge = .55,
    # palette = colors,
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
    # palette = colors,
    legend = True,
    jitter = True,
    alpha = .5
    )
ax.set_yticks(
    ticks = np.arange(1, 10, 1), 
    labels = ["Very Awake : 1", "2", "3", "4", "5", "6", "7", "8", "Falling Asleep : 9"], 
    font = font, 
    size = 10)
ax.set_ylabel("Sleepiness", font = bold_font, size = 18)
ax.set_xticks(
    ticks = np.arange(0, 5, 1), 
    labels = ["Focused", "Mind Wandering", "Mind Blanking", 
              "Hallucination/Illusion", "Forgot"], 
    size = 20,
    font = font
    )
ax.set_ylim(1, 9)
ax.tick_params(axis='both', labelsize=16)
sns.despine()
ax.set_xlabel("Mindstate", font = bold_font, size = 18)
title = """Subjective <Sleepiness> rating according to the <Mindstates>, by <Subtype>"""
fig_text(
   0.1, .94,
   title,
   fontsize=20,
   ha='left', va='center',
   color="k", font=font,
   highlight_textprops=[
      {'font': bold_font},
      {'font': bold_font},
      {'font': bold_font},
      # {'color': colors[0], 'font': bold_font},
      # {'color': colors[1], 'font': bold_font}
   ],
   fig=fig
)
fig.tight_layout(pad = 2)
plt.savefig(f"{behavpath}/point_strip_sleepiness_ms_subtype.png", dpi=200)

#### Stats

# %% LME Sleepiness

# sub_df['sleepiness'] = pd.Categorical(sub_df['sleepiness'], ordered=True)

model_formula = 'rt_go ~ C(subtype, Treatment("HS"))'
model = smf.mixedlm(model_formula, data=this_df, groups = this_df['sub_id'])
result = model.fit(method='bfgs')

print(result.summary())

# %% Misses & False Alarms

this_df = sub_df[
    ['sub_id', 'subtype','rt_go', 'rt_nogo', 'hits', 'miss',
     'correct_rejections', 'false_alarms', 'mindstate', 'sleepiness']
    ].groupby(['sub_id', 'subtype', 'mindstate'], as_index = False).mean()

data = this_df
x = 'mindstate'
order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']
hue = 'subtype'
hue_order = ['HS', 'N1', 'HI']
# colors = ['#51b7ff','#a4abff']
# colors = ['#565B69','#0070C0']

fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (7, 12), sharex = True)

y = 'miss'
sns.pointplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .55,
    # palette = colors,
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
    # palette = colors,
    legend = None,
    alpha = .5,
    size = 3
    )

# ax[0].set_yticks(
#     ticks = np.arange(0, 120, 20), 
#     labels =  np.arange(0, 120, 20), 
#     font = font, 
#     size = 10)
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
    palette = colors,
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
    palette = colors,
    legend = None,
    alpha = .5,
    size = 3
    )
ax[1].set_ylabel("False Alarms (%)", font = bold_font, size = 15)
# ax[1].set_yticks(
#     ticks = np.arange(0, 150, 50), 
#     labels =  np.arange(0, 150, 50), 
#     font = font, 
#     size = 10)

y = 'rt_go'
sns.pointplot(
    data = data,
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .55,
    palette = colors,
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
    palette = colors,
    legend = None,
    alpha = .5,
    size = 3
    )

# ax[1].set_yticks(
#     ticks = np.arange(0, 150, 50), 
#     labels =  np.arange(0, 150, 50), 
#     font = font, 
#     size = 10)
ax[2].set_ylabel("Reaction Time Go (ms)", font = bold_font, size = 15)
ax[2].set_xticks(
    ticks = np.arange(0, 5, 1), 
    labels = ["ON", "MW", "MB", "Hallu/Illu", "Forgot"], 
    size = 20,
    font = font
    )
ax[2].set_xlabel("Mindstate", font = bold_font, size = 15)
sns.despine(fig)

title = """
<Misses>, <False Alarms>, and <Reaction Time>\naccording to the <Mindstates> and <Subtype>
"""
fig_text(
   0.07, .94,
   title,
   fontsize=15,
   ha='left', va='center',
   color="k", font=font,
   highlight_textprops=[
      {'font': bold_font},
      {'font': bold_font},
      {'font': bold_font},
      {'font': bold_font},
      {'font': bold_font},
      # {'color': colors[0], 'font': bold_font},
      # {'color': colors[1], 'font': bold_font}
   ],
   fig=fig
)
plt.savefig(f"{behavpath}/miss_fa_rt_subtype.png", dpi=200)

# %% M & FA - Subtype only

this_df = sub_df[
    ['sub_id', 'subtype','rt_go', 'rt_nogo', 'hits', 'miss',
     'correct_rejections', 'false_alarms']
    ].groupby(['sub_id', 'subtype'], as_index = False).mean()

data = this_df
# x = 'mindstate'
# order = ['ON', 'MW_I', 'MB', 'MW_H', 'FORGOT']
hue = 'subtype'
hue_order = ['HS', 'N1']
# colors = ['#51b7ff','#a4abff']
colors = ['#565B69','#0070C0']

fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (4, 8), sharex = True)

y = 'miss'
sns.pointplot(
    data = data,
    # x = x, 
    # order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .15,
    palette = colors,
    legend = None,
    ax = ax[0],
    errorbar = "se",
    capsize = .05,
    ls = "none"
    )

sns.stripplot(
    data = data,
    # x = x, 
    # order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order,  
    dodge = True,
    ax = ax[0],
    palette = colors,
    legend = None,
    alpha = .5,
    size = 3
    )

# ax[0].set_yticks(
#     ticks = np.arange(0, 120, 20), 
#     labels =  np.arange(0, 120, 20), 
#     font = font, 
#     size = 10)
ax[0].set_ylabel("Misses (%)", font = bold_font, size = 15)


y = "false_alarms"
sns.pointplot(
    data = data,
    # x = x, 
    # order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .15,
    palette = colors,
    # fill = False,
    legend = None,
    # gap = .15,
    # width = .2,
    # showfliers = False,
    ax = ax[1],
    errorbar = "se",
    capsize = .05,
    ls = "none"
    )

sns.stripplot(
    data = data,
    # x = x, 
    # order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax[1],
    palette = colors,
    legend = None,
    alpha = .5,
    size = 3
    )
ax[1].set_ylabel("False Alarms (%)", font = bold_font, size = 15)
# ax[1].set_yticks(
#     ticks = np.arange(0, 150, 50), 
#     labels =  np.arange(0, 150, 50), 
#     font = font, 
#     size = 10)

y = 'rt_go'
sns.pointplot(
    data = data,
    # x = x, 
    # order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = .15,
    palette = colors,
    # fill = False,
    legend = None,
    # gap = .15,
    # width = .2,
    # showfliers = False,
    ax = ax[2],
    errorbar = "se",
    capsize = .05,
    ls = "none"
    )

sns.stripplot(
    data = data,
    # x = x, 
    # order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order, 
    dodge = True,
    ax = ax[2],
    palette = colors,
    legend = None,
    alpha = .5,
    size = 3
    )

# ax[1].set_yticks(
#     ticks = np.arange(0, 150, 50), 
#     labels =  np.arange(0, 150, 50), 
#     font = font, 
#     size = 10)
ax[2].set_ylabel("Reaction Time Go (ms)", font = bold_font, size = 15)
ax[2].set_xticks(
    ticks = [-0.10, 0.10], 
    labels = ["CTL", "NT1"], 
    size = 20,
    font = font
    )
ax[2].set_xlabel("Subtype", font = bold_font, size = 15)
sns.despine(fig)
fig.subplots_adjust(left=0.2, right=0.90, top=0.90, bottom=0.1)

title = """<Misses>, <FA>, and <RT>
according to the <Subtype> : [ <CTL>, <NT1> ]
"""
fig_text(
   0.07, .94,
   title,
   fontsize=15,
   ha='left', va='center',
   color="k", font=font,
   highlight_textprops=[
      {'font': bold_font},
      {'font': bold_font},
      {'font': bold_font},
      {'font': bold_font},
      {'color': colors[0], 'font': bold_font},
      {'color': colors[1], 'font': bold_font}
   ],
   fig=fig
)

plt.savefig(f"{behavpath}/miss_fa_rt_subtype_noms.png", dpi=200)

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

