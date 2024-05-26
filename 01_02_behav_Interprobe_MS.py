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
from scipy.io import loadmat
from glob import glob

import statsmodels.formula.api as smf

rootpath = cfg.rootpath
rawpath = os.path.join(rootpath, "00_Raw")
preprocpath = os.path.join(rootpath, "01_Preproc")

taskepochspath = os.path.join(preprocpath, "epochs_epochs")
probespath = os.path.join(preprocpath, "raws_probes")
restingspath = os.path.join(preprocpath, "raws_restings")
autorejectpath = os.path.join(preprocpath, "autoreject")
icapath = os.path.join(preprocpath, "ica_files")
figpath = os.path.join(preprocpath, "figs")


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

# %% functions

def filter_behav(mat, nblock):
    """ Filter data by nblock and further by 'rt' conditions. """
    return mat.loc[(mat["nblock"] == nblock) & ((mat['rt'] > 0.15) | (mat['rt'].isna()))]
def filter_probe(mat, nblock):
    """ Filter data by nblock and further by 'rt' conditions. """
    return mat.loc[(mat["nblock"] == nblock)]

# def get_interprobe_mat(block_mat, ntrial_list, current_trial):
#     """ Get data slices between ntrial intervals. """
#     if current_trial == 0:
#         return block_mat[block_mat['ntrial'] < ntrial_list[current_trial]]
#     else:
#         return block_mat[
#             (block_mat['ntrial'] > ntrial_list[current_trial-1]) 
#             & (block_mat['ntrial'] < ntrial_list[current_trial])
#             ]

def append_to_lists(list_dict, **kwargs):
    """ Append multiple values to respective lists efficiently. """
    for key, value in kwargs.items():
        list_dict[key].append(value)

def calculate_behavioral_metrics(interprobe_mat):
    """ Calculate behavioral metrics such as hits, misses, correct rejections, and false alarms. """
    # interprobe_mat_20s = pd.concat([
    #     interprobe_mat[interprobe_mat['digit'] != 3].iloc[-8:],
    #     interprobe_mat[interprobe_mat['digit'] == 3].iloc[-2:]
    # ])
    go_trials = interprobe_mat[interprobe_mat['digit'] != 3]
    nogo_trials = interprobe_mat[interprobe_mat['digit'] == 3]
    hits = 100 * len(go_trials[go_trials['corr_go'] == 1]) / len(go_trials)
    miss = 100 * (1 - hits / 100)
    cr = 100 * len(nogo_trials[nogo_trials['corr_nogo'] == 1]) / len(nogo_trials)
    fa = 100 * (1 - cr / 100)
    rtgo = go_trials['rt'].mean()
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
# df.to_csv(f"{figpath}/VDF_dfBEHAV_SLHIP_20sbProbe.csv")

# %% smf

sub_df = df.loc[(df.subtype != 'HI') & (df.mindstate != 'MISS')]
sub_df.mindstate.replace('FORGOT', 'MB', inplace = True)

model_formula = 'false_alarms ~ subtype*daytime'
model = smf.mixedlm(model_formula, sub_df, groups=sub_df['sub_id'], missing = 'omit')
model_result = model.fit()
model_result.summary()

# %% RadarPlot for percentage of mindstates per subtype

# Calculate percentages
mindstate_counts = sub_df.groupby(['subtype', 'mindstate']).size().unstack(fill_value=0)
mindstate_percentages = mindstate_counts.div(mindstate_counts.sum(axis=1), axis=0) * 100

# Create radar plot for each subtype
subtypes = ['HS', 'N1']
kind_thought = list(sub_df.mindstate.unique())
full_thought = ["ON", "MW Internal", "Distracted", "MB", "Hallucination/Illusion"]
# kind_thought.remove('MISS')  # Remove 'MISS' from the radar plot

max_value = mindstate_percentages.max().max()

# Radar plot settings
colors = {'HS': '#51b7ff', 'N1': '#a4abff'}
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 10))

for subtype in subtypes:
    values = mindstate_percentages.loc[subtype, kind_thought].values.tolist()
    values += values[:1]  # to close the radar plot

    angles = np.linspace(0, 2 * np.pi, len(kind_thought), endpoint=False).tolist()
    angles += angles[:1]  # to close the radar plot

    ax.fill(angles, values, color=colors[subtype], alpha=0.25)
    ax.plot(angles, values, color=colors[subtype], linewidth=2, label=subtype)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(full_thought)
ax.set_ylim(0, max_value)
ax.set_yticks(np.linspace(0, max_value, 5))
ax.set_yticklabels([f'{int(tick)}%' for tick in np.linspace(0, max_value, 5)], color='grey')
ax.yaxis.set_ticks_position('left')
ax.set_rlabel_position(0)  # Move radial labels to the top

# Make grid lines more discrete
ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)

# Add legend
ax.legend()

# Remove the outer circle
ax.spines['polar'].set_visible(False)

plt.title('Percentage of Mindstates by Subtype', size=24, y=1.05)
plt.savefig(f"{figpath}/radar_plot_mindstates_by_subtype.png", dpi=500)
plt.show()
