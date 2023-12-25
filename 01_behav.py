#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:10:05 2023

@author: arthurlecoz

behav_explore.py

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
"""
# %% Paths
import os
import config as cfg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from glob import glob

behavData = cfg.behavDataPath
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
    6 : "FORGOT"
    }

# %% 

# go_correct = round(100 * np.nanmean(behav_result.get('test_res')[:,11]), 3)
# gomissed = round((100 - go_correct), 3)
# nogo_correct = round(100 * np.nanmean(behav_result.get('test_res')[:,10]), 3)
# falsealarms = round((100 - nogo_correct), 3)
# temps_reac_s = round(np.nanmean(
#     behav_result.get('test_res')[
#         np.isnan(behav_result.get('test_res')[:,11]), 9
#         ] - behav_result.get('test_res')[
#             np.isnan(behav_result.get('test_res')[:,11]), 7]
#             ), 
#             4
#             )

# print(f"Number of Go Correct is: {go_correct} %\n" +
#       f"  ->   Thus, Number of Go Missed: {gomissed} %\n" +
#       f"Number of Nogo Correct is {nogo_correct} %\n" +
#       f"  ->   Thus, Number of false alarms is {falsealarms} %\n" +
#       f"Time reaction was {temps_reac_s}s")
   
# plt.figure(figsize = (12, 8))
# plt.title("Évolution de la fatigue\n au cours des essais\n")
# plt.plot(behav_result.get('probe_res')[:,17], color = 'k')
# plt.xlabel("nTrial")
# plt.ylim((0,10))
# plt.yticks(ticks = np.arange(1,11))
# plt.ylabel("Échelle de fatigue")    

# %% 

sub_id = cfg.pilot_id
probelist = []
testlist  = []

for session in ["AM", "PM"] :
    temp_results = loadmat(glob(os.path.join(
        behavData, "matlab", f"*{sub_id}*{session}*.mat"))[0]
        )
    
    df_test = pd.DataFrame(
            temp_results['test_res'], 
            columns = test_col)
    df_test['rt'] = df_test["resp_time"] - df_test["stim_onset"]
    
    df_probe = pd.DataFrame(
        temp_results['probe_res'], 
        columns = probe_col)
    
    df_test['session_type'] = [session for i in range(len(df_test))]
    df_probe['session_type'] = [session for i in range(len(df_probe))]
    
    testlist.append(df_test)
    probelist.append(df_probe)

probe = pd.concat(probelist)
probe["mindstate"] = [probe_int_str[value] for value in probe.PQ1_respval]

per_ms = {mindstate : 
          probe.mindstate[probe.mindstate == mindstate
                          ].shape[0]/probe.mindstate.shape[0] 
          for mindstate in probe.mindstate.unique()}

import seaborn as sns

ax = sns.barplot(x = per_ms.keys(), y = per_ms.values())
for p in ax.patches:
    ax.text(
        p.get_x() + p.get_width()/2., 
        p.get_height()+.02, 
        '{:0.3f}'.format(p.get_height()), 
        fontsize=12, 
        color='black', 
        ha='center', 
        va='bottom'
        )
plt.show()

features = ["session", "mindstate", "percentage"]
thisdic = {feature : [] for feature in features}
for session in probe.session_type.unique():
    for mindstate in probe.mindstate.unique() :
        thisdic["session"].append(session)
        thisdic["mindstate"].append(mindstate)
        thisdic["percentage"].append(
            probe.loc[
                (probe.session_type == session) 
                & (probe.mindstate == mindstate)
                ].shape[0] / probe.loc[
                    (probe.session_type == session)
                    ].shape[0]
            )

df = pd.DataFrame.from_dict(thisdic)

sns.pointplot(
    df, x = 'session', y = 'percentage', hue = 'mindstate', dodge = True
    )
