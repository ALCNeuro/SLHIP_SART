#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/07/25

@author: arthurlecoz

09_01_compute_burst_density_NT1_CTL.py
"""
# %%% Paths & Packages

import SLHIP_config_ALC as config
import mne
from glob import glob
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat

cleanDataPath = config.cleanDataPath
burstPath = config.burstPath
rootpath = config.rootpath

path_data=os.path.join(rootpath, '00_Raw')
behavpath = os.path.join(rootpath, "02_BehavResults")
feature_path = os.path.join(burstPath, "features")
figs_path = os.path.join(burstPath, "figs")

bursts_files = glob(os.path.join(burstPath, "bursts_detected", "Bursts_*.csv"))

ms_dic = {
    0 : "MISS",
    1 : 'ON',
    2 : 'MW',
    3 : 'DISTRACTED',
    4 : 'HALLU',
    5 : 'MB',
    6 : 'FORGOT'
    }

probe_col = [
    "nprobe","t_probe_th","t_probe_act","nblock","block_cond","ntrial",
    "PQ1_respkey","PQ2_respkey","PQ3_respkey",
    "PQ1_resptime","PQ2_resptime","PQ3_resptime",
    "PQ1_questime","PQ2_questime","PQ3_questime",
    "PQ1_respval","PQ2_respval","PQ3_respval"
    ]

features = [
    "sub_id", "subtype", "daytime", "burst_type", 
    "density", "mean_cycle_count", "mean_duration_points", "mean_amplitude",
    "mean_pos_period", "mean_neg_period",
    "channel", "n_probe", 
    'mindstate','voluntary', 'sleepiness',
    'rt_go', 'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
    ]

df_behavior = pd.read_csv(f"{behavpath}/VDF_dfBEHAV_SLHIP_20sbProbe.csv")
del df_behavior['Unnamed: 0']

channels = np.array(config.eeg_channels)

fs = 256
window_sec = 10
window_samp = fs * window_sec

redo = 1

# %% Alpha & Theta

df_featlist = []

for i_file, file in enumerate(bursts_files) :
    sub_dic = {feat : [] for feat in features}
    sub_id = file.split('detected/')[1].split('_', 1)[1].split("_raw")[0]
    subtype = sub_id[:2]
    session = sub_id [-2:]
    sub_id = sub_id[:-3]
    
    this_df_savepath = os.path.join(
        burstPath, 
        "features", 
        f"{sub_id}_{session}.csv"
        )
    
    if os.path.exists(this_df_savepath) and not redo :
        print(f"\n[{i_file+1} / {len(bursts_files)}]... Loading {sub_id}'s Burst Features")  
        df_feature = pd.read_csv(this_df_savepath)
        del df_feature['Unnamed: 0']
        df_featlist.append(df_feature)
    
    else :
        print(f"\n[{i_file+1} / {len(bursts_files)}]... Computing {sub_id}'s Burst Features")  
        
        df_bursts = pd.read_csv(file)
        df_bursts.insert(
            4,
            "Act_Band",
            ["Theta" if freq <= 8 else "Alpha" for freq in df_bursts.BurstFrequency.values]
            )
        # del df_bursts['Unnamed: 0']
        
        behav_paths = glob(os.path.join(
            path_data, "experiment", f"*{sub_id}*", "*.mat"
            ))
        raw_file = glob(os.path.join(
            cleanDataPath, 
            "raw_icaed",
            f"*{sub_id}*{session}*raw.fif"
            ))[0]
        raw = mne.io.read_raw(raw_file, preload = False)
        
        events, event_id = mne.events_from_annotations(raw)
        target_id = event_id['Stimulus/S  3']
        timing_probes = events[events[:, 2] == target_id, 0]/raw.info['sfreq']
        t_probes_samp = timing_probes * fs
        
        #### Extract Behav Infos
        if len(behav_paths) < 1 :
            print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
            continue
        
        sub_behav = df_behavior.loc[
            (df_behavior.sub_id == f"sub_{sub_id}")
            & (df_behavior.daytime == session)
            ]
    
        if sub_id == 'HS_007' : behav_path = behav_paths[0]
        else :
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
        
        if sub_id == "HI_002" and session == "AM" :
            ms_answers = ms_answers[1:]
            vol_answers = vol_answers[1:]
            sleepi_answers = sleepi_answers[1:]
            sub_behav = sub_behav.iloc[1:]
        if sub_id == "N1_009" and session == "AM" :
            ms_answers = ms_answers[13:]
            vol_answers = vol_answers[13:]
            sleepi_answers = sleepi_answers[13:]
            sub_behav = sub_behav.iloc[13:]
        if sub_id == "N1_015" and session == "PM" :
            ms_answers = ms_answers[:30]
            vol_answers = vol_answers[:30]
            sleepi_answers = sleepi_answers[:30]
            sub_behav = sub_behav.iloc[:30]
        if sub_id == "N1_012" and session == "AM" :
            ms_answers = ms_answers[:37]
            vol_answers = vol_answers[:37]
            sleepi_answers = sleepi_answers[:37]
            sub_behav = sub_behav.iloc[:37]
        
        print(f"""\nIn {sub_id} file, were found :
            * {timing_probes.shape[0]} Probes (first question)
            -> {ms_answers.shape[0]} MS Answers
            -> {vol_answers.shape[0]} Voluntary Answers
            -> {sleepi_answers.shape[0]} Sleepiness answers""")
            
        if not len(timing_probes) == len(ms_answers) : 
            print(f"!!!\n{sub_id} : Careful, inconsistencies found between EEG and Behav\n!!!")
            input("\nWaiting for further instructions...")
        
        for burst_type in ['Alpha', 'Theta'] :
            for n_probe, t in enumerate(t_probes_samp) :
                for channel in channels :
                    sub_bursts = df_bursts.loc[
                        (df_bursts.Act_Band == burst_type)
                        & (df_bursts.ChannelIndexLabel == channel)
                        ]
                    win_start = t - window_samp
                    win_end   = t
                    mask = (sub_bursts['End']   >= win_start) & \
                           (sub_bursts['Start'] <= win_end)
                    these_bursts = sub_bursts[mask]
                    
                    sub_dic['burst_type'].append(burst_type)
                    sub_dic['density'].append(these_bursts.shape[0]/window_sec)
                    sub_dic['mean_cycle_count'].append(np.nanmean(
                        these_bursts["CyclesCount"]
                        ))
                    sub_dic['mean_duration_points'].append(np.nanmean(
                        these_bursts["DurationPoints"]
                        ))
                    sub_dic['mean_amplitude'].append(np.nanmean(
                        these_bursts["MeanAmplitude"]
                        ))
                    sub_dic['mean_pos_period'].append(np.nanmean(
                        these_bursts["MeanPeriodPos"]
                        ))
                    sub_dic['mean_neg_period'].append(np.nanmean(
                        these_bursts["MeanPeriodNeg"]
                        ))
                    
                    sub_dic['sub_id'].append(sub_id)
                    sub_dic['subtype'].append(subtype)
                    sub_dic['daytime'].append(session)
                    
                    sub_dic['channel'].append(channel)
                    sub_dic['n_probe'].append(n_probe)
                    
                    mindstate = ms_answers[n_probe]
                    voluntary = vol_answers[n_probe]
                    sleepiness = sleepi_answers[n_probe]
                    sub_dic['mindstate'].append(mindstate)
                    sub_dic['voluntary'].append(voluntary)
                    sub_dic['sleepiness'].append(sleepiness)
                    
                    sub_dic['rt_go'].append(sub_behav.rt_go.iloc[n_probe])
                    sub_dic['rt_nogo'].append(sub_behav.rt_nogo.iloc[n_probe])
                    sub_dic['hits'].append(sub_behav.hits.iloc[n_probe])
                    sub_dic['miss'].append(sub_behav.miss.iloc[n_probe])
                    sub_dic['correct_rejections'].append(
                        sub_behav.correct_rejections.iloc[n_probe]
                        )
                    sub_dic['false_alarms'].append(
                        sub_behav.false_alarms.iloc[n_probe]
                        )
            
        df_feature = pd.DataFrame.from_dict(sub_dic)
        df_feature.to_csv(this_df_savepath)
        df_featlist.append(df_feature)
    
df = pd.concat(df_featlist)
df.to_csv(os.path.join(
    burstPath, 
    "features", 
    "All_Bursts_features.csv"
    ))

# %% Slow / Fast [4-6-8-10-12]

df_featlist = []
burst_types = ["slow_theta", "fast_theta", "slow_alpha", "fast_alpha"]

for i_file, file in enumerate(bursts_files) :
    sub_dic = {feat : [] for feat in features}
    sub_id = file.split('detected/')[1].split('_', 1)[1].split("_raw")[0]
    subtype = sub_id[:2]
    session = sub_id [-2:]
    sub_id = sub_id[:-3]
    
    this_df_savepath = os.path.join(
        burstPath, 
        "features", 
        f"{sub_id}_{session}_slow_fast.csv"
        )
    
    if os.path.exists(this_df_savepath) and not redo :
        print(f"\n[{i_file+1} / {len(bursts_files)}]... Loading {sub_id}'s Burst Features")  
        df_feature = pd.read_csv(this_df_savepath)
        del df_feature['Unnamed: 0']
        df_featlist.append(df_feature)
    
    else :
        print(f"\n[{i_file+1} / {len(bursts_files)}]... Computing {sub_id}'s Burst Features")  
        
        df_bursts = pd.read_csv(file)
        
        labels = [
            "slow_theta" if 4 <= f <= 6 
            else "fast_theta" if 6 <  f <= 8 
            else "slow_alpha" if 8 <  f <=10 
            else "fast_alpha" if 10 < f <=12 
            else None
            for f in df_bursts.BurstFrequency
            ]
        
        df_bursts.insert(
            4,
            "Act_Band",
            labels
            )
        # del df_bursts['Unnamed: 0']
        
        behav_paths = glob(os.path.join(
            path_data, "experiment", f"*{sub_id}*", "*.mat"
            ))
        raw_file = glob(os.path.join(
            cleanDataPath, 
            "raw_icaed",
            f"*{sub_id}*{session}*raw.fif"
            ))[0]
        raw = mne.io.read_raw(raw_file, preload = False)
        
        events, event_id = mne.events_from_annotations(raw)
        target_id = event_id['Stimulus/S  3']
        timing_probes = events[events[:, 2] == target_id, 0]/raw.info['sfreq']
        t_probes_samp = timing_probes * fs
        
        #### Extract Behav Infos
        if len(behav_paths) < 1 :
            print(f"\nNo behav_path found for {sub_id}... Look into it! Skipping for now...")
            continue
        
        sub_behav = df_behavior.loc[
            (df_behavior.sub_id == f"sub_{sub_id}")
            & (df_behavior.daytime == session)
            ]
    
        if sub_id == 'HS_007' : behav_path = behav_paths[0]
        else :
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
        
        if sub_id == "HI_002" and session == "AM" :
            ms_answers = ms_answers[1:]
            vol_answers = vol_answers[1:]
            sleepi_answers = sleepi_answers[1:]
            sub_behav = sub_behav.iloc[1:]
        if sub_id == "N1_009" and session == "AM" :
            ms_answers = ms_answers[13:]
            vol_answers = vol_answers[13:]
            sleepi_answers = sleepi_answers[13:]
            sub_behav = sub_behav.iloc[13:]
        if sub_id == "N1_015" and session == "PM" :
            ms_answers = ms_answers[:30]
            vol_answers = vol_answers[:30]
            sleepi_answers = sleepi_answers[:30]
            sub_behav = sub_behav.iloc[:30]
        if sub_id == "N1_012" and session == "AM" :
            ms_answers = ms_answers[:37]
            vol_answers = vol_answers[:37]
            sleepi_answers = sleepi_answers[:37]
            sub_behav = sub_behav.iloc[:37]
        
        print(f"""\nIn {sub_id} file, were found :
            * {timing_probes.shape[0]} Probes (first question)
            -> {ms_answers.shape[0]} MS Answers
            -> {vol_answers.shape[0]} Voluntary Answers
            -> {sleepi_answers.shape[0]} Sleepiness answers""")
            
        if not len(timing_probes) == len(ms_answers) : 
            print(f"!!!\n{sub_id} : Careful, inconsistencies found between EEG and Behav\n!!!")
            input("\nWaiting for further instructions...")
        
        for burst_type in burst_types :
            for n_probe, t in enumerate(t_probes_samp) :
                for channel in channels :
                    sub_bursts = df_bursts.loc[
                        (df_bursts.Act_Band == burst_type)
                        & (df_bursts.ChannelIndexLabel == channel)
                        ]
                    win_start = t - window_samp
                    win_end   = t
                    mask = (sub_bursts['End']   >= win_start) & \
                           (sub_bursts['Start'] <= win_end)
                    these_bursts = sub_bursts[mask]
                    
                    sub_dic['burst_type'].append(burst_type)
                    sub_dic['density'].append(these_bursts.shape[0]/window_sec)
                    sub_dic['mean_cycle_count'].append(np.nanmean(
                        these_bursts["CyclesCount"]
                        ))
                    sub_dic['mean_duration_points'].append(np.nanmean(
                        these_bursts["DurationPoints"]
                        ))
                    sub_dic['mean_amplitude'].append(np.nanmean(
                        these_bursts["MeanAmplitude"]
                        ))
                    sub_dic['mean_pos_period'].append(np.nanmean(
                        these_bursts["MeanPeriodPos"]
                        ))
                    sub_dic['mean_neg_period'].append(np.nanmean(
                        these_bursts["MeanPeriodNeg"]
                        ))
                    
                    sub_dic['sub_id'].append(sub_id)
                    sub_dic['subtype'].append(subtype)
                    sub_dic['daytime'].append(session)
                    
                    sub_dic['channel'].append(channel)
                    sub_dic['n_probe'].append(n_probe)
                    
                    mindstate = ms_answers[n_probe]
                    voluntary = vol_answers[n_probe]
                    sleepiness = sleepi_answers[n_probe]
                    sub_dic['mindstate'].append(mindstate)
                    sub_dic['voluntary'].append(voluntary)
                    sub_dic['sleepiness'].append(sleepiness)
                    
                    sub_dic['rt_go'].append(sub_behav.rt_go.iloc[n_probe])
                    sub_dic['rt_nogo'].append(sub_behav.rt_nogo.iloc[n_probe])
                    sub_dic['hits'].append(sub_behav.hits.iloc[n_probe])
                    sub_dic['miss'].append(sub_behav.miss.iloc[n_probe])
                    sub_dic['correct_rejections'].append(
                        sub_behav.correct_rejections.iloc[n_probe]
                        )
                    sub_dic['false_alarms'].append(
                        sub_behav.false_alarms.iloc[n_probe]
                        )
            
        df_feature = pd.DataFrame.from_dict(sub_dic)
        df_feature.to_csv(this_df_savepath)
        df_featlist.append(df_feature)
    
df = pd.concat(df_featlist)
df.to_csv(os.path.join(
    burstPath, 
    "features", 
    "All_Bursts_features_slow_fast.csv"
    ))
