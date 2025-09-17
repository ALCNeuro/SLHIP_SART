#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:50:28 2024

@author: arthurlecoz

04_03_compute_swdensity_v2.py

Need to 
-> Compute density based on thresholds :
    - 20µV
    - Healthy Subjects : average across channels
    - Heatlhy Subjects : per channels
    
-> Then add the behavior before each probe
=> To correlate SW with behavior

"""
# %%% Paths & Packages

import SLHIP_config_ALC as config
import mne
from glob import glob
import pandas as pd
import numpy as np
from scipy.stats import exponnorm, zscore
import matplotlib.pyplot as plt
import os

cleanDataPath = config.cleanDataPath
wavesPath = config.wavesPath
rootpath = config.rootpath

behavpath = os.path.join(rootpath, "02_BehavResults")
slowwaves_path = os.path.join(wavesPath, "slow_waves")
allwaves_files = glob(os.path.join(wavesPath, "all_waves", "*.csv"))

hs_allwaves_files = glob(os.path.join(wavesPath, "all_waves", "HS*.csv"))
reports_path = os.path.join(wavesPath, "reports")
figs_path = os.path.join(wavesPath, "figs")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

slope_range = [0.25, 2] # in uV/ms
positive_amp = [75] # in uV
amplitude_max = 150

inspect = 0

features = [
    "sub_id", "subtype", "daytime",
    "density_20", "density_p_90","density_p_80", "density_90hs", "density_80hs", 
    "ptp_20", "ptp_p_90", "ptp_p_80", "ptp_90hs", "ptp_80hs", 
    "uslope_20", "uslope_p_90", "uslope_p_80", "uslope_90hs", "uslope_80hs", 
    "dslope_20", "dslope_p_90", "dslope_p_80", "dslope_90hs", "dslope_80hs", 
    "frequency_20", "frequency_p_90", "frequency_p_80", "frequency_90hs", "frequency_80hs", 
    "channel", "n_epoch", "nblock", 
    "nprobe", 'mindstate','voluntary', 'sleepiness',
    'rt_go', 'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
    ]

df_behavior = pd.read_csv(f"{behavpath}/VDF_dfBEHAV_SLHIP_20sbProbe.csv")
del df_behavior['Unnamed: 0']

channels = np.array(config.eeg_channels)

len_epoch = 10


# %% Compute HS Thresh

this_df_thresh_savepath = os.path.join(
    wavesPath, 'features', 'df_threshold_healthy.csv'
    )

if os.path.exists(this_df_thresh_savepath) :
    big_thresh = pd.read_csv(this_df_thresh_savepath)
else : 

    thresh_feats = ["sub_id", "session", "channel", "threshold_90", "threshold_80"]
    
    df_thresh_list = []
    
    for i_file, file in enumerate(hs_allwaves_files) :
        thresh_dic = {feat : [] for feat in thresh_feats}
        sub_id = file.split('all_waves/')[1].split('.')[0]
        subtype = sub_id[:2]
        session = sub_id[-2:]
        
        this_df_swfeat_savepath = os.path.join(
            wavesPath, 
            "features", 
            f"{sub_id}_{1/slope_range[1]}_{round(1/slope_range[0])}.csv"
            )
    
        print(f"\n...Processing {sub_id}... Computing SW Features")  
        
        df_aw = pd.read_csv(file)
        del df_aw['Unnamed: 0']
        
        df_aw = df_aw.loc[
            (df_aw['PTP'] < 150) 
            & (df_aw['PTP'] > 5)
            & (df_aw['pos_amp_peak'] < 75)]
        df_aw = df_aw.loc[
            (df_aw["pos_halfway_period"] <= slope_range[1])
            & (df_aw["pos_halfway_period"] >= slope_range[0])
            ]
            
        for chan in channels :
            temp_df = df_aw.loc[df_aw["channel"] == chan]
            
            thresh_90_chan = np.percentile(temp_df.PTP.values, 90)
            thresh_80_chan = np.percentile(temp_df.PTP.values, 80)
            
            thresh_dic['sub_id'].append(sub_id)
            thresh_dic['session'].append(session)
            thresh_dic['channel'].append(chan)
            thresh_dic['threshold_90'].append(thresh_90_chan)
            thresh_dic['threshold_80'].append(thresh_80_chan)
             
        df_thresh = pd.DataFrame.from_dict(thresh_dic)
        # df_thresh.to_csv("")
        df_thresh_list.append(df_thresh)
        
    big_thresh = pd.concat(df_thresh_list)
    big_thresh.to_csv(this_df_thresh_savepath)
    
# %% Compute density (delta~theta)

df_chanthresh = big_thresh[
    ['channel','threshold_90', 'threshold_80']
    ].groupby('channel', as_index = False).mean()

df_featlist = []

for i_file, file in enumerate(allwaves_files) :
    sub_dic = {feat : [] for feat in features}
    sub_id = file.split('all_waves/')[1].split('.')[0]
    subtype = sub_id[:2]
    session = sub_id [-2:]
    sub_id = sub_id[:-3]
    
    this_df_swfeat_savepath = os.path.join(
        wavesPath, 
        "features", 
        f"{sub_id}_{session}_{1/slope_range[1]}_{round(1/slope_range[0])}.csv"
        )

    print(f"\n...Processing {sub_id}... Computing SW Features")  
    df_aw = pd.read_csv(file)
    del df_aw['Unnamed: 0']
    
    df_aw = df_aw.loc[
        (df_aw['PTP'] < 150) 
        & (df_aw['PTP'] > 5)
        & (df_aw['pos_amp_peak'] < 75)]
    df_aw = df_aw.loc[
        (df_aw["pos_halfway_period"] <= slope_range[1])
        & (df_aw["pos_halfway_period"] >= slope_range[0])
        ]
    
    thresholds_90 = {
        c:np.percentile(df_aw.PTP.loc[df_aw.channel==c], 90) 
        for c in channels}
    thresholds_80 = {
        c:np.percentile(df_aw.PTP.loc[df_aw.channel==c], 80) 
        for c in channels}
    
    epochs_files = glob(os.path.join(
        cleanDataPath, 
        "epochs_probes",
        f"*{sub_id}*{session}*epo.fif"
        ))[0]
    epochs = mne.read_epochs(epochs_files, preload = False)
    metadata = epochs.metadata
    del epochs
    
    epochs_observed = int(df_aw.tot_epochs.unique()[0])
    
    sub_behav = df_behavior.loc[
        (df_behavior.sub_id == f"sub_{sub_id}")
        & (df_behavior.daytime == session)
        ]

    for n_epoch in range(epochs_observed) :
        if sub_id == 'HI_002' and session == "AM" and n_epoch == 0:
            continue
        sub_df = df_aw.loc[df_aw.n_epoch == n_epoch]
        
        if sub_df.empty :
            mindstate = metadata.iloc[n_epoch].mindstate
            voluntary = metadata.iloc[n_epoch].voluntary
            sleepiness = metadata.iloc[n_epoch].sleepiness
            nblock = metadata.iloc[n_epoch].nblock
            nprobe = metadata.iloc[n_epoch].nprobe
        
        else : 
            mindstate = sub_df.mindstate.iloc[0]
            voluntary = sub_df.voluntary.iloc[0]
            sleepiness = sub_df.sleepiness.iloc[0]
            nblock = sub_df.nblock.iloc[0]
            nprobe = sub_df.nprobe.iloc[0]
        
        for chan in channels :
            
            temp_df = sub_df.loc[sub_df["channel"] == chan]
        
            df_20 = temp_df.loc[temp_df.PTP > 20]
            df_p_90 = temp_df.loc[temp_df.PTP > thresholds_90[chan]]
            df_p_80 = temp_df.loc[temp_df.PTP > thresholds_80[chan]]
            df_hs_90 = temp_df.loc[temp_df.PTP > df_chanthresh.loc[
                df_chanthresh.channel == chan].threshold_90.iloc[0]
                ]
            df_hs_80 = temp_df.loc[temp_df.PTP > df_chanthresh.loc[
                df_chanthresh.channel == chan].threshold_80.iloc[0]
                ]
            
            sub_dic['density_20'].append(df_20.shape[0]/len_epoch)
            sub_dic['density_p_90'].append(df_p_90.shape[0]/len_epoch)
            sub_dic['density_p_80'].append(df_p_80.shape[0]/len_epoch)
            sub_dic['density_90hs'].append(df_hs_90.shape[0]/len_epoch)
            sub_dic['density_80hs'].append(df_hs_80.shape[0]/len_epoch)
            
            sub_dic['sub_id'].append(sub_id)
            sub_dic['subtype'].append(subtype)
            sub_dic['daytime'].append(session)
            
            sub_dic['channel'].append(chan)
            sub_dic['n_epoch'].append(n_epoch)
            sub_dic['nblock'].append(nblock)
            
            sub_dic['nprobe'].append(nprobe)
            sub_dic['mindstate'].append(mindstate)
            sub_dic['voluntary'].append(voluntary)
            sub_dic['sleepiness'].append(sleepiness)
            
            sub_dic['rt_go'].append(sub_behav.rt_go.iloc[n_epoch])
            sub_dic['rt_nogo'].append(sub_behav.rt_nogo.iloc[n_epoch])
            sub_dic['hits'].append(sub_behav.hits.iloc[n_epoch])
            sub_dic['miss'].append(sub_behav.miss.iloc[n_epoch])
            sub_dic['correct_rejections'].append(sub_behav.correct_rejections.iloc[n_epoch])
            sub_dic['false_alarms'].append(sub_behav.false_alarms.iloc[n_epoch])
            
            if not df_20.shape[0] :
                sub_dic['ptp_20'].append(np.nan)
                sub_dic['uslope_20'].append(np.nan)
                sub_dic['dslope_20'].append(np.nan)
                sub_dic['frequency_20'].append(np.nan)
            else : 
                sub_dic['ptp_20'].append(np.nanmean(df_20.PTP))
                sub_dic['uslope_20'].append(np.nanmean(
                    df_20.max_pos_slope_2nd_segment
                    ))
                sub_dic['dslope_20'].append(np.nanmean(
                    df_20.inst_neg_1st_segment_slope
                    ))
                sub_dic['frequency_20'].append(np.nanmean(
                    1/df_20.pos_halfway_period
                    ))
                
            if not df_p_90.shape[0] :
                sub_dic['ptp_p_90'].append(np.nan)
                sub_dic['uslope_p_90'].append(np.nan)
                sub_dic['dslope_p_90'].append(np.nan)
                sub_dic['frequency_p_90'].append(np.nan)
            else :    
                sub_dic['ptp_p_90'].append(np.nanmean(
                    df_p_90.PTP
                    ))
                sub_dic['uslope_p_90'].append(np.nanmean(
                    df_p_90.max_pos_slope_2nd_segment
                    ))
                sub_dic['dslope_p_90'].append(np.nanmean(
                    df_p_90.inst_neg_1st_segment_slope
                    ))
                sub_dic['frequency_p_90'].append(np.nanmean(
                    1/df_p_90.pos_halfway_period
                    ))
                
            if not df_p_80.shape[0]:
                sub_dic['ptp_p_80'].append(np.nan)
                sub_dic['uslope_p_80'].append(np.nan)
                sub_dic['dslope_p_80'].append(np.nan)
                sub_dic['frequency_p_80'].append(np.nan)
            else :
                sub_dic['ptp_p_80'].append(np.nanmean(
                    df_p_80.PTP
                    ))
                sub_dic['uslope_p_80'].append(np.nanmean(
                    df_p_80.max_pos_slope_2nd_segment
                    ))
                sub_dic['dslope_p_80'].append(np.nanmean(
                    df_p_80.inst_neg_1st_segment_slope
                    ))
                sub_dic['frequency_p_80'].append(np.nanmean(
                    1/df_p_80.pos_halfway_period
                    ))
                
            if not df_hs_90.shape[0]:
                sub_dic['ptp_90hs'].append(np.nan)
                sub_dic['uslope_90hs'].append(np.nan)
                sub_dic['dslope_90hs'].append(np.nan)
                sub_dic['frequency_90hs'].append(np.nan)
            else :
                sub_dic['ptp_90hs'].append(np.nanmean(
                    df_hs_90.PTP
                    ))
                sub_dic['uslope_90hs'].append(np.nanmean(
                    df_hs_90.max_pos_slope_2nd_segment
                    ))
                sub_dic['dslope_90hs'].append(np.nanmean(
                    df_hs_90.inst_neg_1st_segment_slope
                    ))
                sub_dic['frequency_90hs'].append(np.nanmean(
                    1/df_hs_90.pos_halfway_period
                    ))
                
            if not df_hs_80.shape[0]:
                sub_dic['ptp_80hs'].append(np.nan)
                sub_dic['uslope_80hs'].append(np.nan)
                sub_dic['dslope_80hs'].append(np.nan)
                sub_dic['frequency_80hs'].append(np.nan)
            else :
                sub_dic['ptp_80hs'].append(np.nanmean(
                    df_hs_80.PTP
                    ))
                sub_dic['uslope_80hs'].append(np.nanmean(
                    df_hs_80.max_pos_slope_2nd_segment
                    ))
                sub_dic['dslope_80hs'].append(np.nanmean(
                    df_hs_80.inst_neg_1st_segment_slope
                    ))
                sub_dic['frequency_80hs'].append(np.nanmean(
                    1/df_hs_80.pos_halfway_period
                    ))
            
    df_feature = pd.DataFrame.from_dict(sub_dic)
    df_feature.to_csv(this_df_swfeat_savepath)
    df_featlist.append(df_feature)
    
df = pd.concat(df_featlist)
df.to_csv(os.path.join(
    wavesPath, 
    "features", 
    f"all_SW_features_{1/slope_range[1]}_{round(1/slope_range[0])}.csv"
    ))

# %% Compute density (delta // theta)

features = [
    "sub_id", "subtype", "freq_range",
    "density_20", "density_90_hs_global", "density_90_hs_chan",
    "ptp_20", "ptp_90_hs_global", "ptp_90_hs_chan",
    "uslope_20", "uslope_90_hs_global", "uslope_90_hs_chan",
    "dslope_20", "dslope_90_hs_global", "dslope_90_hs_chan",
    "frequency_20", "frequency_90_hs_global", "frequency_90_hs_chan",
    "channel", "n_epoch", "nblock", 
    "nprobe", 'mindstate','voluntary', 'sleepiness',
    'rt_go', 'rt_nogo', 'hits', 'miss', 'correct_rejections', 'false_alarms'
    ]
# ["delta" if 1/slope <= 4 else "theta" for slope in df_aw.

df_chanthresh = big_thresh[
    ['channel','threshold_90_chan']].groupby('channel', as_index = False).mean()

global_thresh = big_thresh.threshold_90_global.mean()

df_featlist = []

for i_file, file in enumerate(allwaves_files) :
    sub_dic = {feat : [] for feat in features}
    sub_id = file.split('all_waves/')[1].split('.')[0]
    subtype = sub_id[:2]
    session = sub_id [-2:]
    sub_id = sub_id[:-3]
    
    this_df_swfeat_savepath = os.path.join(
        wavesPath, 
        "features", 
        f"{sub_id}_{session}_delta_theta.csv"
        )

    print(f"\n...Processing {sub_id}... Computing SW Features")  
    df_aw = pd.read_csv(file)
    del df_aw['Unnamed: 0']
    
    df_aw = df_aw.loc[
        (df_aw['PTP'] < 150) 
        & (df_aw['PTP'] > 5)
        & (df_aw['pos_amp_peak'] < 75)]
    df_aw = df_aw.loc[
        (df_aw["pos_halfway_period"] <= slope_range[1])
        & (df_aw["pos_halfway_period"] >= slope_range[0])
        ]
    
    df_aw['freq_range'] = ["delta" if 1/slope <= 4 else "theta" 
                           for slope in df_aw.pos_halfway_period]
    
    epochs_files = glob(os.path.join(
        cleanDataPath, 
        "epochs_probes",
        f"*{sub_id}*{session}*epo.fif"
        ))[0]
    epochs = mne.read_epochs(epochs_files, preload = False)
    metadata = epochs.metadata
    del epochs
    
    epochs_observed = int(df_aw.tot_epochs.unique()[0])
    
    sub_behav = df_behavior.loc[
        (df_behavior.sub_id == f"sub_{sub_id}")
        & (df_behavior.daytime == session)
        ]

    for n_epoch in range(epochs_observed) :
        if sub_id == 'HI_002' and session == "AM" and n_epoch == 0:
            continue
        sub_df = df_aw.loc[df_aw.n_epoch == n_epoch]
        
        if sub_df.empty :
            mindstate = metadata.iloc[n_epoch].mindstate
            voluntary = metadata.iloc[n_epoch].voluntary
            sleepiness = metadata.iloc[n_epoch].sleepiness
            nblock = metadata.iloc[n_epoch].nblock
            nprobe = metadata.iloc[n_epoch].nprobe
        
        else : 
            mindstate = sub_df.mindstate.iloc[0]
            voluntary = sub_df.voluntary.iloc[0]
            sleepiness = sub_df.sleepiness.iloc[0]
            nblock = sub_df.nblock.iloc[0]
            nprobe = sub_df.nprobe.iloc[0]
        
        for freq_range in ["delta", "theta"] :
            for chan in channels :
                temp_df = sub_df.loc[
                    (sub_df["channel"] == chan)
                    & (sub_df["freq_range"] == freq_range)
                    ]
            
                df_20 = temp_df.loc[temp_df.PTP > 20]
                df_90_hs_global = temp_df.loc[temp_df.PTP > global_thresh]
                df_90_hs_chan = temp_df.loc[temp_df.PTP > df_chanthresh.loc[
                    df_chanthresh.channel == chan].threshold_90_chan.iloc[0]
                    ]
                
                sub_dic['density_20'].append(df_20.shape[0])
                sub_dic['density_90_hs_global'].append(df_90_hs_global.shape[0])
                sub_dic['density_90_hs_chan'].append(df_90_hs_chan.shape[0])
                
                sub_dic['sub_id'].append(sub_id)
                sub_dic['subtype'].append(subtype)
                
                sub_dic['channel'].append(chan)
                sub_dic['n_epoch'].append(n_epoch)
                sub_dic['nblock'].append(nblock)
                sub_dic['freq_range'].append(freq_range)
                
                sub_dic['nprobe'].append(nprobe)
                sub_dic['mindstate'].append(mindstate)
                sub_dic['voluntary'].append(voluntary)
                sub_dic['sleepiness'].append(sleepiness)
                
                
                sub_dic['rt_go'].append(sub_behav.rt_go.iloc[n_epoch])
                sub_dic['rt_nogo'].append(sub_behav.rt_nogo.iloc[n_epoch])
                sub_dic['hits'].append(sub_behav.hits.iloc[n_epoch])
                sub_dic['miss'].append(sub_behav.miss.iloc[n_epoch])
                sub_dic['correct_rejections'].append(sub_behav.correct_rejections.iloc[n_epoch])
                sub_dic['false_alarms'].append(sub_behav.false_alarms.iloc[n_epoch])
                
                if not df_20.shape[0] :
                    sub_dic['ptp_20'].append(np.nan)
                    sub_dic['uslope_20'].append(np.nan)
                    sub_dic['dslope_20'].append(np.nan)
                    sub_dic['frequency_20'].append(np.nan)
                else : 
                    sub_dic['ptp_20'].append(np.nanmean(df_20.PTP))
                    sub_dic['uslope_20'].append(np.nanmean(
                        df_20.max_pos_slope_2nd_segment
                        ))
                    sub_dic['dslope_20'].append(np.nanmean(
                        df_20.inst_neg_1st_segment_slope
                        ))
                    sub_dic['frequency_20'].append(np.nanmean(
                        1/df_20.pos_halfway_period
                        ))
                    
                if not df_90_hs_global.shape[0] :
                    sub_dic['ptp_90_hs_global'].append(np.nan)
                    sub_dic['uslope_90_hs_global'].append(np.nan)
                    sub_dic['dslope_90_hs_global'].append(np.nan)
                    sub_dic['frequency_90_hs_global'].append(np.nan)
                else :    
                    sub_dic['ptp_90_hs_global'].append(np.nanmean(
                        df_90_hs_global.PTP
                        ))
                    sub_dic['dslope_90_hs_global'].append(np.nanmean(
                        df_90_hs_global.max_pos_slope_2nd_segment
                        ))
                    sub_dic['uslope_90_hs_global'].append(np.nanmean(
                        df_90_hs_global.inst_neg_1st_segment_slope
                        ))
                    sub_dic['frequency_90_hs_global'].append(np.nanmean(
                        1/df_90_hs_global.pos_halfway_period
                        ))
                if not df_90_hs_chan.shape[0]:
                    sub_dic['ptp_90_hs_chan'].append(np.nan)
                    sub_dic['uslope_90_hs_chan'].append(np.nan)
                    sub_dic['dslope_90_hs_chan'].append(np.nan)
                    sub_dic['frequency_90_hs_chan'].append(np.nan)
                else :
                    sub_dic['uslope_90_hs_chan'].append(np.nanmean(
                        df_90_hs_chan.PTP
                        ))
                    sub_dic['ptp_90_hs_chan'].append(np.nanmean(
                        df_90_hs_chan.max_pos_slope_2nd_segment
                        ))
                    sub_dic['dslope_90_hs_chan'].append(np.nanmean(
                        df_90_hs_chan.inst_neg_1st_segment_slope
                        ))
                    sub_dic['frequency_90_hs_chan'].append(np.nanmean(
                        1/df_90_hs_chan.pos_halfway_period
                        ))
            
    df_feature = pd.DataFrame.from_dict(sub_dic)
    df_feature.to_csv(this_df_swfeat_savepath)
    df_featlist.append(df_feature)
    
df = pd.concat(df_featlist)
df.to_csv(os.path.join(
    wavesPath, 
    "features", 
    "all_SW_features_delta_theta_sep.csv"
    ))
