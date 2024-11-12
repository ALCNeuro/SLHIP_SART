#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:50:28 2024

@author: arthurlecoz

04_03_compute_swdensity.py
"""
# %%% Paths & Packages

import SLHIP_config_ALC as config
import mne
from glob import glob
import pandas as pd
import numpy as np
from scipy.stats import exponnorm
import matplotlib.pyplot as plt
import os

cleanDataPath = config.cleanDataPath
wavesPath = config.wavesPath

slowwaves_path = os.path.join(wavesPath, "slow_waves")
slowwaves_files = glob(os.path.join(slowwaves_path, "slow_waves*.csv"))
reports_path = os.path.join(wavesPath, "reports")
figs_path = os.path.join(wavesPath, "figs")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

slope_range = [0.143, 2] # in uV/ms
positive_amp = [75] # in uV
amplitude_max = 150

inspect = 0

features = [
    "sub_id", "subtype", 
    "density", "ptp", "uslope", "dslope", "frequency", 
    "sw_thresh", "channel", "n_epoch", "nblock", 
    "nprobe", 'mindstate','voluntary', 'sleepiness'
    ]

channels = np.array(config.eeg_channels)

# %% Script

df_featlist = []
for i_file, file in enumerate(slowwaves_files) :
    bigdic = {feat : [] for feat in features}
    sub_id = file.split('/slow_waves_')[1].split('.')[0]
    subtype = sub_id[:2]
    
    this_swthresh_path = os.path.join(
        slowwaves_path, f"thresh_{sub_id}.csv"
        )
    df_threshold = pd.read_csv(this_swthresh_path)
    df_threshold.rename(
        columns = {'Unnamed: 0' : 'channel', '0' : 'sw_threshold'}, 
        inplace = True
        )
    
    this_df_swfeat_savepath = os.path.join(
        wavesPath, 
        "features", 
        f"{sub_id}_{1/slope_range[1]}_{round(1/slope_range[0])}.csv"
        )

    print(f"\n...Processing {sub_id}... Computing SW Features")  
    
    df_sw = pd.read_csv(file)
    del df_sw['Unnamed: 0']
    
    # df_threshold = df_sw[["channel", "sw_threshold"]].groupby(
    #     ["channel", "sw_threshold"], as_index = False).mean()
    epochs_observed = int(df_sw.tot_epochs.unique()[0])

    for n_epoch in range(epochs_observed) :
        sub_df = df_sw.loc[df_sw.n_epoch == n_epoch]
        
        if sub_df.empty :
            epochs_files = glob(os.path.join(
                cleanDataPath, 
                "epochs_probes",
                f"*{sub_id}*epo.fif"
                ))[0]
            epochs = mne.read_epochs(epochs_files, preload = False)
            metadata = epochs.metadata
            del epochs
            
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
        
            n_waves = temp_df.shape[0]
            sw_thresh = df_threshold.sw_threshold.loc[
                df_threshold.channel == chan].iloc[0]
            
            bigdic['density'].append(n_waves)
            
            bigdic['sub_id'].append(sub_id)
            bigdic['subtype'].append(subtype)
            
            bigdic['sw_thresh'].append(sw_thresh)
            bigdic['channel'].append(chan)
            bigdic['n_epoch'].append(n_epoch)
            bigdic['nblock'].append(nblock)
            
            bigdic['nprobe'].append(nprobe)
            bigdic['mindstate'].append(mindstate)
            bigdic['voluntary'].append(voluntary)
            bigdic['sleepiness'].append(sleepiness)
            
            if n_waves == 0 :
                bigdic['ptp'].append(np.nan)
                bigdic['uslope'].append(np.nan)
                bigdic['dslope'].append(np.nan)
                bigdic['frequency'].append(np.nan)
            else : 
                bigdic['ptp'].append(np.nanmean(temp_df.PTP))
                bigdic['uslope'].append(np.nanmean(
                    temp_df.max_pos_slope_2nd_segment
                       ))
                bigdic['dslope'].append(np.nanmean(
                    temp_df.inst_neg_1st_segment_slope))
                bigdic['frequency'].append(np.nanmean(
                    1/temp_df.pos_halfway_period))
            
    df_feature = pd.DataFrame.from_dict(bigdic)
    df_feature.to_csv(this_df_swfeat_savepath)
    df_featlist.append(df_feature)
    
df = pd.concat(df_featlist)
df.to_csv(os.path.join(
    wavesPath, 
    "features", 
    f"all_SW_features_{1/slope_range[1]}_{round(1/slope_range[0])}.csv"
    ))
