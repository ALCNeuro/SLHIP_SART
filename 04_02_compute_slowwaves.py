#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21

@author: Arthur Le Coz

04_02_compute_slowwaves.py
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

allwaves_files = glob(os.path.join(wavesPath, "all_waves", "*.csv"))
slowwaves_path = os.path.join(wavesPath, "slow_waves")
reports_path = os.path.join(wavesPath, "reports")
figs_path = os.path.join(wavesPath, "figs")
# epochs_files  = glob(os.path.join(cleanDataPath, "*epo.fif"))

slope_range = [0.143, 2] # in uV/ms
positive_amp = [75] # in uV
amplitude_max = 150

inspect = 0

# %%% Script

for i_file, file in enumerate(allwaves_files) :
    df_aw = pd.read_csv(file) 
    del df_aw['Unnamed: 0']
    
    sub_id = file.split('all_waves/')[1].split('.')[0]
    subtype = sub_id[:2]
    
    this_sw_df_savename = os.path.join(slowwaves_path, f"slow_waves_{sub_id}.csv")
    this_thresh_df_savename = os.path.join(slowwaves_path, f"thresh_{sub_id}.csv")
    
    epochs_observed = df_aw.tot_epochs.unique()[0]
    
    print(f"\n...Processing {sub_id} : {i_file+1} / {len(allwaves_files)}...")
    
    df_sw = df_aw.loc[(df_aw['PTP'] < 150) & (df_aw['pos_amp_peak'] < 75)]
    df_sw = df_sw.loc[
        (df_sw["pos_halfway_period"] <= slope_range[1])
        & (df_sw["pos_halfway_period"] >= slope_range[0])
        ]
    
    if inspect :
        report = mne.Report(title=f"PTP inspections of {sub_id}")
    
    thresh_dic = {}
    print(f"\n...Processing {sub_id}... Computing Thresholds")
    for i, chan in enumerate(df_sw.channel.unique()) :
        temp_p2p = np.asarray(df_sw.PTP.loc[df_sw['channel'] == chan])
        if len(temp_p2p) == 0 :
            continue
        params = exponnorm.fit(temp_p2p)
        mu, sigma, lam = params
        bins = np.arange(0, temp_p2p.max(), 0.1)
        y = exponnorm.pdf(bins, mu, sigma, lam)
        max_gaus = bins[np.where(y == max(y))][0] * 2
        
        thresh_dic[chan] = max_gaus
    
        if inspect :
            fig1 = plt.figure(f"GaussianFit_{sub_id}_{chan}")
            plt.hist(
                temp_p2p, bins = 100, density = True, 
                alpha = 0.5, label = f"PTP_SW_{chan}",
                # color = palette_bins[i]
                )
            plt.plot(bins, y, #color = palette_fit[i], 
                      label = f"Ex-GaussFit_{chan}")
            plt.axvline(
                x = max_gaus, #color = palette_bins[i], 
                label = f"Threshold_{chan}", ls = '--')
            plt.xlabel('Values')
            plt.ylabel('Density')
            plt.title('Ex-Gaussian Fit')
            plt.legend()
            plt.show(block=False)
            plt.close(fig1)
        
    if inspect :
        report.add_figure(
            fig=fig1,
            title=f"Ex Gaussian Fit on PTP distrib at {chan}",
            image_format="PNG"
            )
        report.save(
            f"{reports_path}/PTP_report_{sub_id}.html", 
            overwrite=True,
            open_browser = False)  
          
    df_clean = pd.concat(
        [df_sw[
            (df_sw.channel == chan) 
            & (df_sw.PTP > thresh_dic[chan])]
            for chan in thresh_dic.keys() 
            if len(thresh_dic.keys()) > 0
            ]
        )
    
    df_thresh = pd.DataFrame.from_dict(thresh_dic, orient = 'index')
    df_thresh.to_csv(this_thresh_df_savename)
    
    df_clean.to_csv(this_sw_df_savename)
        