#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21

@author: Arthur Le Coz

04_01_swdet.py

"""

# %%% Paths & Packages

import SLHIP_config_ALC as config
import multiprocessing
import os

cleanDataPath = os.path.join(config.cleanDataPath, "epochs_probes")
    
def SW_detect(file_path):
    ### import libraries
    import numpy as np
    import pandas as pd
    import mne
    import os
    
    import SLHIP_config_ALC as config
    swDataPath = config.wavesPath
    
    eeg_epochs_data = mne.read_epochs(file_path, preload = True)
    eeg_epochs_data.pick('eeg')
    metadata = eeg_epochs_data.metadata
    
    sub_id = metadata.sub_id.unique()[0]
    
    ms_dic_int = {
        'ON' : 1,
        'MW' : 2,
        'MB' : 3,
        'HALLU' : 4,
        'FORGOT' : 5,
        'DISTRACTED' : 6,
        'MISS' : 7
        }
    ms_dic_str = {value : key for key, value in ms_dic_int.items()}
    
    waveSavingPath = f"{swDataPath}/all_waves/{sub_id}.csv"
    
    if not os.path.exists(waveSavingPath):
    
        filt_range = [0.5, 10] # in Hz
        thr_value = 0.1
    
        sfreq = eeg_epochs_data.info['sfreq']
        
        ### Prepare EEG for pre-processing
        eeg_low_bp = eeg_epochs_data.copy()
    
        ### Pre-processing of EEG data: Filtering
        eeg_low_bp.filter(
            filt_range[0], filt_range[1], 
            l_trans_bandwidth='auto', h_trans_bandwidth='auto',
            filter_length='auto', phase='zero'
            )
    
        ### Pre-processing of EEG data: Down-sample to 100Hz
        eeg_low_bp_re = eeg_low_bp.copy().resample(100, npad='auto')
        newsfreq = 100
            
        ### Loop across channels
        n_epochs, nchans, nsamples = eeg_low_bp_re._data.shape
        allWaves=[]
    
        for ch in range(0, nchans):
            print('Processing: ')
            print(eeg_epochs_data.ch_names[ch])
            tempAllWaves = []
            
            for n_epoch in range(0, n_epochs) :
                
                this_metadata = metadata.iloc[n_epoch]
                nblock = this_metadata.nblock
                nprobe = this_metadata.nprobe
                mindstate = ms_dic_int[this_metadata.mindstate]
                voluntary = this_metadata.voluntary
                sleepiness = this_metadata.sleepiness
                
                this_eeg_chan = eeg_low_bp_re.copy().pick(
                    [eeg_epochs_data.ch_names[ch]])
                this_eeg_chan = this_eeg_chan[n_epoch].get_data(units = 'uV')[0][0]
                
                if np.max(this_eeg_chan) < 1 : # probably in V and not uV
                    this_eeg_chan = this_eeg_chan * 1000000
                    print('converting to uV')
        
                ### Detection of SWs: Find 0-crossings
                zero_crossing = this_eeg_chan > 0
                zero_crossing = zero_crossing.astype(int)
            
                # Find positive zero-crossing (start of potential SW)
                pos_crossing_idx = np.where(np.diff(zero_crossing) > 0)[0]
                pos_crossing_idx = [x+1 for x in pos_crossing_idx]
                # Find negative zero-crossing (end of potential SW)
                neg_crossing_idx = np.where(np.diff(zero_crossing) < 0)[0]
                neg_crossing_idx = [x+1 for x in neg_crossing_idx]
                
                # Compute derivative and smooth (5-sample running average)
                der_eeg_chan = np.convolve(
                    np.diff(this_eeg_chan), np.ones((5,))/5, mode='valid'
                    )
                thr_crossing = der_eeg_chan > thr_value
                thr_crossing = thr_crossing.astype(int)
                difference = np.diff(thr_crossing)
                peaks = np.asarray(np.where(difference == -1)) + 1
                peaks = peaks[0]
                troughs = np.asarray(np.where(difference == 1)) + 1
                troughs = troughs[0]
                
                # Rejects peaks below zero and troughs above zero
                peaks = peaks[this_eeg_chan[peaks] > 0]
                troughs = troughs[this_eeg_chan[troughs] < 0]
                if neg_crossing_idx[0] < pos_crossing_idx[0]:   
                    start = 1
                else:
                    start = 2
                if start == 2:
                    pos_crossing_idx = pos_crossing_idx[1:]
                
                ### Detection of SWs
                waves = np.empty((len(neg_crossing_idx) - start + 1, 30))
                waves[:] = np.nan
                lastpk = np.nan
                for wndx in range(start,len(neg_crossing_idx) - 1):
                    wavest = neg_crossing_idx[wndx]
                    wavend = neg_crossing_idx[wndx + 1]
                    # matrix (27) determines instantaneous positive 1st segement slope on smoothed signal, (name not representative)
                    if pos_crossing_idx[wndx] > len(der_eeg_chan):
                        continue
                    mxdn = np.abs(
                        np.min(der_eeg_chan[wavest:pos_crossing_idx[wndx]])
                        ) * newsfreq;  
                    # matrix (28) determines maximal negative slope for 2nd segement (name not representative)
                    mxup = np.max(
                        der_eeg_chan[wavest:pos_crossing_idx[wndx]]
                        ) * newsfreq; 
                    tp1 = np.where(troughs>wavest)
                    tp2 = np.where(troughs<wavend)
                    negpeaks = troughs[np.intersect1d(tp1,tp2)]
                    
                    # In case a peak is not detected for this wave (happens rarely)
                    if np.size(negpeaks) == 0:
                        waves[wndx, :] = np.nan
                        # thisStage=newscoring[wavest];
                        # waves[wndx,22] =thisStage;
                        continue
                    
                    tp1 = np.where(peaks > wavest)
                    tp2 = np.where(peaks <= wavend)
                    pospeaks = peaks[np.intersect1d(tp1,tp2)]
                    
                    # if negpeaks is empty set negpeak to pos ZX
                    if np.size(pospeaks) == 0 :
                        pospeaks = np.append(pospeaks,wavend)
                        
                    period = wavend-wavest #matrix(11) /SR
                    poszx = pos_crossing_idx[wndx] #matrix(10)
                    b = [np.min(this_eeg_chan[negpeaks])][0] #matrix (12) most pos peak /abs for matrix
                    #if len(b)>1:
                    #    b=b[0]
                    bx = negpeaks[np.where(this_eeg_chan[negpeaks]==b)][0] #matrix (13) max pos peak location in entire night
                    c = [np.max(this_eeg_chan[pospeaks])][0] #matrix (14) most neg peak
                    #if len(c)>1:
                    #    c=c[0]
                    cx = pospeaks[np.where(this_eeg_chan[pospeaks]==c)] #matrix (15) max neg peak location in entire night
                    cx = cx[0]
                    maxb2c = c-b #matrix (16) max peak to peak amp
                    nump = len(negpeaks) #matrix(24) now number of positive peaks
                    n1 = np.abs(this_eeg_chan[negpeaks[0]]) #matrix(17) 1st pos peak amp
                    n1x = negpeaks[0] #matrix(18) 1st pos peak location
                    nEnd = np.abs(this_eeg_chan[negpeaks[len(negpeaks)-1]]) #matrix(19) last pos peak amp
                    nEndx = negpeaks[len(negpeaks)-1] #matrix(20) last pos peak location
                    p1 = this_eeg_chan[pospeaks[0]] #matrix(21) 1st neg peak amp
                    p1x = pospeaks[0] #matrix(22) 1st pos peak location
                    meanAmp = np.abs(np.mean(this_eeg_chan[negpeaks])); #matrix(23)
                    nperiod = poszx-wavest; #matrix (25)neghalfwave period
                    mdpt = wavest+np.ceil(nperiod/2); #matrix(9)
                    p2p = (cx-lastpk)/newsfreq; #matrix(26) 1st peak to last peak period
                    lastpk = cx;
                    
                    #### Result Matrix
                    #0:  wave beginning (sample)
                    #1:  wave end (sample)
                    #2:  wave middle point (sample)
                    #3:  wave negative half-way (sample)
                    #4:  period in seconds
                    #5:  negative amplitude peak
                    #6:  negative amplitude peak position (sample)
                    #7:  positive amplitude peak
                    #8:  positive amplitude peak position (sample)
                    #9:  peak-to-peak amplitude
                    #10: 1st neg peak amplitude
                    #11: 1st neg peak amplitude position (sample)
                    #12: Last neg peak amplitude
                    #13: Last neg peak amplitude position (sample)
                    #14: 1st pos peak amplitude
                    #15: 1st pos peak amplitude position (sample)
                    #16: mean wave amplitude
                    #17: number of negative peaks
                    #18: wave positive half-way period
                    #19: 1st peak to last peak period
                    #20: determines instantaneous negative 1st segement slope on smoothed signal
                    #21: determines maximal positive slope for 2nd segement
                    #22: epoch number
                    #23: total epochs explored
                    #24: channel number 
                    #25: block number of the probe (SART)
                    #26: probe number of the block (SART)
                    #27: Mindstate number (1 ON, 2 MW, 3 MB, etc. see dic)
                    #28: voluntary answer to the probe (1 (y) or 2 (n))
                    #29: sleepiness answer to the probe (1 - 9 : KSS)
                    
                    waves[wndx, :] = (
                        wavest, wavend, mdpt, poszx, period/newsfreq, 
                        np.abs(b), bx, c, cx, maxb2c, n1, n1x, nEnd, 
                        nEndx, p1, p1x, meanAmp, nump, nperiod/newsfreq, 
                        p2p, mxdn, mxup, n_epoch, n_epochs, ch, nblock, 
                        nprobe, mindstate, voluntary, sleepiness
                        )
                    waves[wndx, (0,1,2,3,6,8,11,13,15)] = waves[
                        wndx, (0,1,2,3,6,8,11,13,15)]*(sfreq/newsfreq);
                tempAllWaves.append(waves)
            allWaves.append(np.concatenate(tempAllWaves, axis = 0))
            
            print(f"...Number of all-waves : {np.size(allWaves[ch], 0)}...\n")
        
        df = pd.DataFrame(
            np.concatenate(allWaves, axis = 0),
            columns = [
                "start", 
                "end", 
                "middle", 
                "neg_halfway", 
                "period", 
                "neg_amp_peak", 
                "neg_peak_pos", 
                "pos_amp_peak", 
                "pos_peak_pos", 
                "PTP", 
                "1st_negpeak_amp", 
                "1st_negpeak_amp_pos", 
                "last_negpeak_amp", 
                "last_negpeak_amp_pos", 
                "1st_pospeak_amp", 
                "1st_pospeak_amp_pos", 
                "mean_amp", 
                "n_negpeaks", 
                "pos_halfway_period", 
                "1peak_to_npeak_period", 
                "inst_neg_1st_segment_slope", 
                "max_pos_slope_2nd_segment",
                "n_epoch",
                "tot_epochs",
                "n_chan",
                "nblock",
                "nprobe", 
                "mindstate",
                "voluntary", 
                "sleepiness"
                ])
        df = df.dropna()
        df.insert(
            24, 
            "channel", 
            [eeg_epochs_data.ch_names[int(ch)] for ch in df.n_chan]
            )
        df.mindstate = df.mindstate.replace(ms_dic_str)
        
        df.to_csv(waveSavingPath)
        
        return "...\n\n ALL FILES WERE COMPUTED"

if __name__ == '__main__':
    from glob import glob
    # Get the list of EEG files
    eeg_files = glob(f"{cleanDataPath}/*epo.fif")
    
    # Set up a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = 1)
    
    # Process the EEG files in parallel
    pool.map(SW_detect, eeg_files)
    
    # Clean up the pool of worker processes
    pool.close()
    pool.join()

    
 
    