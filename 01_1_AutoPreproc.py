#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:54:51 2024

@author: arthurlecoz

01_1_AutoPreproc.py
"""

# %% Packages & Variables

import mne
import mne_features
import numpy as np
import os
import config as cfg
import autoreject
from glob import glob
from mne.preprocessing import ICA
from autoreject import AutoReject 
from mne_icalabel import label_components

import matplotlib.pyplot as plt

rootpath = cfg.rootpath
rawpath = os.path.join(rootpath, "00_Raw")
preprocpath = os.path.join(rootpath, "01_Preproc")

taskepochspath = os.path.join(preprocpath, "epochs_epochs")
probespath = os.path.join(preprocpath, "raws_probes")
restingspath = os.path.join(preprocpath, "raws_restings")
autorejectpath = os.path.join(preprocpath, "autoreject")
icapath = os.path.join(preprocpath, "ica_files")
figpath = os.path.join(rootpath, "figs")

files = glob(os.path.join(rawpath, "experiment", "**", "*.vhdr"))

# %% Script

savereport = 1
inspect = 1
closefig = 1

for file in files :
    sub_id = file.split('_preproc/')[1][:5]    
    
    thisTrialSavepath = os.path.join(
        preprocpath, 'autopreproc', f"{sub_id }_trials_preproc_epo.fif")
    
    if os.path.exists(thisTrialSavepath) :
        print(f"\n{sub_id} already maximally processed, skipping...")
        continue
    
    print(f"\n... Processing {sub_id}...\n")
    
    raw = mne.io.read_raw_fif(file, preload = True)
    
    if savereport :
        report = mne.Report(title=f'Preproc report for: {sub_id}')
    
    #### 1) INTERPOLATE HIGH VARIANCE CHANNELS
    to_reject_chans = []
    
    eeg_data=raw.copy()
    eeg_data.pick_types(eeg=True)
    eeg_data_matrix=eeg_data.get_data() # Retrieve raw data all chans (voltages along time)
    
    '''A). STANDARD DEVIATION'''
        # Compute std of each chan separately
    std_all=np.std(eeg_data_matrix, axis=1) 
    
        # Plot the distribution of LOG STANDARD DEVIATION 
    log_std_all=np.log(std_all) 
    mean_log_std_all=np.nanmean(log_std_all) # Mean of log(stds)
    std_log_std_all=np.nanstd(log_std_all) # Std of log(stds)
    
        #Define the threshold(s)
    #log_std_thresh=mean_log_std_all + 3*std_log_std_all
    pos_log_std_thresh=mean_log_std_all + 3*std_log_std_all 
    neg_log_std_thresh=mean_log_std_all - 3*std_log_std_all 
    
    if inspect :
        plt.figure(figsize=(8, 4))
        plt.hist(
            log_std_all, bins=20, color='#fee0d2', alpha=1, edgecolor='black'
            )
        plt.axvline(
            x=pos_log_std_thresh, color='#fc9272', alpha=0.7,linestyle='--', 
            label=f'X = {pos_log_std_thresh}') #x=log_std_thresh
        plt.axvline(
            x=neg_log_std_thresh, color='#fc9272', alpha=0.7,linestyle='--', 
            label=f'X = {neg_log_std_thresh}')
        plt.title('Distribution of log(stds) (192 channels)')
        plt.xlabel('Log(std) value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(figpath, "Preproc", 'Log_Std_distrib.png'))
    if closefig : 
        plt.close()
    if savereport : 
        report.add_image(
            os.path.join(figpath, "Preproc", 'Log_Std_distrib.png'), 
            title=' Log(standard deviation) distribution'
            )
    
    
    '''B). KURTOSIS'''
        # Compute kurtosis of each chan separately
    kurt_all = mne_features.univariate.compute_kurtosis(eeg_data_matrix) 
    
        # Plot the distribution of LOG KURTOSIS 
    log_kurt_all = np.log(kurt_all) 
    mean_log_kurt_all = np.nanmean(log_kurt_all) # Mean of log(stds)
    std_log_kurt_all = np.nanstd(log_kurt_all) # Std of log(stds)
    
        #Define the threshold(s)
    #log_kurt_thresh=mean_log_kurt_all + 3*std_log_kurt_all
    pos_log_kurt_thresh=mean_log_kurt_all + 3*std_log_kurt_all
    neg_log_kurt_thresh=mean_log_kurt_all - 3*std_log_kurt_all
    
    if inspect : 
        plt.figure(figsize=(8, 4))
        plt.hist(
            log_kurt_all, bins=20, color='#e7e1ef', alpha=1, edgecolor='black'
            )
        plt.axvline(
            x=pos_log_kurt_thresh, color='#c994c7', alpha=0.7, linestyle='--', 
            label=f'X = {pos_log_kurt_thresh}') #x=log_kurt_thresh
        plt.axvline(
            x=neg_log_kurt_thresh, color='#c994c7', alpha=0.7, linestyle='--', 
            label=f'X = {neg_log_kurt_thresh}') #x=log_kurt_thresh
        plt.title('Distribution of log(kurtosis) (192 channels)')
        plt.xlabel('Log(kurtosis) value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        plt.savefig(
            os.path.join(figpath, "Preproc", 'Log_Kurtosis_distrib.png')
            )
    if closefig :
        plt.close()
    if savereport : 
        report.add_image(
            os.path.join(figpath, "Preproc", 'Log_Kurtosis_distrib.png'), 
            title=' Log(kurtosis) distribution')
        
    for chan in eeg_data.ch_names:
        # Obtain data matrix for the chan
        ch_data=raw.get_data(picks=chan) 
        # Compute standard deviation for the chan
        std_ch = np.std(ch_data)            
        log_std_ch=np.log(std_ch)
        # Compute kurtosis for the chan
        kurt_ch = mne_features.univariate.compute_kurtosis(ch_data)   
        log_kurt_ch=np.log(kurt_ch)
        
        #if log_std_ch > log_std_thresh or log_kurt_ch > log_kurt_thresh:
        if (log_std_ch > pos_log_std_thresh 
            or log_std_ch < neg_log_std_thresh):
            print(f"Channel {chan} should be rejected based on std.")
            to_reject_chans.append(chan)
        elif (log_kurt_ch > pos_log_kurt_thresh 
              or log_kurt_ch < neg_log_kurt_thresh):
            print(f"Channel {chan} should be rejected based on kurtosis.")
            to_reject_chans.append(chan)
        else:
            print(f"Channel {chan} is kept.")
    
    rej_chan_list = to_reject_chans
    raw.info['bads'].extend(rej_chan_list)
    
    # Add rej_chan_list on mne Report
   
    if savereport :
        rej_chan_list_html = "<p>".join(rej_chan_list)
        report.add_html(
            rej_chan_list_html, 
            title='Std/Kurtosis interpolated channels'
            )
    
    # Interpolate bad channels 
    data_bef_Int = raw.copy()
    len(data_bef_Int.info['bads']) #n° of elect to interp
    raw.interpolate_bads(reset_bads=True)
    print(len(data_bef_Int.ch_names), '→', len(
         raw.ch_names))  # should be the same
    
    # Add bads elect to report
    if savereport : 
        report.add_raw(
            raw, 
            title='Raw data - bad chans interpolated', 
            psd=True
            ) 
        
    del eeg_data
    
    raw_trials = raw.copy()
    raw_probes = raw.copy()
    events, event_id = mne.events_from_annotations(raw_probes)
    events_probes = events[np.where(events[:, 2] == 7)[0]]
        # Hardcoding 
    events_probe1 = events_probes[
        np.append(np.array(True), np.diff(events_probes[:, 0]) > 20000)
        ]
    
    raw_probes.annotations.delete(
        np.where(raw_probes.annotations.description)[0]
        )
    annot_probes = mne.annotations_from_events(
        events_probe1, 
        raw.info['sfreq']
        )
    raw_probes.annotations.append(
        onset = annot_probes.onset,
        duration = annot_probes.duration,
        description = annot_probes.description
        )
    
    cfg.sort_out_annot(
        raw_trials, sub_id, behavpath, taskepochspath, save_annot = False
        )
    eeg_ica = raw_trials.copy()
    eeg_probes = raw_probes.copy()
    
    #### 2) PREPARE EEG DATA FOR ICA
    
    print('Working on eeg_ica ...')
    
    # Average referencing
    mne.set_eeg_reference(eeg_ica, ref_channels='average', copy = False)
    
    # Filtering
    eeg_ica.filter(1,30, fir_design='firwin', n_jobs = -1) 
    
    # Data already Notched
    # eeg_ica.notch_filter(
    #     np.arange(50,125,50),
    #     filter_length='auto',
    #     phase='zero'
    #     )
    
    events, event_id = mne.events_from_annotations(eeg_ica)
    
    # Create 3s epochs from events
    flat_criteria = dict(eeg=1e-6) # 1 µV
    epochs = mne.Epochs(
        eeg_ica, 
        events, 
        event_id = event_id, 
        tmin = -1,
        tmax = 2,
        baseline = None,
        preload = True,
        flat = flat_criteria,
        event_repeated = 'merge'
        )
    if savereport : 
        evks = epochs.average()         
        report.add_evokeds(
            evks,
            titles=('3s Evoked after Notch, bad chans interpolation,' +
                    'av reref, [1-30] firwin filtering for ICA, BEFORE AR')
            )
        del evks
        
    # Clear memory
    del eeg_ica
    
    # Autoreject
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)
    ar = AutoReject(n_interpolates, consensus_percs,
                thresh_method='random_search', random_state=42)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    print(f'Number of epochs originally: {len(epochs)}, '
      f'after autoreject: {len(epochs_clean)}')
    
    if savereport :
        report.add_epochs(
            epochs_clean, 
            title=('3s Epochs after Notch, bad chans interpolation, ' +
                   'av reref, [1-30] firwin filtering for ICA, AFTER AR'), 
            psd=True
            )
        
    evks_clean = epochs_clean.average()
    if savereport :
        report.add_evokeds(
            evks_clean,
            titles=('3s Evoked after Notch, bad chans interpolation,' +
                    'av reref, [1-30] firwin filtering for ICA, AFTER AR')
            )
        del evks_clean
    # Visualize cleaning

    if any(reject_log.bad_epochs) :
        if savereport :
            fig = reject_log.plot('horizontal')
            fig.savefig(
                figpath + '/' + sub_id + 
                f':{len(epochs)}'f': {len(epochs_clean)}_AR_matrix.png'
                )
            report.add_image(
                figpath + '/' + sub_id + 
                f':{len(epochs)}'f': {len(epochs_clean)}_AR_matrix.png', 
                  title='5s epochs Autoreject matrix'
                  )
            if closefig :
                plt.close()
            
        #compute ERP of bad/removed epochs
        evoked_bad = epochs[reject_log.bad_epochs].average() 
        plt.figure()
        plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
        #add cleaned ERP to compare
        epochs_clean.average().plot(axes=plt.gca()) 
        if closefig :
            plt.close()
        del evoked_bad
        
    #### 3) COMPUTE ICA - KEEP WEIGHTS
    
    # Compute ICA
    ica = mne.preprocessing.ICA(
        n_components=15, 
        max_iter='auto', 
        random_state=97, 
        method='infomax', 
        fit_params=dict(extended=True)
        )
    ica.fit(epochs_clean) #or onepochs[~reject_log.bad_epochs]
    ica
    fig = ica.plot_sources(epochs_clean, show_scrollbars=True) #temporal
    ica.plot_components(inst=epochs_clean) #spatial
    
    
    # ICA label
    ic_labels=label_components(epochs_clean, ica, method='iclabel')
    print(ic_labels["labels"])
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) 
        if label in ["eye blink", "saccade", "muscle artifact", "channel noise"]
        ]
    print(f"Excluding these ICA components: {exclude_idx}")
    ica.exclude = exclude_idx
    
    report.add_ica(
        ica=ica, 
        inst=epochs_clean, 
        title='3s epochs ICA_display', 
        n_jobs=-1
        )
    report.add_figure(fig, title='ICA temporal sources')
    
    # 5) PREPARE EEG FOR PROBES
    
    print('\nWorking on Probes Epochs ...\n')
    
    # Average referencing
    mne.set_eeg_reference(eeg_probes, ref_channels='average', copy = False)

    # Filtering
        # My data were already notched and filtered
    # hpass=0.1
    # iir_params=dict(ftype='butter', order=6)
    # eeg_probes.filter(
    #     l_freq=hpass,
    #     h_freq=None, 
    #     n_jobs=-1, 
    #     method='iir', 
    #     iir_params=iir_params
    #     )
    
    # lpass=40
    # iir_params=dict(ftype='butter', order=8)
    # eeg_probes.filter(
    #     l_freq=None,
    #     h_freq=lpass, 
    #     n_jobs=-1, 
    #     method='iir', 
    #     iir_params=iir_params
    #     )
    
    # add plot on report
    if savereport : 
        fig=eeg_probes.compute_psd(method='welch', fmin=0, fmax=np.inf).plot()
        report.add_figure(
            fig, 
            title=('Welch PSD on continuous data after Notch,' +
                   ' bad chans interpolation, av reref, [0.1-40]' + sub_id)
            )
        if closefig :
            plt.close()
    
    events, _ = mne.events_from_annotations(eeg_probes)
    probes_num = [i+1 for i in range(events.shape[0])]
    event_id = {f"probe/{i+1}":i+1 for i in range(events.shape[0])}
    events[:, 2] = probes_num
    
    probes_epochs = mne.Epochs(
        eeg_probes,
        events, 
        event_id = event_id, 
        tmin=-10, 
        tmax=0, 
        baseline = None, 
        reject = None,  
        preload = True
        )
    probes_epochs.plot(block=False)
    if closefig :
        plt.close()
    
    # clear memory
    del eeg_probes, raw_probes
    
    if savereport : 
        report.add_epochs(probes_epochs, title='Probe epochs', psd=True)
        
        probe_evks=probes_epochs.average() 
        report.add_evokeds(
            probe_evks,
            titles=('Probes after Notch, bad chans interpolation,' + 
                    'av reref, [0.1,40] filtering, BEFORE AR')
            )
        del probe_evks
                
    # Autoreject
    ar.fit(probes_epochs)
    probes_epochs_clean, probes_reject_log = ar.transform(
        probes_epochs, return_log=True
        )
    print(f'Number of epochs originally: {len(probes_epochs)}, '
      f'after autoreject: {len(probes_epochs_clean)}')
    
    if savereport : 
        report.add_epochs(
            probes_epochs_clean, 
            title=('Stim On Epochs after Notch, bad chans interpolation,' + 
                   ' av reref, [0.1,40] filtering, AFTER AR'), 
            psd=True
            )
        probe_evks_clean = probes_epochs_clean.average()
        report.add_evokeds(
            probe_evks_clean,
            titles=('Probes after Notch, bad chans interpolation, '+ 
                    'av reref, [0.1,40] filtering, AFTER AR')
            )
        del probe_evks_clean
        
    # Visualize cleaning

    if any(probes_reject_log.bad_epochs) :
        if savereport :
            fig = probes_reject_log.plot('horizontal')
            fig.savefig(os.path.join(
                figpath, 
                f"{sub_id}:{len(probes_epochs)}:{len(probes_epochs_clean)}_AR_matrix.png"
                ))
            report.add_image(os.path.join(
                figpath, 
                f"{sub_id}:{len(probes_epochs)}:{len(probes_epochs_clean)}_AR_matrix.png"
                ), 
                title='Stim On epochs Autoreject matrix')
            if closefig :
                plt.close()
            
        #compute ERP of bad/removed epochs
        probes_evoked_bad = probes_epochs[
            probes_reject_log.bad_epochs].average() 
        plt.figure()
        plt.plot(
            probes_evoked_bad.times, 
            probes_evoked_bad.data.T * 1e6, 
            'r', 
            zorder=-1
            )
        #add cleaned ERP to compare
        probes_epochs_clean.average().plot(axes=plt.gca()) 
        if closefig :
            plt.close()
        del probes_evoked_bad
        
    #### 5) APPLY ICA WEIGHTS ON CLEANED PROBES EEG
    
    # make a copy first
    ICA_probes_epochs_clean = probes_epochs_clean.copy() 
    # or ica.apply(ICA_epochs_clean, exclude=exclude_idx)
    ica.apply(ICA_probes_epochs_clean) 
    
    if savereport :
        #after autoreject and ICA
        evks_probe_clean_ICA=ICA_probes_epochs_clean.average() 
        report.add_evokeds(
            evks_probe_clean_ICA,
            titles = ('Probes after Notch, bad chans interpolation,' + 
                      ' av reref, [0.1,40] filtering, ' + 
                      'after AR and ICA weights application')
            )
    
    # with topos
    topo = evks_probe_clean_ICA.plot_joint(picks=None)
    topo.savefig(figpath + '/' + sub_id + '_plotjoint_preproc_ep.png')
    del evks_probe_clean_ICA
    
    #### 6) SAVE AND COPY CLEANED DATA
    
    ICA_probes_epochs_clean.save(os.path.join(
        preprocpath, 'autopreproc', f"{sub_id }_probes_preproc_epo.fif"), 
        overwrite=True)
    print('Preprocessing over, epochs saved here')
    
    del ICA_probes_epochs_clean, probes_epochs, probes_epochs_clean
    
    #### 7) REPEAT FOR TRIALS EPOCHS
    events, event_id = mne.events_from_annotations(raw_trials)
    epochs_trials = mne.Epochs(
        raw_trials, 
        events, 
        event_id = event_id, 
        tmin = -1,
        tmax = 2,
        baseline = None,
        preload = True,
        flat = flat_criteria,
        event_repeated = 'merge'
        )
    
    del raw_trials
    
    if savereport : 
        report.add_epochs(epochs_trials, title='Trials epochs', psd=True)
        
        trial_evks=epochs_trials.average() 
        report.add_evokeds(
            trial_evks,
            titles=('Trial after Notch, bad chans interpolation, ' + 
                    'av reref, [0.1,40] filtering, BEFORE AR')
            )
        del trial_evks
        
    # Autoreject
    ar.fit(epochs_trials)
    epochs_trials_clean, trials_reject_log = ar.transform(
        epochs_trials, return_log=True
        )
    print(f'Number of epochs originally: {len(epochs_trials)}, '
      f'after autoreject: {len(epochs_trials_clean)}')
    
    if savereport : 
        report.add_epochs(
            epochs_trials_clean, 
            title=('Stim On Epochs after Notch, bad chans interpolation, ' +
                   'av reref, [0.1,40] filtering, AFTER AR'), 
            psd=True
            )
        trials_evks_clean = epochs_trials_clean.average()
        report.add_evokeds(
            trials_evks_clean,
            titles=('Trials after Notch, bad chans interpolation, ' +
                    'av reref, [0.1,40] filtering, AFTER AR')
            )
        del trials_evks_clean
        
    # Visualize cleaning

    if any(trials_reject_log.bad_epochs) :
        if savereport :
            fig = trials_reject_log.plot('horizontal')
            fig.savefig(os.path.join(
                figpath, 
                f"{sub_id}:{len(epochs_trials)}:{len(epochs_trials_clean)}_AR_matrix.png"
                ))
            report.add_image(os.path.join(
                figpath, 
                f"{sub_id}:{len(epochs_trials)}:{len(epochs_trials_clean)}_AR_matrix.png"
                ), 
                title='Stim On epochs Autoreject matrix')
            
        #compute ERP of bad/removed epochs
        trials_evoked_bad = epochs_trials[
            trials_reject_log.bad_epochs].average() 
        plt.figure()
        plt.plot(
            trials_evoked_bad.times, 
            trials_evoked_bad.data.T * 1e6, 
            'r', 
            zorder=-1
            )
        #add cleaned ERP to compare
        epochs_trials_clean.average().plot(axes=plt.gca()) 
        del trials_evoked_bad
        
    #### 7.5) APPLY ICA WEIGHTS ON CLEANED PROBES EEG
    
    # make a copy first
    ICA_trials_epochs_clean = epochs_trials_clean.copy() 
    # or ica.apply(ICA_epochs_clean, exclude=exclude_idx)
    ica.apply(ICA_trials_epochs_clean) 
    
    if savereport :
        #after autoreject and ICA
        evks_trials_clean_ICA=ICA_trials_epochs_clean.average() 
        report.add_evokeds(
            evks_trials_clean_ICA,
            titles = ('Probes after Notch, bad chans interpolation,'+
                      ' av reref,[0.1,40] filtering, '+
                      'after AR and ICA weights application')
            )
    
    # with topos
    topo = evks_trials_clean_ICA.plot_joint(picks=None)
    topo.savefig(figpath + '/' + sub_id + '_plotjoint_preproc_ep.png')
    del evks_trials_clean_ICA
    
    #### 7.6) SAVE AND COPY CLEANED DATA
    
    ICA_trials_epochs_clean.save(thisTrialSavepath, overwrite=True)
    print('Preprocessing over, epochs saved here')
    
    del ICA_trials_epochs_clean, epochs_trials_clean, epochs_trials, raw
    

    #### END) SAVE PREPROCESSING REPORT
    report.save(
        os.path.join(
            preprocpath, "mne_reports", f"{sub_id}_AUTOPREPROC.html"
            ), 
        overwrite = True, 
        open_browser = False
        )
    plt.close('all')
    