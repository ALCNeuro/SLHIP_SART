#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:17:38 2024

@author: alessiaruyantbelabbas

Automatized NICE preprocessing pipeline to clean Local-Global raw files 
to match SW detection algorithm: 
    
    STRATEGY : 
        
    Compute and save the timings of the blocks start and end of Local Global, but do preproc on 
    continuous 10s epochs and then classify them back into the different blocks based on their 
    timing so that same chans are interpolated on all blocks and quicker (1 NICE preproc instead of nBl times)

    --> for DoC patients LG
    --> for healthy volunteers LG
    
    NB: with NICE montage, can remove one neck raw to match last CEM/DPM analysis
        but for now pb with montage when plotting topo (step 3), have to debug


INPUT: 'Files_to_preproc.csv', output of identify_eegfiles-to-preproc.py
    

"""

#%% IMPORT MODULES

import os
import glob
from pathlib import Path
import pickle
import json

#run ARB_config.py
#import ARB_config as config 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import mne
from mne.utils import logger
from mne_icalabel import label_components
import mne_features
from autoreject import (AutoReject,set_matplotlib_defaults)


from next_icm.lg.io_aless import(_read_lg_raw_egi_files, _check_clean_trigger,
                                 _read_lg_raw_egi, _read_lga_egi_generic,_read_lga_mff_egi,
                                 _read_lga_raw_egi)
from next_icm.lg.constants import (_icm_lg_event_id, _icm_lg_concatenation_event,
                        _lg_matlab_event_id_map, _arduino_trigger_map,
                        _gtec_trig_map)


from nice_ext.equipments import get_montage, get_ch_names, _egi_filter
from nice_ext.algorithms.adaptive import _adaptive_egi
from nice_ext.api.preprocessing import _check_min_channels, _check_min_events
import next_icm

#%% Define NICE preprocessing function for each LG block individually

def preprocess_NICE(fname,config_params):
    #Config params by default
    n_jobs = config_params.get('n_jobs', 1)
    reject = config_params.get('reject', None)
    min_events = config_params.get('min_events', 0.3)
    min_channels = config_params.get('min_channels', 0.7)
    n_epochs_bad_ch = config_params.get('n_epochs_bad_ch', 0.5)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 4)
    max_iter = config_params.get('max_iter', 4)

    autoreject = config_params.get('autoreject', False) #False to use adaptive egi
    #baseline = config_params.get('baseline', (None, 0))
    run_ica = config_params.get('ica', False) #True if want to run ICA
    summary = config_params.get('summary', None) #added
    
    
    # A) Reading block raw file 
    logger.info('fReading LG {subjectname}_{session_info} raw file')
    

    report.add_raw(raw,title='Raw data after periph elec removal and FILTERING; [0.1-45] iir 6/8th butter + 50-100 Notch fft '+subjectname, psd=True)
    fig=raw.compute_psd(method='welch', fmin=0, fmax=np.inf).plot(show=False)
    report.add_figure(fig,title='PSD raw after periph elec removal and FILTERING; [0.1-45] iir 6/8th butter + 50-100 Notch fft '+subjectname)
    
    
    # B) EPOCHING - 10s continuous 
    logger.info('Cuting epochs - 10s continuous')
    epochs = mne.make_fixed_length_epochs(raw, duration=10.0, preload=True,reject_by_annotation=False, overlap=0.0)
    epochs.apply_baseline(baseline=(None, None))
    
    epochs_plot_orig=epochs.plot(show=False)
    report.add_figure(epochs_plot_orig, title= 'Visualisation 10s epochs BEFORE ADAPTIVE CLEANING')

    # C) CLEANING WITH ADAPTIVE ALGORITHM
    
    # Copy epochs data before cleaning
    orig_epochs = epochs.copy()
    
    # Adaptative EGI preprocessing
    if reject is None:
        reject = {'eeg': 200e-6}


    logger.info("Clean epochs")
    if autoreject is True:
        logger.info('Using autoreject')
        ar = AutoReject(random_state=42, n_interpolate=np.array([1, 2, 32]),
                        consensus=np.linspace(0, 1, 11), n_jobs=2)

        epochs, reject_log = ar.fit_transform(epochs, return_log=True)
        reject_log.bad_epochs
        reject_log.labels
        reject_log.plot_epochs(epochs)
        
        fig = reject_log.plot('horizontal')
        fig.savefig(fig_path +
              '/' + subjectname + session_info + f'Block{i}' + f':{len(epochs)}'f': {len(epochs)}_AR_matrix.png')
        report.add_image(fig_path +
              '/' + subjectname + session_info + f'Block{i}' + f':{len(epochs)}'f': {len(epochs)}_AR_matrix.png', title='Autoreject matrix')
        
        
        if summary is not None: #added for lg script
            summary['autoreject'] = reject_log
            summary['steps'].append(
                dict(step='Autoreject',
                     params={'n_interpolate': ar.n_interpolate_['eeg'],
                             'consensus_perc': ar.consensus_['eeg']},
                     bad_epochs=np.where(reject_log.bad_epochs)[0]))
        _check_min_events(epochs, min_events)
        logger.info('found bad epochs: {} {}'.format(
            np.sum(reject_log.bad_epochs),
            np.where(reject_log.bad_epochs)[0]))
        
    else: #actually using Adaptative
        bad_channels, bad_epochs = _adaptive_egi(
            epochs, reject, n_epochs_bad_ch=n_epochs_bad_ch,
            n_channels_bad_epoch=n_channels_bad_epoch,
            zscore_thresh=zscore_thresh, max_iter=max_iter,
            summary=summary)
        epochs.info['bads'].extend(bad_channels)
        logger.info('found bad channels: {} {}'.format(
            len(bad_channels), str(bad_channels)))

        # _check_min_channels(epochs, bad_channels, min_channels)
        # _check_min_events(epochs, min_events)
        
        adaptative_results.append({'Subject': subjectname, 'Session': session_info, 
                                   'n° bad chans':len(bad_channels),'% bad chans':len(bad_channels)/epochs.info['nchan']*100, 
                                   'n° bad epochs':len(bad_epochs),'% bad epochs': len(bad_epochs)/len(orig_epochs)*100,
                                   'list bad chans': bad_channels, 'list bad epochs': bad_epochs}) #add block info
        
        adaptative_results_SUBJ.append({'Subject': subjectname, 'Session': session_info,
                                   'n° bad chans':len(bad_channels),'% bad chans':len(bad_channels)/epochs.info['nchan']*100, 
                                   'n° bad epochs':len(bad_epochs),'% bad epochs': len(bad_epochs)/len(orig_epochs)*100,
                                   'list bad chans': bad_channels, 'list bad epochs': bad_epochs}) #add block info
        
          
    # E) REREFERENCING TO AVERAGE
    mne.set_eeg_reference(epochs, ref_channels='average', copy = False) #equivalent to proj above

    # F) INTERPOLATE BAD CHANS
    epochs.interpolate_bads(reset_bads=True)
    
    # check bad epochs are dropped
    ep_bef_adapt=len(orig_epochs)
    ep_aft_adapt=len(epochs)
    if ep_bef_adapt == ep_aft_adapt:
        print('Bad epochs were not dropped')
    
    check_epochs_cleaning.append({'Subject': subjectname, 'Session': session_info, 
                               'n° ep bef adaptive': ep_bef_adapt, 'n° ep aft adaptive':ep_aft_adapt}) 
    
    check_epochs_cleaning_SUBJ.append({'Subject': subjectname, 'Session': session_info, 
                               'n° ep bef adaptive': ep_bef_adapt, 'n° ep aft adaptive':ep_aft_adapt}) 
    
     
    # G) ADD PLOTS TO REPORT
        #Epochs
    report.add_epochs(epochs, title= '10s NICE-preproc epochs', psd=True)
    epochs_plot=epochs.plot(show=False)
    report.add_figure(epochs_plot, title= 'Visualisation 10s NICE-preproc epochs')
    
        #PSD
    PSD=epochs.compute_psd(method='multitaper', fmin=0, fmax=np.inf).plot(show=False)
    report.add_figure(PSD, title= 'Multitaper PSD (10s NICE-preproc epochs)')
        #Evoked (ERP)
    erp=epochs.average()
    report.add_evokeds(erp,titles= 'ERP')
    
    # H) ICA
        # Compute ICA
    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97, 
                                method='infomax', fit_params=dict(extended=True))
    ica.fit(epochs) #or onepochs[~reject_log.bad_epochs]
    ica
    fig=ica.plot_sources(epochs, show=False, show_scrollbars=True) #temporal
    #ica.plot_components(inst=epochs_clean) #spatial
    
        # ICA label
    ic_labels=label_components(epochs, ica, method='iclabel')
    print(ic_labels["labels"])
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label in ["eye blink", "heart beat",
                                                                        "channel noise"]]
    
    print(f"Excluding these ICA components: {exclude_idx}")
    ica.exclude = exclude_idx
    
    report.add_ica(ica=ica, inst=epochs, title='ICA_display', n_jobs=None)
    report.add_figure(fig, title='ICA temporal sources')
    
        #Apply ICA
    ica.apply(epochs)
    
        #Evoked (ERP)
    erp_ica=epochs.average()
    report.add_evokeds(erp_ica,titles= 'ERP after ICA')
     
    
    out = epochs
    return out


#%% 1). LOAD FILES TO PREPROC

# Select DoC or healthy : 
pop = 'DoC'
#pop = 'healthy'

if pop == 'DoC':
    data_path = '/Volumes/DISK2/DoC_raw'
    csv_path='/Volumes/DISK2/DoC_raw/1b_read_select/' 
    #save_path = data_path + "/Preproc/10s/NICE-200/Blocks" 
    #save_path = data_path + "/Preproc/10s/NICE_200/Blocks_ICA_newMont" 
    save_path = data_path + "/Preproc/10s/NICE_200/BlocksAfterNICE" 

    # Load list of patients ok to preproc
    with open(csv_path + 'Files_to_preproc.csv', 'rb') as f:
        files_to_preproc = pickle.load(f)
        
    # Remove files with weird LG structure for now :
    weird_files = pd.read_csv(os.path.join(csv_path,
                                         "PB-LG_events_counts.csv"),
                                         sep=',', index_col=False)
    weird_combinations = set(zip(weird_files["Subject"], weird_files["Session"]))

    # Filter files_to_preproc list
    filtered_files_to_preproc = [
        file for file in files_to_preproc
        if (file.split('/')[4], file.split('/')[5]) not in weird_combinations
    ]

    # Load setups table
    setups = pd.read_csv(os.path.join(csv_path,
                                         "Setups_to_read.csv"),
                                         sep=',', index_col=False)
    # SUBSELECT FILES

    ## A) Subselect patients of interest: Anoxia post ardio-respi arrest, acute
    include_subjects = {'sub-YM028','sub-EM047','sub-JR052','sub-JF057'
                        ,'sub-TT058','sub-PP104','sub-DF080','sub-PC087'
                        ,'sub-AM061','sub-BA034','sub-MR026','sub-MC029'
                        ,'sub-IT077','sub-LT116','sub-MR132','sub-FL153'
                        ,'sub-SG172','sub-LL175','sub-ML259','sub-AY134'
                        ,'sub-JS138','sub-SD231','sub-JD298'}

    #subject_set = config.include_subjects # patients of interest' subjectnames
    subject_set = include_subjects # patients of interest' subjectnames

    subset_files = []
    subset_names = []

    for file in filtered_files_to_preproc:
        subjectname = file.split(sep='/')[4]
        session = file.split(sep='/')[5]
        
        # Check if current subjectname is in the set of interest
        #if subjectname in subject_set and session =="ses-01":  # for A
        if  session =="ses-01" : #and not subjectname in config.exclude_subjects:   # B, all sess 01
            subset_files.append(file) 
            subset_names.append(subjectname)

    #Read CRS-R for subset files and count states
    this_csv = os.path.join(csv_path, 'Diag_CRS.csv')
    CRS = pd.read_csv(this_csv,sep=',') 

    CRS_filtered = CRS[(CRS['Subject'].isin(subset_names))&(CRS['Session']=='ses-01')]
    CRS_1_counts = CRS_filtered['CRS-1'].value_counts()
    CRS_final_counts = CRS_filtered['CRS-final'].value_counts()
    
    
else: 
    data_path = '/Volumes/DISK2/Healthy_raw'
    csv_path='/Volumes/DISK2/Healthy_raw/1b_read_select/' 
    
    save_path = data_path + "/Preproc/10s/NICE_200/BlocksAfterNICE" 

    # Load list of patients ok to preproc
    with open(csv_path + 'Files_to_preproc.csv', 'rb') as f:
        files_to_preproc = pickle.load(f)
            
    # Remove files with weird LG structure for now :
    weird_files = pd.read_csv(os.path.join(csv_path,
                                         "PB-LG_events_counts.csv"),
                                         sep=',', index_col=False)
    weird_combinations = set(zip(weird_files["Subject"], weird_files["Session"]))

    # Filter files_to_preproc list
    filtered_files_to_preproc = [
        file for file in files_to_preproc
        if (file.split('/')[4], file.split('/')[5]) not in weird_combinations
    ]
    
    subset_files = []
    subset_names = []

    for file in filtered_files_to_preproc:
        subjectname = file.split(sep='/')[4]
        session = file.split(sep='/')[5]
        
        # Check if current subjectname is in the set of interest
        if session =="ses-01":  
            subset_files.append(file) 
            subset_names.append(subjectname)
            
    # Load setups table
    setups = pd.read_csv(os.path.join(csv_path,
                                         "Setups_to_read.csv"),
                                         sep=',', index_col=False)


#%% 2). DEFINE PERIPH CHANS TO REMOVE FROM ANALYSES

egi256_outlines = {
    'ear1': np.array([190, 191, 201, 209, 218, 217, 216, 208, 200, 190]),
    'ear2': np.array([81, 72, 66, 67, 68, 73, 82, 92, 91, 81]),
    'outer': np.array([9, 17, 24, 30, 31, 36, 45, 243, 240, 241, 242, 246, 250,
                        255, 90, 101, 110, 119, 132, 144, 164, 173, 186, 198,
                        207, 215, 228, 232, 236, 239, 238, 237, 233, 9]),
    'cheekl': np.array([244, 245, 247, 248, 249, 253, 254]),
    'cheekr': np.array([234, 229, 235, 230, 231, 226, 227]),
}

# egi256_outlines = {
#     'ear1': np.array([190, 191, 201, 209, 218, 217, 216, 208, 200, 190]),
#     'ear2': np.array([81, 72, 66, 67, 68, 73, 82, 92, 91, 81]),
#     'outer': np.array([9, 17, 24, 30, 31, 36, 45, 243, 240, 241, 242, 246, 250,
#                        255, 90, 101, 110, 119, 132, 144, 164, 173, 186, 198,
#                        207, 215, 228, 232, 236, 239, 238, 237, 233, 91, 102,
#                        111, 120, 133, 145, 155, 165, 174, 187, 199, 208]),
#     'cheekl': np.array([244, 245, 247, 248, 249, 253, 254]),
#     'cheekr': np.array([234, 229, 235, 230, 231, 226, 227]),
# }


indexes_to_remove = np.concatenate(list(egi256_outlines.values()))

channels_to_remove = ["E" + str(idx + 1) for idx in indexes_to_remove]


#%% 3). NICE PREPROCESSING - LOOP ON SUBJECTS  

error_files = []

adaptative_results = []
adaptative_results_SUBJ = []

check_epochs_cleaning = []
check_epochs_cleaning_SUBJ = []

    
for subset_file in subset_files:
    try:
    
        # Create path to save eeg preprocessed file and plots
        report_path = save_path + '/Reports/'
        timings_path = save_path + '/Timings/'
        epochs_path = save_path + '/EEG/' 
        fig_path = save_path + '/Figs/' 
        
        path_to_create=[save_path,report_path,timings_path,epochs_path,fig_path]
        for p in path_to_create:
            path = Path(p)
            path.mkdir(parents=True, exist_ok=True)
            
        subjectname = subset_file.split(sep='/')[4]
        session_info = subset_file.split(sep='/')[5]
        
        # If preprocessed eeg file already exists, go to next file
        save_directory = epochs_path + '/'
        preproc_report_path = report_path + f'{subjectname}_{session_info}_10s_NICE-PREPROC_blocksAfter_report.html'
        if os.path.exists(preproc_report_path):
            continue
        
        # Create MNE report
        report = mne.Report(title='10s_NICE_preproc_blocksafter_' + subjectname+'_'+session_info)
        
        # 1). READ RAW FILE ACCORDING TO SETUP INFO
        
            # Obtain 'setup' info
        matching_row = setups[(setups['Subject'] == subjectname) & (setups['Session'] == session_info)]
        if not matching_row.empty:
           setup = matching_row['Setup'].iloc[0]
        
            # Read eeg file
        if subset_file.split(sep='/')[-1][-4:] == '.fif':
            print('This concatenated file .fif has already been read by specific NICE function, can read them with MNE basic func')
            raw = mne.io.read_raw_fif(subset_file, preload=True)
        else:
            if setup == 'egiraw':
                raw = _read_lg_raw_egi_files(subset_file, preload=True) 
                    
            elif setup == 'egiraw_a':
                raw = _read_lga_raw_egi(subset_file,config_params={})
                
            elif setup == 'egi256':
                raw = _read_lga_mff_egi(subset_file,config_params={}) 


            # Setting the montage
        HCGSN256_montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
        # to_drop = [x for x in raw.ch_names if x not in HCGSN256_montage.ch_names]
        # raw.drop_channels(to_drop)
        raw.set_montage(HCGSN256_montage)
        raw.info['description'] = 'GSN-HydroCel-256'
        
            # Add raw data to report
        report.add_raw(raw,title='Raw data '+subjectname, psd=True)
        fig=raw.compute_psd(method='welch', fmin=0, fmax=np.inf).plot(show=False)
        report.add_figure(fig,title='PSD raw  '+subjectname)
        
        
        # 2). SMALL PREPROCESSING STEPS
        
            # Remove face and neck channels from the data
        raw.drop_channels(channels_to_remove)
        
            # Create new montage without periph chans
        channels_to_keep = [ch for ch in HCGSN256_montage.ch_names if ch not in channels_to_remove]
    
        dig_dict= HCGSN256_montage.get_positions()
        dig_names = HCGSN256_montage._get_dig_names()
    
        all_ch_pos = dig_dict['ch_pos']
           
        ch_pos = {key: value for key, value in all_ch_pos.items() if key not in channels_to_remove}
        nasion = dig_dict['nasion']
        lpa = dig_dict['lpa']
        rpa = dig_dict['rpa']
        hsp = dig_dict['hsp']
        hpi = dig_dict['hpi']
    
        new_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=nasion, lpa=lpa,
                                                    rpa=rpa, hsp=hsp, hpi=hpi, coord_frame='unknown')
        # new_montage.plot()
        # HCGSN256_montage.plot()
    
            # Set the new montage to the raw data
        raw.set_montage(new_montage)
        raw.info['description'] = 'GSN-HydroCel-256_minus_periph'  # Update description if needed
        
            # Add raw data after periph elec drop to report
        report.add_raw(raw,title='Raw data after periph elec removal '+subjectname, psd=True)  
        fig=raw.compute_psd(method='welch', fmin=0, fmax=np.inf).plot(show=False)
        report.add_figure(fig,title='PSD raw after periph elec removal '+subjectname)
        
        
            # Filtering #put 0.1 (now 0.5)
        config_params={}
        _egi_filter(raw, config_params, summary=True, n_jobs=1) #[0.1-45] iir 6/8 + 50-100 Notch fft ([0.5-45 in NICE orig version]
        
            # Add filtered raw data to report
        report.add_raw(raw,title='Raw data after periph elec removal and FILTERING; [0.1-45] iir 6/8th butter + 50-100 Notch fft '+subjectname, psd=True)
        fig=raw.compute_psd(method='welch', fmin=0, fmax=np.inf).plot(show=False)
        report.add_figure(fig,title='PSD raw after periph elec removal and FILTERING; [0.1-45] iir 6/8th butter + 50-100 Notch fft '+subjectname)
        
        
        # 3). READ EVENTS AND SAVE LG BLOCKS TIMINGS
        
        # i) Read events of interest from already saved .txt file
        
            # Appropriate LG events are loaded (2014 concat events and 1st 9 & 10th trigg have been previously removed)
        #events_file= glob.glob(data_path + '/Events/**/' + f'{subjectname}_{session_info}_filt-LG_eve.txt')
        events_file= glob.glob(data_path + '/1b_read_select/Events/**/' + f'{subjectname}_{session_info}_filt-LG_eve.txt')
        
        events = mne.read_events(events_file[0]) 
        #events=mne.find_events(raw)
    
            # Visualize Local-Global events
        lg_events_dict = {'HSTD': 10 , 'HDVT': 20, 'LSGS': 30, 'LSGD': 40, 
                              'LDGD': 50, 'LDGS':60}
        fig_events=mne.viz.plot_events(events,event_id=lg_events_dict,show=False)
        report.add_events(events, title='Local-Global Events ' + subjectname, event_id=lg_events_dict, sfreq=250)
        
            # Identify blocks Start and End
        sfreq = raw.info['sfreq']  #250Hz
        events_times_seconds = events[:, 0] / sfreq
        
        # Calculate the time differences between events in seconds
        events_ISI = np.diff(events[:, 0]) / sfreq #diff btw n & n+1
        events_ISI = np.append(events_ISI, 0)  # Append a 0 for the last event, which has no subsequent event
        
        # Add the new columns to the events array
        events_with_times = np.column_stack((events, events_times_seconds, events_ISI ))
        
        Bl_End = []
        Bl_Start = []
        
        for i in range(len(events_ISI)):
            if events_ISI[i] >= 5: #PUT 6 ?
                # Store the index of the row where ISI is 5 seconds or more
                Bl_End.append(i)
                events_with_times[i, 2] = 2 #end
                Bl_Start.append(i+1)
                events_with_times[i+1, 2] = 1 #start
    
        # Change first and last events id
        events_with_times[0, 2] = 1 #first
        events_with_times[-1, 2] = 2 #last
        
        block_events=mne.pick_events(events_with_times, include=[1,2])
        blocks_events_dict = {'Block_Start': 1, 'Block_End': 2}
        
        fig_blocks_events=mne.viz.plot_events(block_events,event_id=blocks_events_dict,show=True)
        report.add_events(block_events, title='Local-Global BLOCKS Events ' + subjectname, event_id=blocks_events_dict, sfreq=250)
    
    
        # Extract start and end sample indices for blocks
        block_start_samp = block_events[block_events[:, 2] == 1][:, 0]  # Event ID 1 for Block_Start
        block_end_samp = block_events[block_events[:, 2] == 2][:, 0]  # Event ID 2 for Block_End
        
        block_times_df = pd.DataFrame({
            'block_start_samp': block_start_samp,
            'block_end_samp': block_end_samp
        })
        
        # Save the DataFrame to a CSV file
        csv_save_path = f'{timings_path}/{subjectname}_{session_info}_block_times.csv'
        block_times_df.to_csv(csv_save_path, index=False)
    
        # Compute block duration
        block_dur = (block_end_samp-block_start_samp)/raw.info['sfreq']
        block_dur_diff = np.insert(np.diff(block_dur), 0, 0) #this var may come from var of ISI av per block
        block_dur_min = block_dur/60
        n_10s_epochs = block_dur/10
    
        if block_dur.shape[0] == len(block_dur_min):
            block_dur = np.column_stack((block_dur_min, block_dur, block_dur_diff, n_10s_epochs))
        else:
            print("The number of blocks does not match the number of rows in the events array.")
        
        # Convert the numpy array to a pandas DataFrame
        df_block_dur = pd.DataFrame(block_dur)
        df_block_dur.columns = [
            'Dur block (min)',  
            'Dur block (s)', 
            'Diff dur block (s)', 
            'n 10s epochs', 
        ]
    
        # RUN NICE PREPROCESSING ON ALL EPOCHS 
        epochs_clean = preprocess_NICE(raw,config_params={})
        
        # Save NICE preprocessing output (clean Epoch object, not splitted by block yet)
        epochs_clean_filename = f'{epochs_path}{subjectname}_{session_info}-LG_all10sepochs_postNICE_epo.fif'
        epochs_clean.save(epochs_clean_filename,overwrite=True)
    
    
        # Initialize lists to store epochs for each block
        block_epochs = [[] for _ in range(len(block_start_samp))]
        block_dataframes = [pd.DataFrame(columns=['epoch_start', 'epoch_end', 'in_block', 'block_start', 'block_end']) for _ in range(len(block_start_samp))]
        
        # Initialize counters to track the number of epochs in each block
        num_epochs_in_block = [0 for _ in range(len(block_start_samp))]
        
        # Check if epochs fall within block start and end samples  
        for i in range(len(epochs_clean)):
            epoch_start = epochs_clean.events[i, 0]
            epoch_end = epoch_start + 2500  # Assuming each epoch is 10 seconds long at 250Hz sampling rate
            for j, (start, end) in enumerate(zip(block_start_samp, block_end_samp)):
                in_block = start <= epoch_start <= end and epoch_end <= end
                block_dataframes[j] = block_dataframes[j].append({
                    'epoch_start': epoch_start,
                    'epoch_end': epoch_end,
                    'in_block': in_block,
                    'block_start': start,
                    'block_end': end
                }, ignore_index=True)
                if in_block:
                    block_epochs[j].append(epochs_clean[i])
                    num_epochs_in_block[j] += 1  # Increment the counter for this block
        
        # Add the number of epochs information to each DataFrame
        for j in range(len(block_dataframes)):
            block_dataframes[j]['num_epochs_in_block'] = num_epochs_in_block[j]
        
       
        # Save block_dataframes to CSV files using the correct block number
        for idx, df in enumerate(block_dataframes):
            block_df_save_path = f'{timings_path}/{subjectname}_{session_info}_block_{idx + 1}_epochs.csv'
            df.to_csv(block_df_save_path, index=False)
            
        # Create and save the final summary table
        nepochs_blocks_aftNICE = pd.DataFrame({
            'block_number': range(1, len(num_epochs_in_block) + 1),
            'num_epochs': num_epochs_in_block
        })
        
        # Save n° of epochs per block on which can compute waves 
        nepochs_blocks_aftNICE_path = f'{timings_path}/{subjectname}_{session_info}_nepochs_perblock.csv'
        nepochs_blocks_aftNICE.to_csv(nepochs_blocks_aftNICE_path, index=False)
        
        # Create one Epoch object for each block
        concatenated_epochs = [
        mne.concatenate_epochs(block) if block else 'no clean epochs for this block'
        for block in block_epochs]
        
        # # Save the concatenated epochs
        # for i, epochs_block in enumerate(concatenated_epochs):
        #     block_save_path = f'{epochs_path}{subjectname}_{session_info}_block_{i+1}_epo.fif'
        #     epochs_block.save(block_save_path, overwrite=True)
            
        # Save each concatenated_epochs if it's not a string
        for idx, block_epochs in enumerate(concatenated_epochs):
            if isinstance(block_epochs, str):
                print(f"Block {idx + 1}: {block_epochs}")
                continue
            blocks_save_path = f'{epochs_path}/{subjectname}_{session_info}_block_{idx + 1}_epochs-epo.fif'
            block_epochs.save(blocks_save_path, overwrite=True)
        
        # # Example of adding the concatenated epochs to the report (optional)
        # for i, epochs_block in enumerate(concatenated_epochs):
        #     report.add_epochs(epochs_block, title=f'Concatenated Epochs Block {i+1}', psd=False)
    
            
    # =============================================================================
    #     # 5). CONCAT ALL PREPROC BLOCKS EPOCHS
    #     concat_blocks_preproc_ep = mne.concatenate_epochs(concatenated_epochs, add_offset=True) #can put False
    #     concat_blocks_filename = f'{epochs_path}{subjectname}_{session_info}-LG_blocks_concatAll_NICE_epo.fif'
    #     concat_blocks_preproc_ep.save(concat_blocks_filename, overwrite=True)
    #     
    #     concat_blocks_events = concat_blocks_preproc_ep.events
    #     concat_events_filename = f'{epochs_path}{subjectname}_{session_info}-LG_blocks_concatAll_events_eve.txt'
    #     mne.write_events(concat_events_filename, events, overwrite=True)
    # 
    # 
    #         # add plots to report
    #     fig=concat_blocks_preproc_ep.plot_drop_log(show=False)
    #     report.add_figure(fig,title='All blocks drop log (concat after NICE contiuous then split per block)'+subjectname)
    #     
    #     report.add_epochs(concat_blocks_preproc_ep, title=f'Concatenated Epochs Block {i+1}', psd=False)
    #     
    #     fig=concat_blocks_preproc_ep.plot_image(show=False)
    #     report.add_figure(fig,title='All blocks image (concat after NICE contiuous then split per block)'+subjectname)
    # 
    # =============================================================================
        
    # 6). SAVE PATIENT PREPROCESSING REPORT
        report.save(report_path +f'{subjectname}_{session_info}_10s_NICE-PREPROC_blocksAfter_report.html', 
                    overwrite=True, open_browser=False)
        
        
        # 7). SAVE CLEANING INFO THIS SUBJ
        adaptative_results_SUBJ_df = pd.DataFrame(adaptative_results_SUBJ)
        adaptative_results_SUBJ_df.to_csv(report_path + '/'+ f'{subjectname}_{session_info}_Adaptative_results_blocksAfter.csv', index=False)
        adaptative_results_SUBJ = []
    
        check_epochs_cleaning_SUBJ_df = pd.DataFrame(check_epochs_cleaning_SUBJ)
        check_epochs_cleaning_SUBJ_df.to_csv(report_path + '/'+ f'{subjectname}_{session_info}_Check_epochs_drop_blocksAfter.csv', index=False)
        check_epochs_cleaning_SUBJ = []
       
    except Exception as e:
        # Append error details to error_files list
        error_files.append({
            'subset_file': subset_file,
            'subjectname': subjectname,
            'session_info': session_info,
            'error': str(e)
        })
        # Print the error message for debugging
        print(f"An error occurred with file {subset_file}: {e}")
        # Save error files df as .csv
        error_file_subj_df = pd.DataFrame(error_files)
        error_file_subj_df.to_csv(save_path + f"/error_file{subjectname}_{session_info}", index=False)

# Save error files df as .csv
error_files_all_df = pd.DataFrame(error_files)
error_files_all_df.to_csv(save_path + '/error_files_all.csv', index=False)
    
# 8). ALL SUBJ - SAVE CLEANING INFO
adaptative_results_df = pd.DataFrame(adaptative_results)
adaptative_results_df.to_csv(report_path + '/Adaptative_results_blocksAfter.csv', index=False)

check_epochs_cleaning_df = pd.DataFrame(check_epochs_cleaning)
check_epochs_cleaning_df.to_csv(report_path + '/Check_epochs_drop_blocksAfter.csv', index=False)

#If kernel crashed and had to run the code several times, those df won't contain all subj info
# so need to load each subject df and concatenate them

    # 1). Adaptative results
all_adapt_csv = glob.glob(report_path+'*Adaptative*.csv')
all_adapt_df = []

# Loop through the filenames and read each file into a dataframe
for table in all_adapt_csv:
    try:
        # Read the csv file and append the dataframe to the list
        df = pd.read_csv(table)
        all_adapt_df.append(df)
    except FileNotFoundError as e:
        print(f"File not found: {table}", e)

all_adapt_results = pd.concat(all_adapt_df, ignore_index=True)
all_adapt_results.to_csv(report_path + '/ALL-SUBJ_Adaptative_results_blocksAfter.csv', index=False)

    # 2). Check epochs cleaning
all_check_epochs_csv = glob.glob(report_path+'*Check*.csv')
all_check_epochs_df = []

# Loop through the filenames and read each file into a dataframe
for table in all_check_epochs_csv:
    try:
        # Read the csv file and append the dataframe to the list
        df = pd.read_csv(table)
        all_check_epochs_df.append(df)
    except FileNotFoundError as e:
        print(f"File not found: {table}", e)

all_check_epochs_results = pd.concat(all_check_epochs_df, ignore_index=True)
all_check_epochs_results.to_csv(report_path + '/ALL-SUBJ_Check_epochs_drop_blocksAfter.csv', index=False)
 