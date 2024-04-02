#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:42:27 2024

@author: arthurlecoz

SLHIP_config_ALC.py

"""
# %% Paths

rootpath = "/Volumes/DDE_ALC/PhD/SLHIP"

rawDataPath = rootpath + "/00_Raw"
behavDataPath = rootpath + "/01_BehavData"
cleanDataPath = rootpath + "/02_Preproc"


# %% Functions

    #### Utils

def find_closest_value(value1, arr2):
    import numpy as np
    index_closest_value = []
    closest_value = []
    d_array = [
        abs(value1 - value2) for i2, value2 in enumerate(arr2)
               ]
    index_closest_value.append(np.argmin(d_array))
    closest_value.append(arr2[np.argmin(d_array)])
    
    return(index_closest_value, closest_value)

def display_matrices_info(mat_type):
    """
    Parameters
    ----------
    mat_type : str : 'test_res' or 'probe_res'
        Display detail for either part of the matrix

    Returns
    -------
    None.

    """
    
    assert mat_type in ["test_res", "probe_res"], "Careful, mat_type can only be 'test_res' or 'probe_res'"
    if mat_type == "test_res" :
        print('''
        - test_res (matrices des résultats du SART, 1 ligne = 1 essai)
    
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
        ''')
    elif mat_type == 'probe_res' :
        print('''
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
    
        Probe Q1 : Etat d'esprit juste avant l'interruption.
            Ans :   1 - J'étais concentré-e sur la tâche 
                    2 - Je pensais à autre chose
                    3 - Je ne pensais à rien
                    4 - Je ne me souviens pas
    
        Probe Q2 : Sur quoi étiez-vous concentré-e?
                Sur quelque-chose:
            Ans :   1 - Dans la pièce
                    2 - Une pensee venant de vous
                    3 - Une pensée liée a la tâche
                    4 - Une hallucination dans mon champs de vision
    
        Probe Q3 : Notez votre vigilance
            Ans : 1 - 9 
        ''')

    #### Pipeline
    
import mne
import json

def load_and_preprocess_data(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    file_format = config['file_format']
    settings = config['load_and_preprocess']
    supported_formats = ['BrainVision', 'EDF', 'BDF', 'CTF']
    
    # Ensure the file format is supported
    assert file_format in supported_formats, f"File format {file_format} not supported."

    # Load the raw data based on the file format
    if file_format == "BrainVision":
        raw = mne.io.read_raw_brainvision(settings['file_path'], preload=True)
    elif file_format == "EDF":
        raw = mne.io.read_raw_edf(config['additional_formats']['EDF']['file_path'], preload=True)
    elif file_format == "BDF":
        raw = mne.io.read_raw_bdf(config['additional_formats']['BDF']['file_path'], preload=True)
    elif file_format == "CTF":
        raw = mne.io.read_raw_ctf(config['additional_formats']['CTF']['file_path'], preload=True)
    
    # Apply preprocessing steps
    raw.set_montage(mne.channels.make_standard_montage(settings['montage']), on_missing='ignore')
    raw.filter(settings['l_freq'], settings['h_freq'], fir_design='firwin')
    raw.notch_filter(settings['notch_freq'], filter_length='auto', phase='zero')
    
    # Return the preprocessed data
    return raw


def load_and_preprocess_data(file_path, montage='standard_1020', l_freq=0.1, h_freq=100, notch_freq=50, channel_types=None):
    """
    Load and preprocess EEG data, with the ability to specify channel types.

    Parameters:
    - file_path: str, path to the EEG file.
    - montage: str, name of the montage to use.
    - l_freq: float, high-pass filter frequency.
    - h_freq: float, low-pass filter frequency.
    - notch_freq: float, frequency for notch filtering.
    - channel_types: dict, a dictionary mapping channel names to their types (e.g., {'EOG': ['EOG 061', 'EOG 062']}).

    Returns:
    - mne.io.Raw: preprocessed raw data.
    """
    # Load the raw data
    raw = mne.io.read_raw_brainvision(file_path, preload=True)

    # Set channel types if provided
    if channel_types:
        for ch_type, ch_names in channel_types.items():
            raw.set_channel_types({ch_name: ch_type for ch_name in ch_names})

    # Set montage
    raw.set_montage(
        mne.channels.make_standard_montage(montage), on_missing='ignore'
        )
    
    # Resample, re-reference, and filter
    raw.resample(250)
    raw.set_eeg_reference('average')
    raw.filter(l_freq, h_freq, fir_design='firwin')
    raw.notch_filter(notch_freq, filter_length='auto', phase='zero')
    
    return raw


def handle_events(raw, merge_dict=None):
    """
    Extract and handle events from raw EEG data.

    Parameters:
    - raw: mne.io.Raw, the preprocessed raw data.
    - merge_dict: dict, optional; 
    dictionary with keys as new event IDs and values as lists of event IDs to be merged.

    Returns:
    - events: numpy.ndarray; the events array after handling.
    - event_id: dict; the event dictionary after handling.
    """
    import mne
    # Extract events
    events, event_id = mne.events_from_annotations(raw)
    
    # Merge events if merge_dict is provided
    if merge_dict is not None:
        for new_id, old_ids in merge_dict.items():
            events = mne.merge_events(
                events, 
                old_ids, 
                new_id, 
                replace_events=True
                )

    return events, event_id

def create_epochs(
        raw, 
        events, 
        event_id, 
        tmin=-0.2, 
        tmax=1.2, 
        baseline=None, 
        reject=None, 
        log_errors=True
        ):
    """
    Create epochs from raw EEG data, with additional handling for common preprocessing needs.

    Parameters:
    - raw: mne.io.Raw, the preprocessed raw data.
    - events: numpy.ndarray, the events array.
    - event_id: dict, the event dictionary.
    - tmin: float, start time before event.
    - tmax: float, end time after event.
    - baseline: tuple, the baseline period.
    - reject: dict, rejection parameters.
    - log_errors: bool, whether to log errors encountered during epoch creation.

    Returns:
    - epochs: mne.Epochs, the created epochs or None if an error occurs.
    """
    from mne import Epochs
    try:
        epochs = Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True, reject=reject, event_repeated='drop')
        return epochs
    except Exception as e:
        if log_errors:
            print(f"Error creating epochs: {e}")
        return None



def automatic_ica_and_report(
        raw, 
        sub_id, 
        output_dir, 
        n_components=15, 
        l_freq=1.0, 
        eog_ch=None, 
        ecg_ch=None, 
        iclabel_threshold=0.7
        ):
    """
    Perform ICA to identify and remove artifacts, and generate a comprehensive report.

    Parameters:
    - raw: Instance of Raw.
    - sub_id: Subject identifier.
    - output_dir: Directory to save the outputs.
    - n_components: Number of components for ICA.
    - l_freq: High-pass filter cutoff before ICA.
    - eog_ch: EOG channel name(s) for artifact detection.
    - ecg_ch: ECG channel name for artifact detection.
    - iclabel_threshold: Threshold for excluding components based on ICLabel classification.
    """
    import os
    import mne
    from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
    from mne_icalabel import label_components
    # Preprocess: High-pass filter
    filt_raw = raw.copy().filter(l_freq=l_freq, h_freq=None)

    # Fit ICA
    ica = ICA(n_components=n_components, random_state=97)
    ica.fit(filt_raw)

    # Automatic EOG/ECG artifact detection and labeling with ICLabel
    if eog_ch:
        ica.exclude.extend(ica.find_bads_eog(filt_raw, ch_name=eog_ch)[0])
    if ecg_ch:
        ica.exclude.extend(ica.find_bads_ecg(filt_raw, ch_name=ecg_ch)[0])

    # ICLabel for further classification and exclusion
    ica_labels = label_components(filt_raw, ica, method='iclabel')
    for idx, (label, score) in enumerate(zip(ica_labels['labels'], ica_labels['scores'])):
        if label not in ['brain', 'other'] and score > iclabel_threshold:
            ica.exclude.append(idx)

    # Apply ICA
    ica.apply(filt_raw)

    # Save ICA object and exclusion info
    ica.save(os.path.join(output_dir, f'{sub_id}-ica.fif'))
    with open(os.path.join(output_dir, f'{sub_id}-ica-exclude.txt'), 'w') as f:
        f.write('\n'.join(map(str, ica.exclude)))

    # Generate report
    report = mne.Report(title=f'ICA Report for {sub_id}')
    report.add_ica(ica=ica, title='ICA Components', inst=raw)
    if eog_ch or ecg_ch:
        report.add_epochs(create_eog_epochs(filt_raw) if eog_ch else create_ecg_epochs(filt_raw), title='Artifact Epochs')
    report.save(os.path.join(output_dir, f'report_{sub_id}.html'), overwrite=True)

    return ica


def generate_flexible_report(
        raw, epochs, ica, sub_id, output_dir, 
        compare_evoked=True, 
        include_raw=True, 
        include_psd=True, 
        include_ica=True):
    """
    Generate a comprehensive HTML report with flexible content inclusion and ICA comparison.

    Parameters:
    - raw: mne.io.Raw, the original raw data.
    - epochs: mne.Epochs, the epochs before ICA application.
    - ica: mne.preprocessing.ICA, the ICA object after fitting.
    - sub_id: str, subject identifier for file naming.
    - output_dir: str, directory to save the report.
    - compare_evoked: bool, whether to include comparison of evoked responses before and after ICA.
    - include_raw: bool, whether to include the raw data section.
    - include_psd: bool, whether to include power spectral density of the raw data.
    - include_ica: bool, whether to include ICA components and classification.

    Returns:
    - None, saves the report to the specified path.
    """
    from mne import Report
    report = Report(
        title=f'EEG Preprocessing Report for Subject {sub_id}', verbose=True
        )
    
    if include_raw:
        report.add_raw(raw=raw, title='Raw Data', psd = True)
    if include_ica:
        ecg_evks = epochs.copy().average(picks = 'ECG')
        eog_evks = epochs.copy().average(picks = ['eog'])
        eog_idx, eog_scores = ica.find_bads_eog(
            epochs, 
            ch_name='VEOG', 
            threshold='auto',
            l_freq=1, 
            h_freq=10, 
            measure='correlation')
        ecg_idx, ecg_scores = ica.find_bads_ecg(
            epochs, 
            ch_name='ECG', 
            threshold='auto',
            l_freq=8, 
            h_freq=16, 
            method='ctps', 
            measure='correlation', 
            verbose=None)
        
        report.add_ica(
            ica=ica, 
            title='ICA Components', 
            inst=epochs, 
            ecg_evoked=ecg_evks,
            eog_evoked=eog_evks,
            )
    
    # Compare evoked responses before and after ICA
    if compare_evoked:
        evoked_before = epochs.average()
        epochs_clean = ica.apply(epochs.copy())
        evoked_after = epochs_clean.average()
        
        fig_before = evoked_before.plot(show=False)
        fig_after = evoked_after.plot(show=False)
        
        report.add_figure(fig_before, title='Evoked Response Before ICA')
        report.add_figure(fig_after, title='Evoked Response After ICA')
    
    report_path = f"{output_dir}/{sub_id}_report.html"
    report.save(report_path, overwrite=True)
    print(f"Report saved to {report_path}")


