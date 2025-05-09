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
cleanDataPath = rootpath + "/01_Preproc"
powerPath = rootpath + "/03_Power"
wavesPath = rootpath + "/04_Waves"

config_dict = {
      "file_format": "BrainVision",
      "load_and_preprocess": {
        "referencing" : "average",
        "montage": "standard_1020",
        "l_freq": 0.1,
        "h_freq": 100,
        "notch_freq": [50, 100],
        "f_resample" : 256,
        "channel_types" : {
            'VEOG' : 'eog', 'HEOG' : 'eog', 'ECG' : 'ecg', 'RESP' : 'resp'
            },
        "n_jobs" : -1
      },
      "channel_interpolation": {
        "method": "automatic"
      },
      "ica": {
        "n_components": 15,
        "l_freq": 1.0,
      }
    }

eeg_channels = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
    'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
    'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
    'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1',
    'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6',
    'AF8', 'AF4', 'F2', 'Iz']

incomplete_subjects = ["HS_001", "HS_004"]

# %% Functions

    #### Utils

def find_closest_value(value1, arr2):
    """
    Find the closest value comparing a value to an array and returns index 
    and value of the closest value

    Parameters
    ----------
    value1 : float
        Value to find the closest element to.
    arr2 : np.array
        Array in which you want to find a close element.

    Returns
    -------
    index_closest_value : int
        Index of the closest value.
    closest_value : float
        Closest value.

    """
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

def load_and_preprocess_data(file_path):
    """
    Load and minimally preprocess raw EEG data based on the settings in this script (config).

    Parameters
    ----------
    file_path : str
        Path to the data file.

    Returns
    -------
    raw : mne.io.Raw
        Minimally processed Raw object from the path.
    """
    # Load configuration
    
    file_format = config_dict['file_format']
    settings = config_dict['load_and_preprocess']
    supported_formats = ['BrainVision', 'EDF', 'FIF']
    channel_types = settings["channel_types"]
    
    # Ensure the file format is supported
    assert file_format in supported_formats, f"File format {file_format} not supported."

    # Load the raw data based on the file format
    if file_format == "BrainVision":
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    elif file_format == "EDF":
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif file_format == "FIF":
        raw = mne.io.read_fif(file_path, preload=True)
    
    # Apply preprocessing steps
    raw.set_channel_types(channel_types)
    mne.set_eeg_reference(raw, settings['referencing'], copy = False)
    raw.set_montage(
        mne.channels.make_standard_montage(settings['montage']), 
        on_missing='ignore'
        )
    raw.filter(
        settings['l_freq'], 
        settings['h_freq'], 
        fir_design='firwin',
        picks = ["eeg", "eog"],
        n_jobs = settings["n_jobs"]
        )
    raw.notch_filter(
        settings['notch_freq'], 
        filter_length='auto', 
        phase='zero', 
        n_jobs = settings["n_jobs"]
        )
    raw.resample(settings['f_resample'])
    
    # Return the preprocessed data
    return raw

def handle_events(raw, merge_dict=None):
    """
    Extract and handle events from raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed raw data.
    merge_dict : dict, optional
        Dictionary with keys as new event IDs and values as lists of event IDs to be merged. 
        The default is None.

    Returns
    -------
    events : numpy.ndarray
        The events array after handling.
    event_id : dict
        The event dictionary after handling.

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

def automatic_ica(
        eeg_data, 
        sub_id, 
        output_dir, 
        n_components=15, 
        l_freq=1.0, 
        v_eog_ch=None, 
        h_eog_ch=None, 
        ecg_ch=None, 
        icalabel=False
        ):
    """
    Perform ICA on EEG data and automatically exclude artifacts.
 
    Parameters
    ----------
    eeg_data : mne.io.Raw or mne.Epochs
        Instance of Raw or Epochs containing the EEG data.
    sub_id : str
        Subject identifier.
    output_dir : str
        Directory to save the ICA outputs.
    n_components : int, optional
        Number of components for ICA. The default is 15.
    l_freq : float, optional
        High-pass filter cutoff frequency before ICA. The default is 1.0.
    v_eog_ch : str, optional
        Vertical EOG channel name for artifact detection. The default is None.
    h_eog_ch : str, optional
        Horizontal EOG channel name for artifact detection. The default is None.
    ecg_ch : str, optional
        ECG channel name for artifact detection. The default is None.
    icalabel : bool, optional
        Whether to use ICLabel for further classification and exclusion of components. 
        The default is False.
 
    Returns
    -------
    ica : mne.preprocessing.ICA
        ICA object fitted on the EEG data.
    """
    import os
    # import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    # Preprocess: High-pass filter
    assert type(eeg_data) in [mne.io.brainvision.brainvision.RawBrainVision, mne.epochs.Epochs], "The class of the eeg_data is not supported..."
    filt_data = eeg_data.copy().filter(
        l_freq=l_freq, h_freq=None, n_jobs = -1
        )

    # Fit ICA
    
    ica = ICA(
        n_components=n_components, 
        random_state=97,
        method='infomax', 
        fit_params=dict(extended=True)
        )
    ica.fit(filt_data)

    # Automatic EOG/ECG artifact detection and labeling with ICLabel
    if h_eog_ch:
        _, weights_heog=ica.find_bads_eog(filt_data, ch_name = 'HEOG')
        bads_heog = [i for i, weight in enumerate(weights_heog)
                    if weight > weights_heog.mean()+3*weights_heog.std()]
        for i in bads_heog:
            ica.exclude.append(i)
    if v_eog_ch:
        _, weights_veog=ica.find_bads_eog(filt_data, ch_name = 'VEOG')
        bads_veog = [i for i, weight in enumerate(weights_veog)
                    if weight > weights_veog.mean()+1*weights_veog.std()]
        for i in bads_veog:
            ica.exclude.append(i)
    if ecg_ch:
        _, weights_ecg=ica.find_bads_ecg(filt_data, "ECG")
        bads_ecg = [i for i, weight in enumerate(weights_ecg)
                    if weight > weights_ecg.mean()+4*weights_ecg.std()
                    and weight > .1]
        for i in bads_ecg:
            ica.exclude.append(i)
    
    if icalabel or not any([h_eog_ch, v_eog_ch, ecg_ch]):
    # ICLabel for further classification and exclusion
        ica_labels = label_components(filt_data, ica, method='iclabel')
        for idx, (label, score) in enumerate(
                zip(ica_labels['labels'], ica_labels['y_pred_proba'])):
            if label not in ['brain', 'other']:
                ica.exclude.append(idx)

    # Save ICA object and exclusion info
    if output_dir : 
        ica.save(os.path.join(output_dir, f'{sub_id}-ica.fif'), overwrite = True)
        with open(os.path.join(output_dir, f'{sub_id}-ica-exclude.txt'), 'w') as f:
            f.write('\n'.join(map(str, ica.exclude)))
    return ica


# def generate_flexible_report(
#         raw, epochs, ica, sub_id, output_dir, 
#         compare_evoked=True, 
#         include_raw=True, 
#         include_psd=True, 
#         include_ica=True):
#     """
#     Generate a comprehensive HTML report with flexible content inclusion and ICA comparison.

#     Parameters:
#     - raw: mne.io.Raw, the original raw data.
#     - epochs: mne.Epochs, the epochs before ICA application.
#     - ica: mne.preprocessing.ICA, the ICA object after fitting.
#     - sub_id: str, subject identifier for file naming.
#     - output_dir: str, directory to save the report.
#     - compare_evoked: bool, whether to include comparison of evoked responses before and after ICA.
#     - include_raw: bool, whether to include the raw data section.
#     - include_psd: bool, whether to include power spectral density of the raw data.
#     - include_ica: bool, whether to include ICA components and classification.

#     Returns:
#     - None, saves the report to the specified path.
#     """
#     from mne import Report
#     report = Report(
#         title=f'EEG Preprocessing Report for Subject {sub_id}', verbose=True
#         )
    
#     if include_raw:
#         report.add_raw(raw=raw, title='Raw Data', psd = True)
#     if include_ica:
#         ecg_evks = epochs.copy().average(picks = 'ECG')
#         eog_evks = epochs.copy().average(picks = ['eog'])
#         eog_idx, eog_scores = ica.find_bads_eog(
#             epochs, 
#             ch_name='VEOG', 
#             threshold='auto',
#             l_freq=1, 
#             h_freq=10, 
#             measure='correlation')
#         ecg_idx, ecg_scores = ica.find_bads_ecg(
#             epochs, 
#             ch_name='ECG', 
#             threshold='auto',
#             l_freq=8, 
#             h_freq=16, 
#             method='ctps', 
#             measure='correlation', 
#             verbose=None)
        
#         report.add_ica(
#             ica=ica, 
#             title='ICA Components', 
#             inst=epochs, 
#             ecg_evoked=ecg_evks,
#             eog_evoked=eog_evks,
#             )
    
#     # Compare evoked responses before and after ICA
#     if compare_evoked:
#         evoked_before = epochs.average()
#         epochs_clean = ica.apply(epochs.copy())
#         evoked_after = epochs_clean.average()
        
#         fig_before = evoked_before.plot(show=False)
#         fig_after = evoked_after.plot(show=False)
        
#         report.add_figure(fig_before, title='Evoked Response Before ICA')
#         report.add_figure(fig_after, title='Evoked Response After ICA')
    
#     report_path = f"{output_dir}/{sub_id}_report.html"
#     report.save(report_path, overwrite=True)
#     print(f"Report saved to {report_path}")


