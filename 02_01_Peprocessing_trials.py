# -*- coding: utf-8 -*-
"""
author: Arthur Le Coz

01_01_Preprocessing_trials.py

******************************************************************************

This pipeline was built based on data collected during a SART.
This allow us to get short traditional epochs to clean first before doing ICA.
With ICA label, and possibly with EOG and ECG channel, ICA sources containing:
blinks, eye movements, muscle movement, heartbeat, and other artifact will be
automatically removed from the recordings.

At the end of the script, will be saved:
    For each subject : 
    * Trial Epochs clean (AutoReject + ICAed) on Go and NoGo events
    * Probes epochs cleaned (Threshold 300µV + ICAed) with metadata :
        in the metadata of the probe epochs will be found :
    [sub_id, subtype, nblock, nprobe, mindstate, voluntary, sleepiness]
    
    Across all subjects : MNE.Reports :
        - Events informations
        - Auto Reject informations
        - ERP informations before and after cleaning
        - ICA sources and topographies (bads in lightgrey)
        
******************************************************************************
        
"""
# %% Packages, Paths, Variables
#### Packages
import os
import numpy as np
import pandas as pd
import mne
from glob import glob
from autoreject import AutoReject, Ransac
import SLHIP_config_ALC as cfg 
import matplotlib.pyplot as plt
import warnings

import matplotlib
matplotlib.use('Agg')
# matplotlib.use('QtAgg')

#### Paths
if 'arthur' in os.getcwd():
    path_root='/Volumes/DDE_ALC/PhD/SLHIP'
else:
    path_root='your_path'

path_data=os.path.join(path_root, '00_Raw')
path_preproc=os.path.join(path_root, '01_Preproc')

if os.path.exists(os.path.join(path_data, "reports"))==False:
    os.makedirs(os.path.join(path_data, "reports"))
if os.path.exists(os.path.join(path_data, "intermediary"))==False:
    os.makedirs(os.path.join(path_data,"intermediary"))
    
#### Variables    

# Paths to the EEG files, here brainvision files
files = glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))
# amount of cpu used for functions with multiprocessing :
n_jobs = 4

# Events handling to simplify script
ms_dic = {
    177 : 'ON',
    178 : 'MW',
    179 : 'DISTRACTED',
    180 : 'HALLU',
    181 : 'MB',
    182 : 'FORGOT'
    }
vol_dic = {
    177 : 1,
    178 : 0,
    }
sleepi_dic = {
    177 : 1,
    178 : 2,
    179 : 3,
    180 : 4,
    181 : 5,
    182 : 6,
    183 : 7,
    184 : 8,
    185 : 9
    }

# Event of interest here, for the SART : Go and NoGo trials 
go_id=101
nogo_id=100
conditions=[str(go_id),str(nogo_id)]
merge_dict = {
    101 : [65,66,68,69,70,71,72,73],
    100 : [67]
    }

# Epochs threshold to have epochs with : 300µV > PTP amplitude > 1µV
flat_criteria = dict(eeg=1e-6)
threshold = 300e-6

# AR config
ar_dic = dict(
    n_interpolates = np.array([1, 8, 64]),
    consensus_percs = np.linspace(0, 1.0, 11),
    picks = 'eeg',
    thresh_method = "bayesian_optimization",
    random_state = 42
    )

# ICA config
ica_dic = dict(
    n_components = 15,
    l_freq = 1.0,
    h_eog_ch = "HEOG",
    v_eog_ch = "VEOG",
    ecg_ch = "ECG",
    iclabel_threshold=0.7
    )

# Function to check if a set of three events is complete
def is_complete_set(probe_set):
    return all(event[2] in ms_dic 
               or event[2] in vol_dic 
               or event[2] in sleepi_dic 
               for event in probe_set)


# %% Simplified version

report_Event = mne.Report(title='Event')
report_AR = mne.Report(title='Auto Reject')
report_ERP = mne.Report(title='ERP')
report_ICA = mne.Report(title='ICA')

for i, file_path in enumerate(files) :
    #### [1] Import Data and Minimally Process it
    sub_id = f"{file_path.split('/sub_')[1][:6]}{file_path.split('SART')[1][:3]}"
    
    if "HS_007" in sub_id :
        continue
    
    print(f"...Processing {sub_id}, file {i+1} / {len(files)}...")
    
    subtype = sub_id[:2]
    
    this_trialepochs_savename = os.path.join(
        path_preproc, "epochs_task", f"{sub_id}_epo.fif"
        )
    # this_probes_savename = os.path.join(
    #     path_preproc, "epochs_probes", f"{sub_id}_epo.fif"
    #     )
    
    if os.path.exists(this_trialepochs_savename) :
    # if (os.path.exists(this_trialepochs_savename) and 
    #     os.path.exists(this_probes_savename)):
        print(f"...{sub_id}, file {i+1} / {len(files)} Already processed, skipping...")
        continue
    
    raw = cfg.load_and_preprocess_data(file_path)
    sf = raw.info['sfreq']
    
    """
    
    HERE ADD PART TO INTERPOLATE CHANNEL OR IN FUNCTION LOAD AND PREPROCESS
    
    """
    
    #### [2] Handle Events
    events, event_id = cfg.handle_events(raw, merge_dict=merge_dict)
    stim_events = mne.pick_events(events, include=[101,100])
    
    #### [3] Trial epochs based on events
    epochs = mne.Epochs(
        raw, 
        stim_events, 
        event_id = [101, 100], 
        tmin = -.2,
        tmax = 1.2,
        baseline = None,
        preload = True,
        flat = flat_criteria,
        event_repeated = 'merge'
        )
    
    #### [4] ICA on trial epochs
    ica = cfg.automatic_ica(
        eeg_data = epochs, 
        sub_id = sub_id, 
        output_dir = os.path.join(path_preproc, "ica_files"), 
        n_components = ica_dic['n_components'], 
        l_freq = ica_dic['l_freq'], 
        v_eog_ch='VEOG', 
        h_eog_ch='HEOG', 
        ecg_ch='ECG',
        icalabel=False
        )
    
    report_ICA.add_ica(
        ica=ica,
        title=f"All Sources ICA: {sub_id}",
        inst=epochs,
        n_jobs=n_jobs  
    )
    report_ICA.save(overwrite = True, open_browser=False)
    
    epochs_ica = ica.apply(epochs.copy())
    epochs_ica.drop_bad(reject = dict(eeg=threshold))
    
    #### [5] AutoReject on trial epochs
    ar_epochs = epochs_ica.copy().filter(1, None, n_jobs = n_jobs)
    ar = AutoReject(
        ar_dic["n_interpolates"], 
        ar_dic["consensus_percs"],
        ar_dic["picks"],
        thresh_method=ar_dic["thresh_method"], 
        random_state=ar_dic["random_state"]
        )
    ar.fit(ar_epochs[:round(len(epochs)*.2)])
    epochs_ar_ica, reject_log = ar.transform(epochs_ica, return_log=True)
    epochs_ar_ica.apply_baseline((-.2, 0))
    epochs_ar_ica.save(this_trialepochs_savename, overwrite = True)
    
    fig = reject_log.plot(orientation = 'horizontal', show=False)
    report_AR.add_figure(
        fig=fig,
        title=f"Reject log: {sub_id}",
        section=sub_id,
        caption="The reject log returned by autoreject",
        image_format="PNG",
        )
    
    #### [6] ERP Trial and Contrast
    epochs.apply_baseline((-.2, 0))
    evoked_go = epochs[str(go_id)].average()
    evoked_nogo = epochs[str(nogo_id)].average()
    fig = mne.viz.plot_compare_evokeds(
        [evoked_go, evoked_nogo],
        picks='Pz', 
        show_sensors='upper right',
        title='Averaged ERP at Pz',
        show = False
        )
    report_ERP.add_figure(
        fig,
        f"ERP NoGo vs Go of {sub_id} at Pz",
        image_format='png',
        tags=('P300', 'Pz'),
        section=sub_id,
        replace=False,
        )
    # fig = mne.viz.plot_compare_evokeds(
    #     [evoked_go, evoked_nogo],
    #     combine='gfp', 
    #     show_sensors='upper right',
    #     title='Averaged ERP across channels',
    #     show = False
    #     )
    # report_ERP.add_figure(
    #     fig,
    #     f"ERP NoGo vs Go of {sub_id} GFP",
    #     image_format='png',
    #     tags=('P300', 'GFP'),
    #     section=sub_id,
    #     replace=False,
    #     )
    epochs_ica.apply_baseline((-.2, 0))
    evoked_go = epochs_ica[str(go_id)].average()
    evoked_nogo = epochs_ica[str(nogo_id)].average()
    fig = mne.viz.plot_compare_evokeds(
        [evoked_go, evoked_nogo],
        picks='Pz', 
        show_sensors='upper right',
        title='Averaged ERP at Pz',
        show = False
        )
    report_ERP.add_figure(
        fig,
        f"ERP NoGo vs Go of {sub_id} at Pz after ICA",
        image_format='png',
        tags=('P300', 'Pz', 'ICA'),
        section=sub_id,
        replace=False,
        )
    # fig = mne.viz.plot_compare_evokeds(
    #     [evoked_go, evoked_nogo],
    #     combine='gfp', 
    #     show_sensors='upper right',
    #     title='Averaged ERP across channels',
    #     show = False
    #     )
    # report_ERP.add_figure(
    #     fig,
    #     f"ERP NoGo vs Go of {sub_id} GFP after ICA",
    #     image_format='png',
    #     tags=('P300', 'GFP'),
    #     section=sub_id,
    #     replace=False,
    #     )
    
    evoked_go = epochs_ar_ica[str(go_id)].average()
    evoked_nogo = epochs_ar_ica[str(nogo_id)].average()
    fig = mne.viz.plot_compare_evokeds(
        [evoked_go, evoked_nogo],
        picks='Pz', 
        show_sensors='upper right',
        title='Averaged ERP at Pz',
        show = False
        )
    report_ERP.add_figure(
        fig,
        f"ERP NoGo vs Go of {sub_id} at Pz after AutoReject and ICA",
        image_format='png',
        tags=('P300', 'Pz', 'ICA', 'AutoReject'),
        section=sub_id,
        replace=False,
        )
    # fig = mne.viz.plot_compare_evokeds(
    #     [evoked_go, evoked_nogo],
    #     combine='gfp', 
    #     show_sensors='upper right',
    #     title='Averaged ERP across channels',
    #     show = False
    #     )
    # report_ERP.add_figure(
    #     fig,
    #     f"ERP NoGo vs Go of {sub_id} GFP after AutoReject and ICA",
    #     image_format='png',
    #     tags=('P300', 'GFP', 'AutoReject', 'ICA'),
    #     section=sub_id,
    #     replace=False,
    #     )
    del evoked_go, evoked_nogo, epochs_ica, epochs
    
    conditions=[str(go_id),str(nogo_id)];
    evoked_clean_perCond = {c:epochs_ar_ica[c].average() for c in conditions}
    savename = f"ERP_{sub_id}_ave.fif"
    mne.write_evokeds(
        os.path.join(path_data,"intermediary",savename), 
        list(evoked_clean_perCond.values()), overwrite=True
        )
    del evoked_clean_perCond

    
report_Event.save(
    os.path.join(path_data,"reports","Events_3.html"), 
    overwrite=True, 
    open_browser=False
    )
report_AR.save(
    os.path.join(path_data,"reports","AutoRej_3.html"), 
    overwrite=True, 
    open_browser=False
    )
report_ERP.save(
    os.path.join(path_data,"reports","ERP_3.html"), 
    overwrite=True, 
    open_browser=False
    )
report_ICA.save(
    os.path.join(path_data,"reports","ICA_3.html"), 
    overwrite=True,
    open_browser=False
    )
    
# %% GET ERPs across subjects
evokeds_files = glob(os.path.join(path_data,"intermediary" ,'ERP_*.fif'))
evokeds = {} #create an empty dict
conditions = ['101','100']
# #convert list of evoked in a dict (w/ diff conditions if needed)
for idx, c in enumerate(conditions):
    evokeds[c] = [mne.read_evokeds(d)[idx] for d in evokeds_files]

evokeds # We can see that he matched the conditions by treating each as if it was 2 objcts as before 


# "Plot averaged ERP on all subj"
ERP_mean = mne.viz.plot_compare_evokeds(evokeds,
                              picks='Pz', show_sensors='upper right',
                              title='Averaged ERP all subjects',
                            )
plt.show()


#gfp: "Plot averaged ERP on all subj"
ERP_gfp = mne.viz.plot_compare_evokeds(evokeds,
                              combine='gfp', show_sensors='upper right',
                              title='Averaged ERP all subjects',
                            )
plt.show()


# evokeds_files = glob.glob(path_data+"intermediary/" + '/cont_ce_*.fif')
# evokeds_diff = {} #create an empty dict
# # #convert list of evoked in a dict (w/ diff conditions if needed)
# for idx, d in enumerate(evokeds_files):
#     evokeds_diff[idx] = mne.read_evokeds(d)[0]
    
# ERP_mean = mne.viz.plot_evoked(evokeds_diff,
#                               picks='PO8',
#                               title='Averaged difference wave all subjects',
#                             )
# plt.show()