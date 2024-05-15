# -*- coding: utf-8 -*-
"""
author = Arthur LC

01_01_Preprocessing_trials.py


"""
# %% IMPORT MODULES
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from glob import glob
from autoreject import AutoReject
from mne_icalabel import label_components
import SLHIP_config_ALC as cfg 

#from copy import deepcopy

# %% Paths & Variables
# Paths
if 'arthur' in os.getcwd():
    path_data='/Volumes/DDE_ALC/PhD/SLHIP/00_Raw'
else:
    path_data='your_path'

if os.path.exists(os.path.join(path_data, "reports"))==False:
    os.makedirs(os.path.join(path_data, "reports"))
if os.path.exists(os.path.join(path_data, "intermediary"))==False:
    os.makedirs(os.path.join(path_data,"intermediary"))
    
files = glob(os.path.join(path_data, 'experiment', '**' , '*SART*.vhdr'))

# %% Simplified version

# Load raw data
file_path = files[0]
sub_id = file_path.split('/sub_')[1][:6]

raw = cfg.load_and_preprocess_data(file_path)

# Handle events
go_id=101
nogo_id=100
conditions=[str(go_id),str(nogo_id)]
merge_dict = {
    101 : [65,66,68,69,70,71,72,72],
    100 : [67]
    }
events, event_id = cfg.handle_events(raw, merge_dict=merge_dict)
stim_events = mne.pick_events(events, include=[101,100])

# Create epochs
flat_criteria = dict(eeg=1e-6)
threshold = 150
epochs = mne.Epochs(
    raw, 
    stim_events, 
    event_id = [101, 100], 
    tmin = -.2,
    tmax = 1.2,
    baseline = None,
    preload = True,
    flat = flat_criteria,
    reject=dict(eeg=threshold),
    event_repeated = 'merge'
    )

evoked_go  = epochs[str(go_id)].average()
evoked_nogo = epochs[str(nogo_id)].average()

# Ica & ica label
ica, epochs_clean = cfg.ICA_auto(
    epochs, 
    sub_id,
    os.path.join(path_data, "reports"),
    n_components=15, 
    l_freq=1.0, 
    method='infomax', 
    iclabel_threshold=0
    )


# %% LOAD, FILTER, CLEAN

report_Event = mne.Report(title='AutEvent')
report_AR = mne.Report(title='Auto Reject')
report_ERP = mne.Report(title='ERP')
report_ICA = mne.Report(title='ICA')

for file in files:

    # [1] LOAD RAW DATA
    file_name=file.split(os.sep)[-1]
    sub_ID=file_name.split('_')[5]
    sess_ID=file_name.split('_')[7][0:2]
    group_ID=file_name.split('_')[4]
    name_ID=group_ID + '_' + sub_ID+ "_" + sess_ID
    report_prefix=path_data+"reports"+os.sep + group_ID + '_' + sub_ID+ "_" + sess_ID + '_'
    raw = mne.io.read_raw_brainvision(file, preload=True)
    raw_eeg = raw.copy().drop_channels(['VEOG','HEOG','ECG'])
    
    
    print(raw_eeg)
    #HCGSN256_montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
    #raw.set_montage(HCGSN256_montage)

    # [2] MONTAGE
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_eeg.set_montage(montage, on_missing='ignore')

    
    # [3] REREFERNCING AND FILTERING
    raw_eeg.resample(250)
    sfreq = raw_eeg.info["sfreq"]
    raw_eeg.set_eeg_reference("average")
    raw_eeg.filter(0.1, 100, fir_design='firwin')
    raw_eeg.notch_filter(50,
                    filter_length='auto', phase='zero')
    report = mne.Report(title=name_ID)
    report.add_raw(raw=raw_eeg, title='Filtered Cont from"raw_eeg"', psd=False)  # omit PSD plot
    
    # To display EEG
    # raw_eeg.plot(n_channels=50, butterfly=True, group_by='position')

    # [4] EVENTS
    # stim, block, probes
    # trig_start          = 1; %S
    # trig_end            = 11; %E
    # trig_startBlock     = 2; %B
    # trig_endBlock       = 22; %K
    # trig_startTrial     = 64; %T
    # trig_startQuestion  = 128; %Q
    # trig_probestart     = 3; %P
    # trig_probeend       = 33; %C
    # trig_response       = 5; %C
    (events, event_dict) = mne.events_from_annotations(raw_eeg)
    
    go_trials=[65,66,68,69,70,71,72,72];
    nogo_trials=[67];
    events=mne.merge_events(events, list(go_trials), 101, replace_events=True)
    events=mne.merge_events(events, list(nogo_trials), 100, replace_events=True)
    count_events=mne.count_events(events)
    stim_events = mne.pick_events(events, include=[101,100])

    report.add_events(events=stim_events, title='Events from "stim_events"', sfreq=sfreq)

    report_Event.add_events(events=stim_events, title='Events: '+name_ID, sfreq=sfreq)

    # [5] EPOCHS
    go_id=101;
    nogo_id=100;
    threshold = 150
    epochs = mne.Epochs(raw_eeg, events, event_id=[go_id,nogo_id],
                        tmin=-0.2, tmax=1.2, preload=True, event_repeated='drop',reject=dict(eeg=threshold))
    report.add_epochs(epochs=epochs, title='Epochs from "epochs"')
    savename = "e_" + name_ID + ".fif"
    epochs.save(os.path.join(path_data,"intermediary",savename), overwrite=True)
    
    # [6] AUTOREJECT
    # n_interpolates = np.array([1, 4, 32])
    # consensus_percs = np.linspace(0, 1.0, 11)
    # ar = AutoReject(n_interpolates, consensus_percs,
    #                 thresh_method='random_search', random_state=42)
    # ar.fit(epochs)
    # epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    
    epochs_clean = epochs
    epochs_clean.set_eeg_reference("average")

    savename = "ce_" + name_ID + ".fif"
    epochs_clean.save(os.path.join(path_data,"intermediary",savename), overwrite=True)
    
    #fig = reject_log.plot(orientation = 'horizontal', show=False)

    # report.add_figure(
    #     fig=fig,
    #     title="Reject log",
    #     caption="The rejct log returned by autoreject",
    #     image_format="PNG",
    # )
    # report_AR.add_figure(
    #     fig=fig,
    #     title="Reject log: "+name_ID,
    #     caption="The rejct log returned by autoreject",
    #     image_format="PNG",
    # )
    
    # [8] ICA
    ica_epochs = mne.Epochs(raw_eeg.copy().filter(l_freq=1.0, h_freq=None), events, event_id=[go_id,nogo_id],
                        tmin=-0.2, tmax=1.2, reject=dict(eeg=threshold), event_repeated='drop',preload=True,baseline=None)
    ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=97,method='infomax', fit_params=dict(extended=True))
    # ar_ica = AutoReject(n_interpolates, consensus_percs,
    #                 thresh_method='random_search', random_state=42)
    # ar_ica.fit(ica_epochs)
    # ica_epochs_clean, reject_log = ar_ica.transform(ica_epochs, return_log=True)
    ica_epochs_clean = ica_epochs
    ica_epochs_clean.set_eeg_reference("average")
    ica.fit(ica_epochs_clean)
    savename = "ica_ce_" + name_ID + ".fif"
    ica.save(os.path.join(path_data,"intermediary",savename), overwrite=True)
    ica_fig=ica.plot_sources(ica_epochs_clean, show_scrollbars=True, show=False)
    
    # ICA rejection
    ica_classification=label_components(ica_epochs_clean, ica, method='iclabel')
    ica_labels=pd.DataFrame(ica_classification)
    ica_labels.to_csv(report_prefix+'ICAlabels.csv')
    labels = ica_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    caption_str=''
    for idx, label in enumerate(labels):
        caption_str=caption_str+'ICA'+str(idx)+': '+label+'; '
        
    print(f"Excluding these ICA components: {exclude_idx}")
    epochs_clean_ica=ica.apply(epochs_clean, exclude=exclude_idx)
    epochs_clean_ica.set_eeg_reference("average")

    report.add_ica(
        ica=ica,
        title='ICA cleaning',
        inst=epochs_clean_ica,
        n_jobs=2  # could be increased!
    )
    report.add_figure(
        ica_fig,
        title="ICA sources",
        caption=caption_str,
    )
    
    report_ICA.add_ica(
        ica=ica,
        title='ICA:' + name_ID,
        inst=epochs_clean_ica,
        n_jobs=2  # could be increased!
    )
    report_ICA.add_figure(
        ica_fig,
        title='ICA sources:' + name_ID,
        caption=caption_str,
    )

    # [7] ERP
    evoked_go  = epochs[str(go_id)].average()
    evoked_nogo = epochs[str(nogo_id)].average()
    
    evoked_clean_go  = epochs_clean[str(go_id)].average()
    evoked_clean_nogo = epochs_clean[str(nogo_id)].average()
    
    evoked_ica_clean_go  = epochs_clean_ica[str(go_id)].average()
    evoked_ica_clean_nogo = epochs_clean_ica[str(nogo_id)].average()
    
    conditions=[str(go_id),str(nogo_id)];
    evoked_clean_perCond = {c:epochs_clean_ica[c].average() for c in conditions}
    savename = "erp_ce_" + name_ID + ".fif"
    mne.write_evokeds(os.path.join(path_data,"intermediary",savename), 
                      list(evoked_clean_perCond.values()), overwrite=True
                     )
    report.add_evokeds(
        evokeds=[evoked_go,evoked_nogo,evoked_ica_clean_go,evoked_ica_clean_nogo],
        titles=["go", "nogo","ica+clean go", "ica+clean nogo"],  # Manually specify titles
        n_time_points=5,
        replace=True)

    
    # [9] CONTRAST
    picks = 'Pz'
    evokeds_ica_clean = dict(nogo=evoked_ica_clean_nogo, go=evoked_ica_clean_go)
    erp_ica_clean_fig=mne.viz.plot_compare_evokeds(evokeds_ica_clean, picks=picks, show=False)
    evokeds = dict(nogo=evoked_nogo, go=evoked_go)
    erp_fig=mne.viz.plot_compare_evokeds(evokeds, picks=picks, show=False)
    evokeds_clean = dict(nogo=evoked_clean_nogo, go=evoked_clean_go)
    erp_fig_clean=mne.viz.plot_compare_evokeds(evokeds_clean, picks=picks, show=False)

    report.add_figure(
         erp_fig,
         title="ERP contrast",
         caption="NoGo vs Go at Pz",
     )
    report.add_figure(
         erp_fig_clean,
         title="ERP contrast AR",
         caption="NoGo vs Go at Pz",
     )
    report.add_figure(
          erp_ica_clean_fig,
          title="ERP contrast AR+ica",
          caption="NoGo vs Go at Pz",
      )
 
    nogo_ica_clean_vis = mne.combine_evoked([evoked_ica_clean_nogo, evoked_ica_clean_go], weights=[1, -1])
    erp_ica_clean_but_fig=nogo_ica_clean_vis.plot_joint(show=False)
    nogo_clean_vis = mne.combine_evoked([evoked_clean_nogo, evoked_clean_go], weights=[1, -1])
    erp_clean_but_fig=nogo_clean_vis.plot_joint(show=False)
    nogo_vis = mne.combine_evoked([evoked_nogo, evoked_go], weights=[1, -1])
    erp_but_fig=nogo_vis.plot_joint(show=False)
    report.add_figure(
          erp_but_fig,
          title="ERP contrast (butterfly)",
          caption="NoGo vs Go across all Elec",
      )
    report.add_figure(
          erp_clean_but_fig,
          title="AR ERP contrast (butterfly)",
          caption="NoGo vs Go across all Elec",
      )
    report.add_figure(
          erp_ica_clean_but_fig,
          title="AR+ica ERP contrast (butterfly)",
          caption="NoGo vs Go across all Elec",
      )
    savename = "cont_ce_" + name_ID + ".fif"
    nogo_ica_clean_vis.save(os.path.join(path_data,"intermediary",savename), overwrite=True)
    

    report_ERP.add_figure(
          erp_but_fig,
          title="diff: "+name_ID,
          caption="NoGo vs Go across all Elec",
      )
    report_ERP.add_figure(
          erp_clean_but_fig,
          title="AR diff: "+name_ID,
          caption="NoGo vs Go across all Elec",
      )
    report_ERP.add_figure(
          erp_ica_clean_but_fig,
          title="AR+ICA diff: "+name_ID,
          caption="NoGo vs Go across all Elec",
      )
    
    report.save(report_prefix+"pipeline.html", overwrite=True, open_browser=False)
    
    report_Event.save(os.path.join(path_data,"reports","Events.html"), overwrite=True, open_browser=False)
    report_AR.save(os.path.join(path_data,"reports","AutoRej.html"), overwrite=True, open_browser=False)
    report_ERP.save(os.path.join(path_data,"reports","ERP.html"), overwrite=True, open_browser=False)
    report_ICA.save(os.path.join(path_data,"reports","ICA.html"), overwrite=True, open_browser=False)
    
    plt.close('all')
    
# %% GET ERPs across subjects
evokeds_files = glob.glob(os.path.join(path_data,"intermediary" ,'erp_ce_*.fif'))
evokeds = {} #create an empty dict
conditions = ['100','101']
# #convert list of evoked in a dict (w/ diff conditions if needed)
for idx, c in enumerate(conditions):
    evokeds[c] = [mne.read_evokeds(d)[idx] for d in 
    evokeds_files]

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
#                              picks='PO8',
#                              title='Averaged difference wave all subjects',
#                             )
# plt.show()