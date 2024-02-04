#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:50:50 2024

@author: arthurlecoz

00_inspect_eeg.py

Need to create a file in your computer/cold storage with the following architecture :
    rootpath = "/path/to/root/"

then in the rootpath you'll have 5 files :
    - Raw           -> raw eeg files
    - Preproc       -> preprocessed eeg files
    - Dynamo        -> Grip sensor files
    - Audio         -> Free report files
    - Figs          -> Figures
    
"""
# %% paths and variables

import numpy as np
import mne
from glob import glob
import os

if "arthur" in os.getcwd() :
    root_path = '/Volumes/DDE_ALC/PhD/SLHIP'

raw_path = os.path.join(root_path, '00_Raw')
prepoc_path = os.path.join(root_path, '02_Preproc')
# figs_path = os.path.join(root_path, "Figs")

visualize = 0

# %% script

raw = mne.io.read_raw_brainvision(
    glob(os.path.join(raw_path, "*.vhdr"))[0],
    preload = True
    )
print(raw.info)

if visualize :
    raw.plot(duration = 30)

# raw.filter(0.1, 40)

if visualize :
    raw.plot(duration = 30)

events, event_id = mne.events_from_annotations(raw)

"""
Triggers from Matlab :
    trig_start          = 1; 
    trig_end            = 11; 
    trig_startBlock     = 2; 
    trig_endBlock       = 22; 
    trig_startTrial     = 64; 
    trig_startQuestion  = 128; 
    trig_probestart     = 3; 
    trig_probeend       = 33; 
    trig_response       = 5;
"""

