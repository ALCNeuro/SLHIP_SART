# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np
import scipy

import re
from collections import namedtuple

import mne

# from .stats import linear_regression
from .equipments import get_roi_ch_names


def _evoked_apply_method(epochs, method):
    data = method(epochs, axis=0)
    out = mne.evoked.EvokedArray(data, epochs.info, epochs.tmin)
    return out


def get_evoked(epochs, condition, method=np.mean, roi_name=None):
    epochs = epochs[condition]

    if roi_name is not None:
        roi_channels = get_roi_ch_names(
            config=epochs.info['description'], roi_name=roi_name)
        epochs.pick_channels(roi_channels)
        this_data = epochs.get_data().mean(1, keepdims=True)
        to_drop = epochs.info['ch_names'][1:]
        to_rename = epochs.info['ch_names'][0]
        epochs.drop_channels(to_drop)
        epochs._data = this_data
        epochs.rename_channels({to_rename: 'ROI-MEAN'})

    evoked = _evoked_apply_method(epochs[condition], method)
    evoked_stderr = _evoked_apply_method(epochs[condition], scipy.stats.sem)

    return evoked, evoked_stderr


def get_contrast(epochs, conditions, method=np.mean, roi_name=None,
                 roi_channels=None, paired=False):
    """Aux function"""

    if len(conditions) != 2:
        raise ValueError('I need 2 conditions to make a contrast.')

    all_conditions = (
        sum(conditions, []) if isinstance(conditions[0], list)
        else conditions)
    epochs = epochs[all_conditions]

    if roi_name is not None:
        if roi_channels is None:
            roi_channels = get_roi_ch_names(
                config=epochs.info['description'], roi_name=roi_name)
        epochs.pick_channels(roi_channels)
        this_data = epochs.get_data().mean(1, keepdims=True)
        to_drop = epochs.info['ch_names'][1:]
        to_rename = epochs.info['ch_names'][0]
        epochs.drop_channels(to_drop)
        epochs._data = this_data
        epochs.rename_channels({to_rename: 'ROI-MEAN'})

    evokeds = [_evoked_apply_method(epochs[c], method) for c in
               conditions]
    evoked_out = mne.combine_evoked(evokeds, [1., -1.])

    evokeds_stderr = [_evoked_apply_method(epochs[c], scipy.stats.sem)
                      for c in conditions]

    stats = namedtuple('stats', 'p_val mlog10_p_val')
    epochs_a = epochs[conditions[0]]
    epochs_b = epochs[conditions[1]]
    if paired is False:
        t_val, p_val = scipy.stats.ttest_ind(
            epochs_a.get_data(), epochs_b.get_data(), axis=0,
            equal_var=False)
    else:
        # evoked_data = epochs_a.get_data() - epochs_b.get_data()
        t_val, p_val = scipy.stats.ttest_rel(
            epochs_a.get_data(), epochs_b.get_data(), axis=0)
    stats.p_val = mne.evoked.EvokedArray(p_val, epochs.info, epochs.tmin)
    stats.mlog10_p_val = mne.evoked.EvokedArray(
        -np.log10(p_val), epochs.info, epochs.tmin)
    return evoked_out, evokeds, evokeds_stderr, stats


def get_contrast_1samp(epochs, conditions, method=np.mean, roi_name=None):
    """Aux function"""
    if conditions == 'all':
        conditions = [list(epochs.event_id.keys())]
    all_conditions = (
        sum(conditions, []) if isinstance(conditions[0], list)
        else conditions)
    epochs = epochs[all_conditions]

    if roi_name is not None:
        roi_channels = get_roi_ch_names(
            config=epochs.info['description'], roi_name=roi_name)
        epochs.pick_channels(roi_channels)
        this_data = epochs.get_data().mean(1, keepdims=True)
        to_drop = epochs.info['ch_names'][1:]
        to_rename = epochs.info['ch_names'][0]
        epochs.drop_channels(to_drop)
        epochs._data = this_data
        epochs.rename_channels({to_rename: 'ROI-MEAN'})
    evokeds = [_evoked_apply_method(epochs[c], method) for c in
               conditions]
    evokeds_stderr = [_evoked_apply_method(epochs[c], scipy.stats.sem)
                      for c in conditions]

    # TODO: What's the 1-samp stuff? Bertrand

    return evokeds, evokeds_stderr


def fname_regexp_event(fname, regex_map, event_id):
    found = 0
    match = 0
    for reg, evt in regex_map.items():
        if re.match(reg, fname) is not None:
            found = found + 1
            match = evt
    if found == 0:
        raise ValueError('No regexp match for {}'.format(fname))
    elif found > 1:
        raise ValueError('More than one match for {}'.format(fname))

    return event_id[match]
