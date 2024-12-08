# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from mne.utils import logger

from . montages import get_ch_names

_roi_names = ['Fz', 'Cz', 'Pz', 'p3a', 'p3b', 'mmn', 'cnv', 'scalp', 'nonscalp']


_rois_map = {}


def define_rois(montage, rois_map):
    if not all(x in rois_map for x in _roi_names):
        raise ValueError('All ROIS are not defined for {}'.format(montage))
    if montage in _rois_map:
        logger.warning('Warning: ROIs already defined for {}'.format(montage))
    _rois_map[montage] = rois_map


def get_roi(config, roi_name):
    out = None
    if config in _rois_map:
        this_rois = _rois_map[config]
        if roi_name not in this_rois:
            raise ValueError(
                'ROI {} not defined for montage {}'.format(roi_name, config))
        out = this_rois[roi_name]
    else:
        raise ValueError(
            'ROIs not defined for montage {}'.format(config))

    return out


def get_roi_ch_names(config, roi_name):
    orig_names = get_ch_names(config)
    roi_idx = get_roi(config, roi_name)
    ch_names = []
    if roi_idx is not None:
        ch_names = [orig_names[i] for i in roi_idx]
    return ch_names
