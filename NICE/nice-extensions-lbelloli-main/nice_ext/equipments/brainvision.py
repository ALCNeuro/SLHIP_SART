# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np

import os
import os.path as op

from .montages import define_equipment
from .rois import define_rois


_bv_montages = {
    '11monkey': 'easycap-M1',
    '32': 'easycap-M1',
    '128': op.join(op.abspath(os.path.dirname(os.path.realpath(__file__))),
                   'data/128_MNIregistered_ActiCap.sfp')
}

_bv_32_rois = {
    'p3a': np.array([3, 4, 7, 8, 12, 17, 18]),
    'p3b': np.array([7, 8, 12, 17, 18, 22, 23, 24]),
    'mmn': np.array([3, 4, 7, 8, 12, 17, 18]),
    'cnv': np.array([3, 4, 7, 8, 12, 17, 18]),
    'Fz': np.array([3, 4, 7, 8]),
    'Cz': np.array([7, 8, 12, 17, 18]),
    'Pz': np.array([17, 18, 22, 23, 24]),
    'scalp': np.arange(31),
    'nonscalp': None
}

_bv_128_rois = {
    'p3a': np.arange(128),
    'p3b': np.arange(128),
    'mmn': np.arange(128),
    'cnv': np.arange(128),
    'Fz': np.arange(128),
    'Cz': np.arange(128),
    'Pz': np.arange(128),
    'scalp': np.arange(128),
    'nonscalp': None
}

_bv_11monkey_rois = {
    'p3a': np.arange(11),
    'p3b': np.arange(11),
    'mmn': np.arange(11),
    'cnv': np.arange(11),
    'Fz': np.arange(11),
    'Cz': np.arange(11),
    'Pz': np.arange(11),
    'scalp': np.arange(11),
    'nonscalp': None
}


_bv_ch_names = {
    '11monkey': ['Fp1', 'Fp2', 'F3', 'C4', 'T3', 'T4', 'P3', 'P4', 'Pz', 'Oz',
                 'O2'],
    '32': [
        'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
        'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7',
        'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'],
    '128': [
        'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
        'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
        'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2',
        'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3',
        'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz',
        'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4',
        'FT8', 'F6', 'F2', 'AF4', 'AF8', 'F9', 'AFF1h', 'FFC1h', 'FFC5h',
        'FTT7h', 'FCC3h', 'CCP1h', 'CCP5h', 'TPP7h', 'P9', 'PPO9h', 'PO9',
        'I1', 'OI1h', 'PPO1h', 'CPP3h', 'CPP4h', 'PPO2h', 'OI2h', 'I2',
        'PO10', 'PPO10h', 'P10', 'TPP8h', 'CCP6h', 'CCP2h', 'FCC4h',
        'FTT8h', 'FFC6h', 'FFC2h', 'AFF2h', 'F10', 'AFp1', 'AFF5h',
        'FFT9h', 'FFT7h', 'FFC3h', 'FCC1h', 'FCC5h', 'FTT9h', 'TTP7h',
        'CCP3h', 'CPP1h', 'CPP5h', 'TPP9h', 'PPO5h', 'POO1', 'POO9h',
        'POO10h', 'POO2', 'PPO6h', 'TPP10h', 'CPP6h', 'CPP2h', 'CCP4h',
        'TTP8h', 'FTT10h', 'FCC6h', 'FCC2h', 'FFC4h', 'FFT8h', 'FFT10h',
        'AFF6h', 'AFp2']
}


def register():
    define_equipment('bv', _bv_montages, _bv_ch_names)
    define_rois('bv/32', _bv_32_rois)
    define_rois('bv/128', _bv_128_rois)
    define_rois('bv/11monkey', _bv_11monkey_rois)
