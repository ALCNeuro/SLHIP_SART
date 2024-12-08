# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import os.path as op

import numpy as np

from .montages import (define_equipment, get_montage, define_neighbor,
                       get_ch_names, define_map_montage)

from .rois import define_rois

_ant_montages = {
    '32': 'standard_1020',
    '63': 'standard_1020',
    '124': 'standard_1020'
}

_ant_ch_names = {
    '32': [
        'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
        'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
        'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2',
        'Oz'],
    '63': [
        'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
        'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
        'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2',
        'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3',
        'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2',
        'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7',
        'PO8', 'Oz'],
    '124': [
        'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5',
        'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3',
        'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3',
        'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CP2',
        'CP4', 'CP6', 'TP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
        'P6', 'P8', 'P10', 'PO9', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'PO10',
        'O1', 'O2', 'I1', 'Iz', 'I2', 'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h',
        'FFT7h', 'FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h', 'FFT8h',
        'FTT9h', 'FTT7h', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h',
        'FTT8h', 'FTT10h', 'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h',
        'CCP4h', 'CCP6h', 'TTP8h', 'TPP9h', 'TPP7h', 'CPP5h', 'CPP3h', 'CPP1h',
        'CPP2h', 'CPP4h', 'CPP6h', 'TPP8h', 'TPP10h', 'PPO9h', 'PPO5h',
        'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h',
        'AFF1', 'AFF2', 'PPO1', 'PPO2', 'M1', 'M2']
    
}


_ant_32_rois = {
    'p3a': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21]),
    'p3b': np.array([9, 10, 14, 15, 16, 20, 21, 24, 25, 26, 28]),
    'mmn': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21]),
    'cnv': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21]),
    'Fz': np.array([4, 5, 6, 9, 10]),
    'Cz': np.array([9, 10, 14, 15, 16, 20, 21]),
    'Pz': np.array([20, 21, 24, 25, 26, 28]),
    'scalp': np.arange(32),
    'nonscalp': None
}


_ant_63_rois = {
    'p3a': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21, 37, 38, 41, 44, 45]),
    'p3b': np.array([9, 10, 14, 15, 16, 20, 21, 24, 25, 26, 28,
                     41, 44, 45, 50, 51, 54, 55]),  # surrounding Cz Cz+Pz
    'mmn': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21, 37, 38, 41, 44, 45]),
    'cnv': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21, 37, 38, 41, 44, 45]),
    'Fz': np.array([4, 5, 6, 9, 10, 37, 38, 41]),
    'Cz': np.array([9, 10, 14, 15, 16, 20, 21, 41, 44, 45]),
    'Pz': np.array([20, 21, 24, 25, 26, 28, 50, 51, 54, 55]),
    'scalp': np.concatenate(
        [np.arange(0, 12), np.arange(13, 18), np.arange(19, 31),
         np.arange(32, 64)]),
    'nonscalp': np.array([12, 18, 31])
}

_ant_124_rois = {
    'p3a': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21, 37, 38, 41, 44, 45]),
    'p3b': np.array([9, 10, 14, 15, 16, 20, 21, 24, 25, 26, 28,
                     41, 44, 45, 50, 51, 54, 55]),  # surrounding Cz Cz+Pz
    'mmn': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21, 37, 38, 41, 44, 45]),
    'cnv': np.array([4, 5, 6, 9, 10, 14, 15, 16, 20, 21, 37, 38, 41, 44, 45]),
    'Fz': np.array([4, 5, 6, 9, 10, 37, 38, 41]),
    'Cz': np.array([9, 10, 14, 15, 16, 20, 21, 41, 44, 45]),
    'Pz': np.array([20, 21, 24, 25, 26, 28, 50, 51, 54, 55]),
    'scalp': np.concatenate(
        [np.arange(0, 12), np.arange(13, 18), np.arange(19, 124)]),
    'nonscalp': np.array([12, 18])
}


def register():
    define_equipment('ant', _ant_montages, _ant_ch_names)
    define_rois('ant/32', _ant_32_rois)
    define_rois('ant/63', _ant_63_rois)
    define_rois('ant/124', _ant_124_rois)
    c_path = op.abspath(op.dirname(op.realpath(__file__)))
    fname = op.join(c_path, 'data', 'ant_124_neighbours.mat')
    define_neighbor('ant/124', fname)
