# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np
import os.path as op

from .montages import define_equipment, define_map_montage, define_neighbor
from .rois import define_rois


_eximia_montages = {
    '60': 'standard_1020'
}

_eximia_60_rois = {
    'p3a': np.array([8, 9, 10, 17, 18, 19, 27, 28, 29]),
    'p3b': np.array([27, 28, 29, 37, 38, 39, 47, 48, 49]),
    'mmn': np.array([8, 9, 10, 17, 18, 19, 27, 28, 29]),
    'cnv': np.array([8, 9, 10, 17, 18, 19, 27, 28, 29]),
    'Fz': np.array([8, 9, 10]),
    'Cz': np.array([27, 28, 29]),
    'Pz': np.array([47, 48, 49]),
    'scalp': np.arange(60),
    'nonscalp': None
}

_eximia_ch_names = {
    '60': [
        'Fp1', 'Fpz', 'Fp2', 'AF1', 'AFz', 'AF2', 'F7', 'F5', 'F1', 'Fz', 'F2',
        'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        'TP8', 'TP10', 'P9', 'P7', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P8', 'P10',
        'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2', 'Iz']
}


# AF1 becomes AF3, and AF2 becomes AF4 and CZ = E81
# (they are too close in the EGI)
_eximia60_egi256_names = [
    'E37', 'E26', 'E18', 'E34', 'E20', 'E12', 'E47', 'E48', 'E29', 'E21', 'E5',
    'E222', 'E2', 'E67', 'E62', 'E49', 'E42', 'E24', 'E15', 'E207', 'E206',
    'E213', 'E211', 'E219', 'E69', 'E64', 'E59', 'E44', 'E9', 'E185', 'E183',
    'E194', 'E202', 'E94', 'E84', 'E76', 'E66', 'E79', 'E81', 'E143', 'E164',
    'E172', 'E179', 'E190', 'E106', 'E96', 'E87', 'E88', 'E101', 'E142',
    'E153', 'E170', 'E169', 'E109', 'E119', 'E140', 'E116', 'E126', 'E150',
    'E137']

_egi256_eximia60_map = {
    k: v for k, v in zip(_eximia60_egi256_names, _eximia_ch_names['60']) if
    k is not None}


def register():
    define_equipment('eximia', _eximia_montages, _eximia_ch_names)
    define_rois('eximia/60', _eximia_60_rois)
    define_map_montage('egi/256', 'eximia/60', _egi256_eximia60_map)
    c_path = op.abspath(op.dirname(op.realpath(__file__)))
    fname = op.join(c_path, 'data', 'eximia_60_neighbours.mat')
    define_neighbor('eximia/60', fname)
