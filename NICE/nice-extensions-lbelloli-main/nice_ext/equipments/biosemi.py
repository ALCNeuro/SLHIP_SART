# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np

from .montages import define_equipment
from .rois import define_rois

_biosemi_montages = {
    '128': 'biosemi128'
}


_biosemi128_rois = {
    'p3a': np.array([0, 1, 32, 64, 75, 83, 84, 85, 88, 96, 110]),
    'p3b': np.array([0, 1, 3, 4, 18, 19, 31, 32, 64, 96, 110]),
    'mmn': np.array([0, 1, 32, 64, 75, 83, 84, 85, 88, 96, 110]),
    'cnv': np.array([0, 1, 32, 64, 75, 83, 84, 85, 88, 96, 110]),
    'Fz': np.array([75, 83, 84, 85, 88]),
    'Cz': np.array([0, 1, 32, 64, 96, 110]),
    'Pz': np.array([3, 4, 18, 19, 31]),
    'scalp': np.arange(128),
    'nonscalp': None
}

_biosemi_ch_names = {
    '128':
        ['A{}'.format(x) for x in range(1, 33)] +
        ['B{}'.format(x) for x in range(1, 33)] +
        ['C{}'.format(x) for x in range(1, 33)] +
        ['D{}'.format(x) for x in range(1, 33)]
}


def register():
    define_equipment('biosemi', _biosemi_montages, _biosemi_ch_names)
    define_rois('biosemi/128', _biosemi128_rois)
