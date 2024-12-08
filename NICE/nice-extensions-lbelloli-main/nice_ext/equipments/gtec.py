# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np

from .montages import define_equipment
from .rois import define_rois


_gtec_montages = {
    '12': 'standard_1020'
}

_gtec_12_rois = {
    'p3a': np.array([0, 1, 8, 9]),
    'p3b': np.array([1, 2, 3, 4, 8, 9, 10, 11]),
    'mmn': np.array([0, 1, 8, 9]),
    'cnv': np.array([0, 1, 8, 9]),
    'Fz': np.array([0]),
    'Cz': np.array([1, 8, 9]),
    'Pz': np.array([2, 3, 4, 10, 11]),
    'scalp': np.arange(12),
    'nonscalp': None
}

_gtec_ch_names = {
    '12': [
        'Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8', 'FC3',
        'FC4', 'CP3', 'CP4']
}


def register():
    define_equipment('gtec', _gtec_montages, _gtec_ch_names)
    define_rois('gtec/12', _gtec_12_rois)
