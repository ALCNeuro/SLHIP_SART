# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import os.path as op
import numpy as np

import mne

from .montages import (define_equipment, get_montage, define_neighbor,
                       get_ch_names, define_map_montage)
from .rois import define_rois

_egi_montages = {
    '16a': 'None',
    '32a': 'None',
    '64a': 'None',
    '64': 'GSN-HydroCel-64_1.0',
    '65': 'GSN-HydroCel-65_1.0',
    '128': 'GSN-HydroCel-128',
    '129': 'GSN-HydroCel-129',
    '256': 'GSN-HydroCel-256',
    '257': 'GSN-HydroCel-257'
}


_egi256_rois = {
    'p3a': np.array([6, 7, 9, 14, 15, 16, 22, 23, 45, 81, 132, 186]) - 1,
    'p3b': np.array([9, 45, 81, 100, 101, 110, 119, 128, 129, 132, 186]) - 1,
    'mmn': np.array([6, 7, 9, 14, 15, 16, 22, 23, 45, 81, 132, 186]) - 1,
    'cnv': np.array([6, 7, 14, 15, 16, 22, 23]) - 1,
    'Fz': np.array([6, 7, 14, 15, 16, 22, 23]) - 1,
    'Cz': np.array([9, 45, 81, 132, 186]) - 1,
    'Pz': np.array([100, 101, 110, 119, 128, 129]) - 1,
    'scalp': np.arange(224),
    'nonscalp': np.arange(224, 256),
    'Fp1': np.array([26, 27, 32, 33, 34, 37, 38]) - 1,
    'Fp2': np.array([11, 12, 18, 19, 20, 25, 26]) - 1,
    'F3': np.array([30, 36, 40, 41, 42, 49, 50]) - 1,
    'F4': np.array([205, 206, 213, 214, 215, 223, 224]) - 1,
    'C3': np.array([51, 52, 58, 59, 60, 65, 66]) - 1,
    'C4': np.array([155, 164, 182, 183, 184, 195, 196]) - 1,
    'P3': np.array([76, 77, 85, 86, 87, 97, 98]) - 1,
    'P4': np.array([152, 153, 161, 162, 163, 171, 172]) - 1,
    'T5': np.array([83, 84, 85, 94, 95, 96, 104, 105, 106]) - 1,
    'T6': np.array([169, 170, 171, 177, 178, 179, 189, 190, 191]) - 1,
    'Oz': np.array([125, 136, 137, 138, 148]) - 1,
}

_egi128_rois = {
    'p3a': np.array([4, 5, 11, 12, 16, 19, 7, 31, 55, 80, 106]) - 1,
    'p3b': np.array([7, 31, 55, 80, 106, 61, 62, 72, 78]) - 1,
    'mmn': np.array([4, 5, 11, 12, 16, 19, 7, 31, 55, 80, 106]) - 1,
    'cnv': np.array([4, 5, 11, 12, 16, 19]) - 1,
    'Fz': np.array([4, 5, 11, 12, 16, 19]) - 1,
    'Cz': np.array([7, 31, 55, 80, 106]) - 1,
    'Pz': np.array([61, 62, 72, 78]) - 1,
    'scalp': np.arange(124),
    'nonscalp': np.arange(124, 127)
}

_egi64_rois = {
    'p3a': np.array([3, 6, 8, 9, 4, 7, 16, 21, 41, 51, 54]) - 1,
    'p3b': np.array([4, 7, 16, 21, 41, 51, 54, 34, 33, 36, 38]) - 1,
    'mmn': np.array([3, 6, 8, 9, 4, 7, 16, 21, 41, 51, 54]) - 1,
    'cnv': np.array([3, 6, 8, 9]) - 1,
    'Fz': np.array([3, 6, 8, 9]) - 1,
    'Cz': np.array([4, 7, 16, 21, 41, 51, 54]) - 1,
    'Pz': np.array([34, 33, 36, 38]) - 1,
    'scalp': np.arange(60),
    'nonscalp': np.arange(60, 64)
}

_egi64a_rois = {
    'p3a': np.array([]),
    'p3b': np.array([]),
    'mmn': np.array([]),
    'cnv': np.array([]),
    'Fz': np.array([]),
    'Cz': np.array([]),
    'Pz': np.array([]),
    'scalp': np.arange(64),
    'nonscalp': None
}

_egi32a_rois = {
    'p3a': np.array([]),
    'p3b': np.array([]),
    'mmn': np.array([]),
    'cnv': np.array([]),
    'Fz': np.array([]),
    'Cz': np.array([]),
    'Pz': np.array([]),
    'scalp': np.arange(32),
    'nonscalp': None
}

_egi16a_rois = {
    'p3a': np.array([]),
    'p3b': np.array([]),
    'mmn': np.array([]),
    'cnv': np.array([]),
    'Fz': np.array([]),
    'Cz': np.array([]),
    'Pz': np.array([]),
    'scalp': np.arange(16),
    'nonscalp': None
}

_egi8a_rois = {
    'p3a': np.array([]),
    'p3b': np.array([]),
    'mmn': np.array([]),
    'cnv': np.array([]),
    'Fz': np.array([]),
    'Cz': np.array([]),
    'Pz': np.array([]),
    'scalp': np.arange(8),
    'nonscalp': None
}

_egi4a_rois = {
    'p3a': np.array([]),
    'p3b': np.array([]),
    'mmn': np.array([]),
    'cnv': np.array([]),
    'Fz': np.array([]),
    'Cz': np.array([]),
    'Pz': np.array([]),
    'scalp': np.arange(4),
    'nonscalp': None
}

_egi2a_rois = {
    'p3a': np.array([]),
    'p3b': np.array([]),
    'mmn': np.array([]),
    'cnv': np.array([]),
    'Fz': np.array([]),
    'Cz': np.array([]),
    'Pz': np.array([]),
    'scalp': np.arange(2),
    'nonscalp': None
}


_egi256_outlines = {
    'ear1': np.array([190, 191, 201, 209, 218, 217, 216, 208, 200, 190]),
    'ear2': np.array([81, 72, 66, 67, 68, 73, 82, 92, 91, 81]),
    'outer': np.array([9, 17, 24, 30, 31, 36, 45, 243, 240, 241, 242, 246, 250,
                       255, 90, 101, 110, 119, 132, 144, 164, 173, 186, 198,
                       207, 215, 228, 232, 236, 239, 238, 237, 233, 9]),
}

_egi128_outlines = {
    'ear1': np.array([99, 100, 107, 114, 113, 112, 106, 99]),
    'ear2': np.array([49, 56, 55, 48, 43, 38, 44, 49]),
    'outer': np.array([0, 7, 13, 16, 20, 24, 31, 126, 127, 47, 48, 55, 62, 67,
                       72, 80, 87, 93, 98, 106, 112, 118, 124, 125, 0]),
}


_egi64_outlines = {
    # 'ear1': np.array([22, 23, 28]),
    # 'ear2': np.array([54, 51, 46]),
    'outer': np.array([0, 4, 7, 9, 16, 62, 63, 22, 23, 28, 31, 34, 36, 38, 42,
                       46, 51, 54, 60, 61, 0]),
}

_egi_outlines = {
    64: _egi64_outlines,
    128: _egi128_outlines,
    256: _egi256_outlines
}

_egi_ch_names = {}
for i in [64, 65, 128, 129, 256, 257]:
    _egi_ch_names['{}'.format(i)] = ['E{}'.format(c) for c in range(1, i + 1)]

for i in [2, 4, 8, 16, 32, 64]:
    _egi_ch_names['{}a'.format(i)] = ['E{}'.format(c) for c in range(1, i + 1)]


# Map from 256 to 128 by using the closest electrode (euclidean distance)
# egi256 = mne.channels.read_montage('GSN-HydroCel-256')
# egi128 = mne.channels.read_montage('GSN-HydroCel-128')

# __ch_names = [x for x in egi128.ch_names if x.startswith('E')]
#
# _egi256_egi128_map = {}
# for t_ch in __ch_names:
#     t_idx = egi128.ch_names.index(t_ch)
#     t_pos = egi128.pos[t_idx]
#     idx_min = np.argmin(np.sqrt(np.sum(
#         np.power(egi256.pos - t_pos, 2), axis=-1)))
#     _egi256_egi128_map[t_ch] = egi256.ch_names[idx_min]
#

_egi256_egi128_map = {
    'E1': 'E1', 'E100': 'E61', 'E101': 'E62', 'E103': 'E63', 'E105': 'E64',
    'E107': 'E65', 'E108': 'E66', 'E110': 'E67', 'E116': 'E70', 'E118': 'E71',
    'E119': 'E72', 'E121': 'E68', 'E122': 'E69', 'E125': 'E75', 'E127': 'E76',
    'E128': 'E77', 'E129': 'E78', 'E131': 'E79', 'E132': 'E80', 'E134': 'E73',
    'E136': 'E74', 'E141': 'E85', 'E143': 'E86', 'E144': 'E87', 'E147': 'E81',
    'E148': 'E82', 'E15': 'E11', 'E150': 'E83', 'E151': 'E84', 'E152': 'E91',
    'E153': 'E92', 'E160': 'E90', 'E162': 'E97', 'E163': 'E98', 'E164': 'E93',
    'E166': 'E88', 'E167': 'E89', 'E17': 'E13', 'E170': 'E96', 'E175': 'E94',
    'E177': 'E95', 'E179': 'E101', 'E18': 'E8', 'E181': 'E102', 'E182': 'E103',
    'E184': 'E104', 'E185': 'E105', 'E186': 'E106', 'E19': 'E9',
    'E190': 'E100', 'E192': 'E108', 'E194': 'E109', 'E195': 'E110',
    'E197': 'E111', 'E198': 'E112', 'E2': 'E2', 'E20': 'E10', 'E200': 'E99',
    'E202': 'E115', 'E204': 'E116', 'E209': 'E107', 'E21': 'E16',
    'E214': 'E117', 'E215': 'E118', 'E218': 'E113', 'E219': 'E114',
    'E220': 'E121', 'E221': 'E122', 'E223': 'E123', 'E224': 'E124',
    'E226': 'E125', 'E227': 'E120', 'E228': 'E119', 'E23': 'E12',
    'E238': 'E126', 'E241': 'E127', 'E25': 'E14', 'E252': 'E128',
    'E254': 'E43', 'E255': 'E48', 'E26': 'E15', 'E27': 'E18', 'E29': 'E19',
    'E30': 'E20', 'E31': 'E17', 'E32': 'E21', 'E33': 'E22', 'E35': 'E23',
    'E36': 'E24', 'E37': 'E25', 'E4': 'E3', 'E40': 'E27', 'E41': 'E28',
    'E43': 'E29', 'E44': 'E30', 'E45': 'E31', 'E47': 'E26', 'E5': 'E4',
    'E52': 'E36', 'E53': 'E37', 'E54': 'E32', 'E55': 'E33', 'E57': 'E34',
    'E58': 'E35', 'E6': 'E5', 'E61': 'E38', 'E64': 'E40', 'E65': 'E41',
    'E66': 'E42', 'E67': 'E44', 'E69': 'E39', 'E71': 'E46', 'E73': 'E49',
    'E74': 'E45', 'E77': 'E47', 'E79': 'E53', 'E8': 'E6', 'E80': 'E54',
    'E81': 'E55', 'E84': 'E50', 'E86': 'E51', 'E87': 'E52', 'E9': 'E7',
    'E92': 'E56', 'E94': 'E57', 'E96': 'E58', 'E98': 'E59', 'E99': 'E60'}


_egi256_egi64a_map = {
    'E2': 'E1', 'E5': 'E2', 'E8': 'E3', 'E10': 'E4', 'E12': 'E5', 'E15': 'E6',
    'E18': 'E7', 'E20': 'E8', 'E21': 'E9', 'E24': 'E10', 'E26': 'E11',
    'E27': 'E12', 'E29': 'E13', 'E34': 'E14', 'E36': 'E15', 'E37': 'E16',
    'E42': 'E17', 'E44': 'E18', 'E46': 'E19', 'E47': 'E20', 'E48': 'E21',
    'E49': 'E22', 'E59': 'E23', 'E62': 'E24', 'E64': 'E25', 'E66': 'E26',
    'E68': 'E27', 'E69': 'E28', 'E76': 'E29', 'E79': 'E30', 'E81': 'E31',
    'E84': 'E32', 'E86': 'E33', 'E87': 'E34', 'E88': 'E35', 'E96': 'E36',
    'E97': 'E37', 'E101': 'E38', 'E109': 'E39', 'E116': 'E40', 'E119': 'E41',
    'E126': 'E42', 'E140': 'E43', 'E142': 'E44', 'E143': 'E45', 'E150': 'E46',
    'E153': 'E47', 'E161': 'E48', 'E162': 'E49', 'E164': 'E50', 'E170': 'E51',
    'E172': 'E52', 'E179': 'E53', 'E183': 'E54', 'E185': 'E55', 'E194': 'E56',
    'E202': 'E57', 'E206': 'E58', 'E207': 'E59', 'E210': 'E60', 'E211': 'E61',
    'E213': 'E62', 'E222': 'E63', 'E224': 'E64'}


_egi256_egi32a_map = {
    'E37': 'E1', 'E18': 'E2', 'E34': 'E3', 'E12': 'E4', 'E47': 'E5',
    'E36': 'E6', 'E21': 'E7', 'E224': 'E8', 'E2': 'E9', 'E49': 'E10',
    'E24': 'E11', 'E207': 'E12', 'E213': 'E13', 'E69': 'E14', 'E59': 'E15',
    'E81': 'E16', 'E183': 'E17', 'E202': 'E18', 'E76': 'E19', 'E79': 'E20',
    'E143': 'E21', 'E172': 'E22', 'E96': 'E23', 'E87': 'E24', 'E101': 'E25',
    'E153': 'E26', 'E170': 'E27', 'E109': 'E28', 'E116': 'E29', 'E126': 'E30',
    'E150': 'E31', 'E140': 'E32'}

_egi256_egi16a_map = {
    'E37': 'E1', 'E18': 'E2', 'E47': 'E3', 'E2': 'E4', 'E36': 'E5',
    'E224': 'E6', 'E59': 'E7', 'E183': 'E8', 'E69': 'E9', 'E202': 'E10',
    'E87': 'E11', 'E153': 'E12', 'E96': 'E13', 'E170': 'E14', 'E116': 'E15',
    'E150': 'E16'}


_egi256_egi8a_map = {
    'E36': 'E1',
    'E224': 'E2',
    'E59': 'E3',
    'E183': 'E4',
    'E87': 'E5',
    'E153': 'E6',
    'E116': 'E7',
    'E150': 'E8'}


_egi256_egi4a_map = {
    'E36': 'E1',
    'E224': 'E2',
    'E87': 'E3',
    'E153': 'E4'
}

_egi256_egi2a_map = {
    'E36': 'E1',
    'E224': 'E2'
}


_sphere_ch_names = {
    '256': ['E137', 'E26', 'E69', 'E202'],
}


def _prepare_egi_layout(ch_config):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    montage = get_montage('egi/{}'.format(ch_config))
    n_chan = int(ch_config)
    names = get_ch_names('egi/{}'.format(ch_config))

    info = mne.create_info(names, 1, ch_types='eeg')
    info.set_montage(montage)

    sphere = None
    check_ch = _sphere_ch_names.get(ch_config, None)
    if check_ch is not None:
        ch_idx = [names.index(ch) for ch in check_ch]
        pos = np.stack([info['chs'][idx]['loc'][:3] for idx in ch_idx])
        radius = np.abs(pos[[2, 3], 0]).mean()
        x = pos[0, 0]
        y = pos[-1, 1]
        z = pos[:, -1].mean()
        sphere = (x, y, z, radius)

    _, pos, _, _, _, this_sphere, clip_origin = \
        mne.viz.topomap._prepare_topomap_plot(info, 'eeg', sphere=sphere)

    outlines = {}
    codes = []
    vertices = []
    for k, v in _egi_outlines[n_chan].items():
        t_verts = pos[v, :]
        outlines[k] = (t_verts[:, 0], t_verts[:, 1])
        t_codes = 2 * np.ones(v.shape[0])
        t_codes[0] = 1
        codes.append(t_codes)
        vertices.append(t_verts)
    vertices = np.concatenate(vertices, axis=0)
    codes = np.concatenate(codes, axis=0)

    path = Path(vertices=vertices, codes=codes)

    def patch():
        return PathPatch(path, color='white', alpha=0.1)

    outlines['mask_pos'] = outlines['outer']
    outlines['patch'] = patch
    outlines['clip_radius'] = clip_origin
    return this_sphere, outlines


def register():
    define_equipment('egi', _egi_montages, _egi_ch_names, _prepare_egi_layout)
    define_rois('egi/256', _egi256_rois)
    define_rois('egi/128', _egi128_rois)
    define_rois('egi/64', _egi64_rois)
    define_rois('egi/64a', _egi64a_rois)
    define_rois('egi/32a', _egi32a_rois)
    define_rois('egi/16a', _egi16a_rois)
    define_rois('egi/8a', _egi8a_rois)
    define_rois('egi/4a', _egi4a_rois)
    define_rois('egi/2a', _egi2a_rois)
    define_map_montage('egi/256', 'egi/128', _egi256_egi128_map)
    define_map_montage('egi/256', 'egi/64a', _egi256_egi64a_map)
    define_map_montage('egi/256', 'egi/32a', _egi256_egi32a_map)
    define_map_montage('egi/256', 'egi/16a', _egi256_egi16a_map)
    define_map_montage('egi/256', 'egi/8a', _egi256_egi8a_map)
    define_map_montage('egi/256', 'egi/4a', _egi256_egi4a_map)
    define_map_montage('egi/256', 'egi/2a', _egi256_egi2a_map)
    c_path = op.abspath(op.dirname(op.realpath(__file__)))
    fname = op.join(c_path, 'data', 'egi_256_neighbours.mat')
    define_neighbor('egi/256', fname)
