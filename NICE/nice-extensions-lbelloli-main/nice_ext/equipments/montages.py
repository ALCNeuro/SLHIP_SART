# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import mne
from mne.utils import logger

_montages = {}
_ch_names = {}
_layouts = {}
_neighbors = {}

# This one contains montages translations
# It is a dict of dicts, so for every montage (src) it has a dict
# with montages (dst) that can be translated too.  The value of the last
# dict should be the channel rename dict
_montage_map = {}


def define_equipment(equipment, montages, ch_names, layout_fun=None):
    """
        Defines an equipement montage, ch_channels and layout function 

        Args:
            equipement (str): The equipement name
            ch_channels (list(str)): The names of the EEG channels
            layout_fun (callable([ch_config],)): Optional. A function that consumes a 
            channels configuration and returns a layout of the channels positions  
    """
    
    if equipment in _montages:
        logger.warning('{} already defined as equipment'.format(equipment))
    else:
        logger.info('Defining {}'.format(equipment))
    _montages[equipment] = montages
    _ch_names[equipment] = ch_names
    _layouts[equipment] = layout_fun


def define_neighbor(montage, fname):
    """
    Defines a montage neighbor file to parse with MNE.channels.adjacency.

    Args:
        montage (str): The montage to define the neighbors
        fname (str): The file name. Example: 'neuromag306mag',
        'neuromag306planar', 'ctf275', 'biosemi64', etc.
    """
    if montage in _neighbors:
        logger.warning('Neighbors for {} already defined'.format(montage))
    else:
        logger.info('Defining {} neighbors'.format(montage))
    _neighbors[montage] = fname


def define_map_montage(src, dst, ch_names):
    """
    Defines a new montage map from src to dst.

    Args:
        src (str): The name of the source montage to translate from
        dst (str): The name of the final montage where to transle to
        ch_names (dict(str,str)): The map of channels name from the src montage 
        to the dst montage  
    """
    if src in _montage_map:
        if dst in _montage_map[src]:
            logger.warning('{} already translated to {}'.format(src, dst))
    else:
        _montage_map[src] = {}
    _montage_map[src][dst] = ch_names


def map_montage(inst, dst):
    """
    Translates a source montage according to channels' map and returns only the 
    channels of the new montage.

    Args:
        inst (TODO: look what is the montage type): The original montage
        dst (str): The name of the already registered destination montage

    Returns:
        montage (instance of DigMontage): A new montage with the destination
        channels translated from the original montage.
    
    Raises:
        ValueError: source montage is not defined in montage maps
        ValueError: source to destination montage map is not defined
    """

    src = inst.info['description']
    
    # Lao modification: Check if the montage maps is defined
    if src not in _montage_map:
        raise ValueError(f'There is no montage map for src {src},'
                          ' the options are:'
                          '\n\t'.join(['', *_montage_map.keys()]))
    
    if dst not in _montage_map[src]:
        raise ValueError(f'There is no montage map for dst {dst} in src {src},'
                          ' the options are:'
                          '\n\t'.join(['', *_montage_map[src].keys()]))
    
    logger.info('Mapping montage from {} to {}'.format(src, dst))
    
    rename = _montage_map[src][dst]
    to_keep = list(rename.keys())
    translated = inst.copy().pick_channels(to_keep)
    translated.rename_channels(rename)
    translated.info['description'] = dst
    return translated.pick_channels(get_ch_names(dst))


def _check_get_eq_config(config):
    """
    Checks if the equipement defined in config exist or not and return the 
    equipement name and channels config name.

    Args:
        config (str): The montage config formatted as <equipement_name>/<channels_config_name>

    Returns:
        split_montage (tuple(str,str)): a tupple with (equipement_name, channels_config_name)

    Raises:
        ValueError: equipement not defined
    """
    equipment = config.split('/')[0]
    ch_config = config.split('/')[1]
    if equipment not in _montages:
        raise ValueError('{} not defined as an equipment'.format(equipment))
    return equipment, ch_config


def get_ch_names(config):
    """
    Obtains the montage channel names from a montage config string

    Args:
        config (str): The montage config formatted as <equipement_name>/<channels_config_name>

    Returns:
        ch_names (list(str)): The names of the EEG channels
    """
    equipment, ch_config = _check_get_eq_config(config)
    eq = _ch_names[equipment]
    ch_names = None
    if eq is not None and eq[ch_config] is not None:
        ch_names = eq[ch_config]
    if ch_names is None:
        ch_names = get_montage(config).ch_names
    return ch_names


def get_montage_name(config):
    """
    Return the name of the montage specified in the config parameter

    Args:
        config (str): The montage config formatted as <equipement_name>/<channels_config_name>
    
    Returns:
        name (str): The name of the montage
    
    Raises:
        ValueError: equipment/channels_config_name not a defined montage
    """
    equipment, ch_config = _check_get_eq_config(config)
    eq = _montages[equipment]
    if ch_config not in eq:
        raise ValueError('Montage not defined for {} with {} '
                         'config'.format(equipment, ch_config))

    return eq[ch_config]


def get_montage(config):
    """
    Creates a DigMontage out of a montage config string

    Args:
        config (str): The montage config formatted as <equipement_name>/<channels_config_name>
    
    Returns:
        montage (instance of DigMontage): A montage created from the config param
    """
    montage_name = get_montage_name(config)
    # m_path = None
    # if config.startswith('egi'):
    #     out = mne.channels.read_montage(montage_name, path=m_path, unit='cm')
    # else:
    #     out = mne.channels.read_montage(montage_name, path=m_path)
    return mne.channels.make_standard_montage(montage_name)


def prepare_layout(config, info=None, return_info=False):
    layout_fun = None
    if config != 'head':
        equipment, ch_config = _check_get_eq_config(config)
        layout_fun = _layouts[equipment]
    if layout_fun is not None:
        sphere, outlines = layout_fun(ch_config)
        # outlines['clip_radius'] = (0, 0)  # (0.5, 0.5)
    else:
        # layout = mne.channels.make_eeg_layout(info)
        sphere = None
        outlines = 'head'
    # if info is not None:
    #     idx = np.array([layout.names.index(x) for x in info['ch_names'] if
    #                     x not in info['bads']])
    #     layout.names = [layout.names[x] for x in idx]
    #     layout.ids = idx
    #     layout.pos = layout.pos[idx, :]
    out = sphere, outlines
    if return_info is True:
        if info is None:
            ch_names = get_ch_names(config)
            montage = get_montage(config)
            info = mne.create_info(
                ch_names, 1, ch_types='eeg', verbose=None)
            info.set_montage(montage)
        out = sphere, outlines, info
    return out


def get_ch_adjacency(epochs):
    montage = epochs.info['description']
    if montage in _neighbors:
        fname = _neighbors[montage]
        adjacency, ch_names = mne.channels.read_ch_adjacency(fname)
        picks = [ch_names.index(v) for v in epochs.ch_names if v in ch_names]
        adjacency, ch_names = mne.channels.read_ch_adjacency(
            fname, picks=picks)
    else:
        logger.warning(f'Neighbors for {montage} not defined. Using None')
        adjacency = None
    return adjacency


def get_ch_adjacency_montage(config, pick_names=None):
    if config in _neighbors:
        fname = _neighbors[config]
        adjacency, ch_names = mne.channels.read_ch_adjacency(fname)
        if pick_names is not None:
            picks = [ch_names.index(v) for v in pick_names if v in ch_names]
            adjacency, ch_names = mne.channels.read_ch_adjacency(
                fname, picks=picks)
    else:
        logger.warning(f'Neighbors for {config} not defined. Using None')
        adjacency = None
    return adjacency
