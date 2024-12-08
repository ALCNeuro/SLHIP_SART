# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from mne.utils import logger

from .modules import _get_module_func
from ..equipments import map_montage

class ValidationError(Exception):
    def __init__(self, message):            
        super().__init__(message)


def preprocess(instance, config='default', config_params=None):
    """
    Process the instance with the preprocessing function specified in config 
    using the config_params

    Args:
        instance (TODO: Check the type): The instance to preprocess
        config (str): The string key of the already registered preprocess function
        config_params (dict(str,Any)): Optional. The parameters to use in the 
        preprocess function

    Returns:
        Preprocessed epochs (MNE.Epochs): 
        summary (dict(str, any)): A dictionary with information like the found bad channels of epochs. Only returned if the preprocess function returns a summary 
    """
    logger.info('Preprocessing from {} config'.format(config))
    out = None
    if config_params is None:
        config_params = {}
    func = _get_module_func('preprocess', config)
    out = func(instance, config_params=config_params)

    new_montage = config_params.get('map_montage', None)
    if new_montage is not None:
        if isinstance(out, tuple):
            epochs, summary = out
            new_epochs = map_montage(epochs, new_montage)
            out = (new_epochs, summary)
        else:
            out = map_montage(out, new_montage)
    return out


def _check_min_events(epochs, min_events):
    n_orig_epochs = len([x for x in epochs.drop_log if 'IGNORED' not in x])
    if isinstance(min_events, float):
        logger.info('Using relative min_events: {} * {} = {} '
                    'epochs remaining to reject preprocess'.format(
                        min_events, n_orig_epochs,
                        int(n_orig_epochs * min_events)))
        min_events = int(n_orig_epochs * min_events)

    epochs_remaining = len(epochs)
    if epochs_remaining < min_events:
        msg = ('Can not clean data. Only {} out of {} epochs '
               'remaining.'.format(epochs_remaining, n_orig_epochs))
        logger.error(msg)
        raise ValidationError(msg)


def _check_min_channels(epochs, bad_channels, min_channels):
    if isinstance(min_channels, float):
        logger.info('Using relative min_channels: {} * {} = {} '
                    'channels remaining to reject preprocess'.format(
                        min_channels, epochs.info['nchan'],
                        epochs.info['nchan'] * min_channels))
        min_channels = int(epochs.info['nchan'] * min_channels)

    chans_remaining = epochs.info['nchan'] - len(bad_channels)
    if chans_remaining < min_channels:
        msg = ('Can not clean data. Only {} out of {} channels '
               'remaining.'.format(chans_remaining, epochs.info['nchan']))
        logger.error(msg)
        raise ValidationError(msg)
