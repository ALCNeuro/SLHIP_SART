import numpy as np

import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.api.io import register_suffix, _check_io_suffix


def register():
    register_module('io', 'pitie/rs/raw/edf', _read_rs_raw_edf)
    register_suffix('pitie/rs/raw/edf', 'pitie-rs-raw.edf')


def _read_rs_raw_edf(path, config_params):
    config = 'pitie/rs/raw/edf'
    files = _check_io_suffix(path, config, multiple=False)
    logger.info('Reading from {}'.format(files[0]))
    raw = mne.io.read_raw_edf(files[0], preload=True, verbose=True)
    remap = dict()
    for name in raw.ch_names:
        for key in ('ECG', 'EOG', 'EMG', 'EEG'):
            if name.startswith(key):
                remap[name] = key.lower()
        if name not in remap:
            if name.startswith('STI') or name.startswith('MKR'):
                remap[name] = 'stim'
            else:
                remap[name] = 'misc'
    raw.set_channel_types(remap)

    eeg_rename = {x: x.split(' ')[1] for x in raw.ch_names
                  if x.startswith('EEG')}
    raw.rename_channels(eeg_rename)

    # Apply montage
    logger.info('Adding standard channel locations to info.')
    montage = mne.channels.read_montage('standard_1020')
    raw.set_montage(montage)
    raw.info['description'] = 'pitie_1020'
    logger.info('Reading done')
    return raw
