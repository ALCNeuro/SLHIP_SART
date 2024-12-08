import numpy as np

from pathlib import Path
import tempfile

import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.api.io import register_suffix, _check_io_suffix
from nice_ext.equipments import get_montage


def register():
    register_module('io', 'icm/rs/raw/egi', _read_rs_raw_egi)
    register_suffix('icm/rs/raw/egi', 'icm-rs-egi.raw')
    register_module('io', 'icm/rs/mff/egi', _read_rs_mff_egi)
    register_suffix('icm/rs/mff/egi', 'icm-rs-egi.mff')
    register_module('io', 'icm/rs/mffzip/egi', _read_rs_mffzip_egi)
    register_suffix('icm/rs/mffzip/egi', 'icm-rs-egi.mff.zip')


def _read_rs_raw_egi(files, config_params):
    if not isinstance(files, list):
        files = [files]

    return _read_rs_egi_generic(files, config_params)


def _read_rs_mff_egi(files, config_params):
    if not isinstance(files, list):
        files = [files]

    return _read_rs_egi_generic(files, config_params)


def _read_rs_egi_generic(files, config_params):
    raw = mne.io.read_raw_egi(files[0], preload=True, verbose=True)
    n_eeg = 0
    for idx in range(len(raw.ch_names)):
        n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')

    replacement = {k: k.replace('EG', '')
                       .replace(' 00', '')
                       .replace(' 0', '')
                       .replace(' ', '')
                       .replace('E%d' % n_eeg, 'Cz')
                   for k in raw.ch_names}
    mne.rename_channels(raw.info, replacement)

    if n_eeg in [257, 129]:
        n_eeg -= 1
        raw.drop_channels(['Cz'])

    if 'comm' in raw.ch_names:
        raw.drop_channels(['comm'])
    if 'STI14' in raw.ch_names:
        raw.drop_channels(['STI14'])
    logger.info('Adding standard channel locations to info.')

    ch_config = 'egi/{}'.format(n_eeg)
    montage = get_montage(ch_config)

    raw.set_montage(montage)

    if n_eeg in (257, 129, 65):
        ch_pos_is_not_zero = \
            not np.all(raw.info['chs'][n_eeg - 1]['loc'][:3] == 0.0)
        assert ch_pos_is_not_zero

    raw.info['description'] = ch_config
    logger.info('Reading done')

    return raw


def _read_rs_mffzip_egi(path, config_params):
    import zipfile
    config = 'icm/rs/mffzip/egi'
    files = _check_io_suffix(path, config, multiple=False)
    fname = files[0]
    logger.info('Extracting zip file')
    zip_ref = zipfile.ZipFile(fname, 'r')
    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)
        zip_ref.extractall(tdir.as_posix())

        logger.info('Checking content of zip file')
        fnames = tdir.glob('*')
        fnames = [x for x in fnames if x.name not in ['__MACOSX']]
        fnames = [x for x in fnames if not x.name.startswith('.')]
        if len(fnames) != 1:
            raise ValueError('Wrong ZIP file content (n files = {})'.format(
                len(fnames)))

        new_fname = Path(f'{fnames[0].as_posix()}-icm-rs-egi.mff')
        fnames[0].rename(new_fname)

        raw = _read_rs_egi_generic([new_fname], config_params=config_params)
    return raw
