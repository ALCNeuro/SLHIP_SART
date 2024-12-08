import numpy as np

import os
import os.path as op
from glob import glob

import mne
from mne.utils import logger
from mne.utils import _TempDir

from nice_ext.api.modules import register_module
from nice_ext.api.io import register_suffix, _check_io_suffix
from nice_ext.equipments import get_montage


def register():
    register_module('io', 'icm/mcp/raw/egi', _read_mcp_raw_egi)
    register_suffix('icm/mp/raw/egi', 'icm-mcp-egi.raw')

    register_module('io', 'icm/mcp/mff/egi', _read_mcp_mff_egi)
    register_suffix('icm/mcp/mff/egi', 'icm-mcp-egi.mff')

    register_module('io', 'icm/mcp/mffzip/egi', _read_mcp_mffzip_egi)
    register_suffix('icm/mcp/mffzip/egi', 'icm-mcp-egi.mff.zip')


def _read_mcp_raw_egi(path, config_params):
    config = 'icm/mcp/raw/egi'
    raw = None
    files = _check_io_suffix(path, config, multiple=True,
                             missing_allowed=True)
    if len(files) > 0:
        raw = _read_mcp_egi_generic(files, config_params)

    if raw is None:
        raise ValueError('No files for icm/lg/raw/egi config')
    return raw


def _read_mcp_egi_generic(files, config_params):
    logger.info('Reading {} files'.format(len(files)))
    raws = []
    for fname in files:
        raw = mne.io.read_raw_egi(fname, preload=True, verbose=True)
        dnames = [x for x in raw.ch_names if x.startswith('D')]
        ttl_fix = config_params.get('ttl_fix', 'auto')
        if ttl_fix == 'auto':
            ttl_fix = True
            if 'DIN3' in raw.ch_names:
                din3 = mne.pick_channels(raw.ch_names, ['DIN3'])[0]
                if np.sum(raw._data[din3, :]) > 10:
                    ttl_fix = False
            logger.info('{} AUTO Reversing TTL values'.format(
                '' if ttl_fix else 'NOT '))
        else:
            logger.info('{}Reversing TTL values'.format(
                '' if ttl_fix else 'NOT '))
        values = [int(x.replace('DIN', '').replace('DI', '').replace('D', ''))
                  for x in dnames]
        if ttl_fix:
            values = 255 - np.array(values)
        stim_data = np.zeros_like(raw._data[raw.ch_names.index(dnames[0]), :])

        for dchan, dvalue in zip(dnames, values):
            if dvalue == 0:
                continue
            ddata = raw._data[raw.ch_names.index(dchan), :]
            idx = np.where(ddata != 0)[0]
            stim_data[idx] = dvalue

        stim_data = stim_data.astype(np.int)

        if 'sync' in raw.ch_names:
            raw.drop_channels(['sync'])
        if 'epoc' in raw.ch_names:
            raw.drop_channels(['epoc'])
        if 'comm' in raw.ch_names:
            raw.drop_channels(['comm'])

        # Fix for when STI 014 is not available due to overlapping events
        if 'STI 014' not in raw.ch_names:
            stim_name = [
                x['ch_name'] for x in raw.info['chs']
                if x['kind'] == mne.io.constants.FIFF.FIFFV_STIM_CH][0]
            mne.rename_channels(raw.info, {stim_name: 'STI 014'})
            dnames.remove(stim_name)
        stim = mne.pick_channels(raw.info['ch_names'], include=['STI 014'])
        raw._data[stim] = stim_data
        raw.drop_channels(dnames)

        raws.append(raw)
    raws = sorted(raws, key=lambda x: x.info['meas_date'][0])
    raw = mne.io.concatenate_raws(raws)

    n_eeg = 0
    for idx in range(len(raw.ch_names)):
        n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')
    replacement = {k: k.replace('EG', '')
                       .replace(' 00', '')
                       .replace(' 0', '')
                       .replace(' ', '')
                   for k in raw.ch_names}

    if n_eeg == 257:
        if 'EEG 257' in raw.ch_names:
            replacement['EEG 257'] = 'Cz'
        elif 'E257' in raw.ch_names:
            replacement['E257'] = 'Cz'
    del replacement['STI 014']
    mne.rename_channels(raw.info, replacement)
    if 'Cz' in raw.ch_names:
        n_eeg -= 1
        raw.drop_channels(['Cz'])

    eq_config = 'egi/{}'.format(n_eeg)
    montage = get_montage(eq_config)

    to_drop = [x for x in raw.ch_names
               if x not in montage.ch_names and x != 'STI 014']

    raw.drop_channels(to_drop)
    raw.set_montage(montage)
    if n_eeg in (257, 129, 65):
        ch_pos_is_not_zero = \
            not np.all(raw.info['chs'][n_eeg - 1]['loc'][:3] == 0.0)
        assert ch_pos_is_not_zero

    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw


def _read_mcp_mff_egi(path, config_params):
    config = 'icm/mcp/mff/egi'
    files = _check_io_suffix(path, config, multiple=True)
    raw = _read_mcp_egi_generic(files, config_params)
    return raw


def _read_mcp_mffzip_egi(path, config_params):
    import zipfile
    config = 'icm/mcp/mffzip/egi'
    files = _check_io_suffix(path, config, multiple=False)
    tdir = _TempDir()
    fname = files[0]
    logger.info('Extracting zip file')
    zip_ref = zipfile.ZipFile(fname, 'r')
    zip_ref.extractall(tdir)

    logger.info('Checking content of zip file')
    fnames = glob(op.join(tdir, '*'))
    fnames = [x for x in fnames if op.basename(x) not in ['__MACOSX']]
    fnames = [x for x in fnames if not op.basename(x).startswith('.')]
    if len(fnames) != 1:
        raise ValueError('Wrong ZIP file content (n files = {})'.format(
            len(fnames)))

    new_fname = fnames[0] + 'icm-mcp-egi.mff'
    os.rename(fnames[0], new_fname)

    raw = _read_mcp_mff_egi(op.dirname(new_fname), config_params=config_params)
    return raw
