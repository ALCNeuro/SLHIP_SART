import numpy as np

from pathlib import Path
import tempfile

import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.api.io import register_suffix, _check_io_suffix, _check_path
from nice_ext.equipments import get_montage, get_ch_names

from .constants import (_icm_lg_event_id, _icm_lg_concatenation_event,
                        _lg_matlab_event_id_map, _arduino_trigger_map,
                        _gtec_trig_map)


def register():
    register_module('io', 'icm/lg/epochs/matlab', _read_lg_epochs_matlab)
    register_suffix('icm/lg/epochs/matlab', 'icm-lg-epochs.mat')

    register_module('io', 'icm/lg/epochs/egi', _read_lg_epochs_egi)
    register_suffix('icm/lg/epochs/egi', 'icm-lg-egi.mat')

    register_module('io', 'icm/lg/raw/egi', _read_lg_raw_egi)
    register_suffix('icm/lg/raw/egi', 'icm-lg-egi.raw')

    register_module('io', 'icm/lg_a/raw/egi', _read_lga_raw_egi)
    register_suffix('icm/lg_a/raw/egi', 'icm-lg_a-egi.raw')

    register_module('io', 'icm/lg_a/mff/egi', _read_lga_mff_egi)
    register_suffix('icm/lg_a/mff/egi', 'task-lg*_eeg.mff')

    register_module('io', 'icm/lg_a/mffzip/egi', _read_lga_mffzip_egi)
    register_suffix('icm/lg_a/mffzip/egi', 'icm-lg_a-egi.mff.zip')

    register_module('io', 'icm/lg_a/raw/bv', _read_lga_raw_bv)
    register_suffix('icm/lg_a/raw/bv', 'icm-lg_a-bv.vhdr')

    register_module('io', 'icm/lg_a/cnt/ant', _read_lga_cnt_ant)
    register_suffix('icm/lg_a/cnt/ant', 'icm-lg_a-ant.cnt')

    register_module('io', 'icm/lg_a/hdf5/gtec', _read_lga_hdf5_gtec)
    register_suffix('icm/lg_a/hdf5/gtec', 'icm-lg_a-gtec.hdf5')

    register_module('io', 'icm/lg_a/bdf/biosemi', _read_lga_bdf_bs)
    register_suffix('icm/lg_a/bdf/biosemi', 'icm-lg_a-biosemi.bdf')

    register_module('io', 'icm/lg_a/mat/egi', _read_lga_mat_egi)
    register_suffix('icm/lg_a/mat/egi', '-icm-lg_a-egi.mat')


def _read_lga_mat_egi(path, config_params):
    from scipy import io as sio
    config = 'icm/lg_a/mat/egi'
    files = _check_io_suffix(path, config, multiple=True)
    files = sorted(files)
    raws = []

    no_data_keys = [
        'ECG',
        'Body_Position',
        'EMG',
        'Resp',
        'Calibration',
        'DIN_1',
        'Impedances',
        'samplingRate'
    ]

    for fname in files:
        mc = sio.loadmat(fname, squeeze_me=True)
        data_key = [y for y in mc.keys()
                    if not any([x in y for x in no_data_keys])
                    and not y.startswith('_')][0]  # noqa
        ecg_key = [y for y in mc.keys() if y.endswith('ECG')][0]
        resp_key = [y for y in mc.keys() if y.endswith('Chest')][0]
        data = np.concatenate(
            [mc[data_key], mc[ecg_key][None, :], mc[resp_key][None, :]],
            axis=0)
        data *= 1e-6
        ch_names = ['E{}'.format(x) for x in range(1, 257)] + ['Cz']
        ch_types = ['eeg'] * 257

        ch_names += ['ECG', 'RESP', 'STI 014']
        ch_types += ['ecg', 'bio', 'stim']

        # TODO: Create Stim channel
        stim_chan = mc['DIN_1']
        stim_values = np.array([
            int(x.replace('DIN', '').replace('DI', '').replace('D', ''))
            for x in stim_chan[0, :]])
        stim_values = 255 - stim_values
        stim_samples = np.squeeze(
            np.concatenate(stim_chan[3, :][None, :])).astype(np.int)
        masked_data = stim_values & 0xF8
        new_trigger_data = np.zeros_like(data[0, :])

        # Fix values
        masked_data[masked_data < 0x40] = 0

        # Find blocks
        idx = np.where(np.ediff1d(masked_data) > 0)[0] + 1

        for block in idx:
            this_event = _icm_lg_event_id[
                _arduino_trigger_map[masked_data[block]]]
            new_trigger_data[stim_samples[block]] = this_event

        data = np.concatenate([data, new_trigger_data[None, :]], axis=0)
        sfreq = 250.0
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                               ch_types=ch_types)
        montage = mne.channels.read_montage('GSN-HydroCel-257')

        raw = mne.io.RawArray(data, info)
        raw.set_montage(montage)
        raw.drop_channels(['Cz'])
        raw.info['description'] = 'egi/256'
        raws.append(raw)
    raw = mne.io.concatenate_raws(raws)
    return raw


def _read_lg_epochs_egi(path, config_params):
    from scipy import io as sio
    config = 'icm/lg/epochs/egi'
    files = _check_io_suffix(path, config, multiple=False)

    data = []
    epochs_id = []

    # Load File
    mc = sio.loadmat(files[0])

    # Get a dict of the IDs present in the recording
    this_ids = dict()

    # Concatenate data
    for condition, value in _icm_lg_event_id.items():
        if condition in mc.keys():
            this_ids[condition] = value
            this_data = mc[condition]
            data.append(this_data)
            epochs_id.append(np.ones(this_data.shape[2], dtype=np.int) * value)

    data = np.concatenate(data, axis=2)
    epochs_id = np.concatenate(epochs_id, axis=0)
    data = np.transpose(data, [2, 0, 1])

    data *= 1e-6  # Rescale to Volts
    n_epochs, n_chans, n_times = data.shape

    events = np.c_[np.arange(1, n_epochs * n_times, n_times),
                   np.zeros(n_epochs, dtype=np.int),
                   epochs_id]
    ch_names = ['E{}'.format(x) for x in range(1, 257)] + ['Cz']
    ch_types = ['eeg'] * 257
    sfreq = 250
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    montage = get_montage('egi/257')
    epochs = mne.EpochsArray(data, info, events, event_id=this_ids, tmin=-.2)
    epochs.set_montage(montage)
    epochs.drop_channels(['Cz'])
    epochs.info['description'] = 'egi/256'
    return epochs


def _read_lg_epochs_matlab(path, config_params):
    import h5py

    config = 'icm/lg/epochs/matlab'
    files = _check_io_suffix(path, config, multiple=False)

    data = None
    epochs_id = None

    with h5py.File(files[0]) as f:
        data = np.array(f['data'])
        epochs_id = np.squeeze(np.array(f['condition'])).astype(np.int)

    if data is None or epochs_id is None:
        raise ValueError('Incorrect matlab structure for {}'.format(config))

    data = np.transpose(data, [0, 2, 1])
    data *= 1e-6  # Rescale to Volts
    n_epochs, n_chans, n_times = data.shape

    # Map event id to LG convention
    all_epochs_id = np.unique(epochs_id)
    for i in all_epochs_id:
        epochs_id[epochs_id == i] = _lg_matlab_event_id_map[i]

    events = np.c_[np.arange(1, n_epochs * n_times, n_times),
                   np.zeros(n_epochs, dtype=np.int),
                   epochs_id]
    ch_names = ['E{}'.format(x) for x in range(1, 257)] + ['Cz']
    ch_types = ['eeg'] * 257
    sfreq = 250
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    montage = get_montage('egi/257')
    epochs = mne.EpochsArray(
        data, info, events, event_id=_icm_lg_event_id, tmin=-.2)
    epochs.set_montage(montage)
    epochs.drop_channels(['Cz'])
    epochs.info['description'] = 'egi/256'
    return epochs


def _read_lg_raw_egi(files, config_params):
    if not isinstance(files, list):
        files = [files]

    return _read_lg_raw_egi_files(files, config_params)


def _read_lg_raw_egi_files(files, config_params):
    logger.info('Reading {} files'.format(len(files)))
    files.sort()
    raws = []
    for ii, fname in enumerate(files):
        raw = mne.io.read_raw_egi(fname.as_posix(), preload=True, verbose=True)
        logger.info('Creating common trigger channel')
        _check_clean_trigger(raw)
        if ii > 0:
            idx = raw.ch_names.index('STI 014')
            raw._data[idx, 0] = _icm_lg_concatenation_event
        raws.append(raw)

    # Sort by recording date
    raws = sorted(raws, key=lambda x: x.info['meas_date'])
    raw = mne.io.concatenate_raws(raws)

    logger.info('Adding standard channel locations to info.')

    n_eeg = 0
    for idx in range(len(raw.ch_names)):
        n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')

    replacement = {k: k.replace('EG', '')
                       .replace(' 00', '')
                       .replace(' 0', '')
                       .replace(' ', '')
                       .replace('E%d' % n_eeg, 'Cz')
                   for k in raw.ch_names}
    del replacement['STI 014']
    mne.rename_channels(raw.info, replacement)
    if n_eeg == 257:
        n_eeg -= 1
        raw.drop_channels(['Cz'])
    eq_config = 'egi/{}'.format(n_eeg)
    montage = get_montage(eq_config)

    raw.set_montage(montage)

    if n_eeg in (257, 129, 65):
        ch_pos_is_not_zero = \
            not np.all(raw.info['chs'][n_eeg - 1]['loc'][:3] == 0.0)
        assert ch_pos_is_not_zero

    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw


def _check_clean_trigger(raw):
    """Assign triggers to conditions and clean channel values

    Trial definition
    a series of 5 sounds
    from -200 ms of sound 1 (onset)
    to 1340 ms after the onset of that sound
    total 1540 ms

    Parameters
    ----------
    raw : instance of mne.io.egi.Raw
        The egi imported raw.
    event_id
    """

    event_id = _icm_lg_event_id
    has_hstd = (any([k in raw.ch_names for k in ['HXX1', 'HXX2']])
                or any([k in raw.ch_names for k in ['HXY1', 'HXY2']]))  # noqa
    assert has_hstd is True
    graph1, graph2 = ('HXX1', 'HXX2'), ('HXY1', 'HXY2')
    rest = ('XXX1', 'XXX2', 'XXY1', 'XXY2')
    has_requirements = (any([any([k in raw.ch_names for k in graph1]),
                             any([k in raw.ch_names for k in graph2])])
                        and any([k in raw.ch_names for k in rest]))  # noqa
    if not has_requirements:
        logger.warning('This file should be discarded, since no complete'
                       'local global was found')
    is_hstd = (any([k in raw.ch_names for k in ['HXX1', 'HXX2']])
               and not any(  # noqa
                   [k in raw.ch_names for k in ['HXY1', 'HXY2']]))

    if 'STI 014' not in raw.ch_names:
        stim_data = np.zeros((1, len(raw.times)))
        info = mne.create_info(['STI 014'], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        raw.add_channels([stim_raw], force_update_info=True)
    trigger = raw._data[raw.ch_names.index('STI 014')]
    for k, v in raw.event_id.items():
        if k in ('HXX1', 'HXX2'):
            new_val = event_id['HSTD']
        elif k in ('HXY1', 'HXY2'):
            new_val = event_id['HDVT']
        elif k in ('XXX1', 'XXX2') and is_hstd:
            new_val = event_id['LSGS']
        elif k in ('XXX1', 'XXX2') and not is_hstd:
            new_val = event_id['LSGD']
        elif k in ('XXY1', 'XXY2') and is_hstd:
            new_val = event_id['LDGD']
        elif k in ('XXY1', 'XXY2') and not is_hstd:
            new_val = event_id['LDGS']
        # just set to zero if event not in target events
        elif k not in graph1 + graph2 + rest:
            new_val = 0
        value_index = np.where(trigger == v)
        trigger[value_index] = new_val

    # remove old event channels and make runs mergable
    drop_channels = [k for k in raw.ch_names
                     if not k.startswith('E') and k != 'STI 014']
    # add trigger for concat
    raw.drop_channels(drop_channels)


def fix_reverse(x):
    if x == 0:
        return 0
    y = 0
    y += (1 - (x & 0x1)) << 7
    y += (x & 0x1) << 6
    y += (x & 0x2) << 4
    y += (x & 0x4) << 2
    y += (x & 0x8)
    y += (x & 0x10) >> 2
    y += (x & 0x20) >> 4
    y += (x & 0x20) >> 5
    return y


def _read_lga_egi_generic(files, config_params):
    logger.info('Reading {} files'.format(len(files)))
    raws = []
    for fname in files:
        fname = _check_path(fname)
        raw = mne.io.read_raw_egi(fname.as_posix(), preload=True, verbose=True)
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
            if config_params.get('fix_reverse', False):
                dvalue = fix_reverse(dvalue)
            if dvalue >= 192:
                dvalue = 0
            stim_data[idx] = dvalue

        if config_params.get('fix_reverse', False):
            idx = np.where(stim_data != 0)[0]
            fix_data = stim_data[idx]
            idx_128 = np.where(fix_data == 128)[0]
            fix_data[idx_128] = 0
            fix_data[idx_128 + 1] = 0
            stim_data[idx] = fix_data
        stim_data = stim_data.astype(np.int)
        masked_data = stim_data & 0xF8
        new_trigger_data = np.zeros_like(masked_data)

        # Fix inconsistent values
        masked_data[masked_data < 0x3F] = 0

        # Fix consecutive samples with same trigger value
        repeated = np.logical_and(
            masked_data[:-1] != 0, np.ediff1d(masked_data) == 0)
        repeated = np.where(repeated)[0] + 1
        masked_data[repeated] = 0

        # Find blocks
        idx = np.where(masked_data != 0)[0]
        if np.median(np.diff(idx)) / raw.info['sfreq'] <= 0.100:
            logger.info('Grouping by 10 triggers (old arduino)')
            idx = idx.reshape(-1, 10)[:, 0]  # 10 events by block

        for block in idx:
            t_block = block  # It was block[0]
            this_event_value = masked_data[t_block]
            if this_event_value in _arduino_trigger_map:
                this_event = _icm_lg_event_id[
                    _arduino_trigger_map[this_event_value]]
                new_trigger_data[t_block] = this_event

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
        raw._data[stim] = new_trigger_data
        raw.drop_channels(dnames)

        raws.append(raw)
    raws = sorted(raws, key=lambda x: x.info['meas_date'])
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

    to_drop = [x for x in raw.ch_names if x not in montage.ch_names
               and x != 'STI 014']  # noqa

    raw.drop_channels(to_drop)
    raw.set_montage(montage)
    if n_eeg in (257, 129, 65):
        ch_pos_is_not_zero = \
            not np.all(raw.info['chs'][n_eeg - 1]['loc'][:3] == 0.0)
        assert ch_pos_is_not_zero

    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw


def _read_lga_raw_egi(files, config_params):
    if not isinstance(files, list):
        files = [files]

    return _read_lga_egi_generic(files, config_params)


def _read_lga_mff_egi(files, config_params):
    if not isinstance(files, list):
        files = [files]

    return _read_lga_egi_generic(files, config_params)


def _read_lga_mffzip_egi(path, config_params):
    import zipfile
    config = 'icm/lg_a/mffzip/egi'
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

        new_fname = Path(f'{fnames[0].as_posix()}-icm-lg_a-egi.mff')
        fnames[0].rename(new_fname)

        raw = _read_lga_egi_generic([new_fname], config_params=config_params)
    return raw


def _read_lga_raw_bv(path, config_params):

    config = 'icm/lg_a/raw/bv'
    files = _check_io_suffix(path, config, multiple=True)
    logger.info('Reading {} files'.format(len(files)))
    raws = []
    for fname in files:
        raw = mne.io.read_raw_brainvision(fname, preload=True, verbose=True)
        stim_chan = mne.pick_channels(raw.ch_names, ['STI 014'])[0]
        stim_data = raw._data[stim_chan, :]
        stim_data = stim_data.astype(np.int) & 0xF8
        raw._data[stim_chan, :] = stim_data.astype(np.float)
        events = mne.find_events(raw)
        new_data = np.zeros_like(raw._data[stim_chan, :])
        to_use = events[::10]
        for key, value in _arduino_trigger_map.items():
            idx = to_use[to_use[:, 2] == key, 0]
            new_data[idx] = _icm_lg_event_id[value]
        raw._data[stim_chan, :] = new_data
        raws.append(raw)
    raws = sorted(raws, key=lambda x: x.info['meas_date'])
    raw = mne.io.concatenate_raws(raws)
    eq_config = 'bv/32'
    montage = mne.channels.read_montage(eq_config)

    raw.set_montage(montage)

    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw


def _read_lga_cnt_ant(path, config_params):

    config = 'icm/lg_a/cnt/ant'
    files = _check_io_suffix(path, config, multiple=False)
    logger.info('Reading {} files'.format(len(files)))

    raw = mne.io.read_raw_cnt(files[0], montage='standard_1020', eog='auto',
                              preload=True)
    idx = mne.pick_types(raw.info, eeg=True)
    eeg_chs = [raw.ch_names[x] for x in idx if not
               raw.ch_names[x].startswith('BIP')]
    n_chans = len(eeg_chs)
    logger.info('Autodetected {} EEG channels'.format(n_chans))

    eq_config = 'ant/{}'.format(n_chans)
    montage_name = get_montage(eq_config)
    raw.set_montage(montage_name)
    to_keep = get_ch_names(eq_config) + ['STI 014']
    raw.pick_channels(to_keep)

    logger.info('Remapping trigger values')
    stim_idx = raw.ch_names.index('STI 014')
    stim_data = raw._data[stim_idx]
    new_stim_data = np.zeros_like(stim_data)
    for t_val in np.unique(stim_data).astype(np.int):
        if t_val == 0:
            continue
        t_type = _arduino_trigger_map[t_val & 0xF8]
        new_stim_data[stim_data == t_val] = _icm_lg_event_id[t_type]

    raw._data[stim_idx] = new_stim_data
    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw


def _read_lga_hdf5_gtec(path, config_params):
    import h5py

    config = 'icm/lg_a/hdf5/gtec'
    files = _check_io_suffix(path, config, multiple=False)
    logger.info('Reading {} files'.format(len(files)))

    with h5py.File(files[0]) as f:
        data = f['RawData']['Samples'].value.T * 1e-6
        sfreq = 512.0  # Fixed
        sample_times = np.squeeze(f['AsynchronData']['Time'].value)
        sample_values = np.squeeze(f['AsynchronData']['Value'].value)
        stim_data = np.zeros((data.shape[1]), dtype=np.float)
        stim_data_lg = np.zeros((data.shape[1]), dtype=np.float)
        for idx, val in zip(sample_times, sample_values):
            stim_data[idx] += val
        for k, v in _gtec_trig_map.items():
            mask = stim_data == k
            if np.any(mask):
                stim_data_lg[mask] = _icm_lg_event_id[_arduino_trigger_map[v]]

    data = np.concatenate((data, stim_data_lg[None, :]), axis=0)
    eq_config = 'gtec/{}'.format(data.shape[1])
    ch_names = get_ch_names(eq_config) + ['STI 014']
    ch_types = ['eeg'] * len(ch_names) + ['stim']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    montage = get_montage(eq_config)
    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)
    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw


def _read_lga_bdf_bs(path, config_params):
    config = 'bsas/lg/bdf/biosemi'
    files = _check_io_suffix(path, config, multiple=False)
    raw = mne.io.read_raw_edf(files[0], preload=True, verbose=True)

    ch_types = {x: 'eeg' for x in raw.ch_names}

    ch_types['EXG1'] = 'eog'
    ch_types['EXG2'] = 'eog'
    ch_types['EXG3'] = 'eog'
    ch_types['EXG4'] = 'eog'
    ch_types['EXG5'] = 'misc'   # Mastoide reference
    ch_types['EXG6'] = 'misc'
    ch_types['EXG7'] = 'ecg'
    ch_types['EXG8'] = 'ecg'
    ch_types['STI 014'] = 'stim'

    raw.set_channel_types(ch_types)

    # TODO: Make bipolar channels

    # TODO: Change A1 to correponding channel due to bad electrode
    raw.drop_channels(
        ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
    # Change Stimulator codes for LG IDS
    stim = mne.pick_channels(raw.info['ch_names'], include=['STI 014'])
    stim_data = raw._data[stim].astype(np.int)
    masked_data = stim_data & 0xF8
    new_trigger_data = np.zeros_like(masked_data)

    for value, kind in _arduino_trigger_map.items():
        new_value = _icm_lg_event_id[kind]
        new_trigger_data[masked_data == value] = new_value

    raw._data[stim] = new_trigger_data
    eq_config = 'biosemi/128'
    montage = get_montage(eq_config)
    raw.set_montage(montage)
    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw
