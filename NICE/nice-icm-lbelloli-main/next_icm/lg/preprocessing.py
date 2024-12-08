import numpy as np

import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.api.preprocessing import _check_min_channels, _check_min_events
from nice_ext.equipments import _egi_filter, _bv_filter, _bs_filter
from nice_ext.algorithms.adaptive import _adaptive_egi

from .constants import (_icm_lg_event_id, _icm_lg_concatenation_event)


def register():
    register_module('preprocess', 'icm/lg/raw/egi',
                    _preprocess_lg_raw_egi)
    register_module('preprocess', 'icm/lg/raw/egi/late',
                    _preprocess_lg_raw_egi_late)
    register_module('preprocess', 'icm/lg/epochs/matlab',
                    _preprocess_lg_epochs_matlab)
    register_module('preprocess', 'icm/lg/raw/bv', _preprocess_lg_raw_bv)
    register_module('preprocess', 'icm/lg/cnt/ant', _preprocess_lg_cnt_ant)
    register_module('preprocess', 'icm/lg/hdf5/gtec',
                    _preprocess_lg_hdf5_gtec)


def _preprocess_lg_epochs_matlab(epochs, config_params):
    from nice_ext.api.preprocessing import (_check_min_events,
                                            _check_min_channels)
    reject = config_params.get('reject', None)
    min_events = config_params.get('min_events', 200)
    min_channels = config_params.get('min_channels', 0.7)
    n_epochs_bad_ch = config_params.get('n_epochs_bad_ch', 0.5)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 4)
    max_iter = config_params.get('max_iter', 4)

    if reject is None:
        reject = {'eeg': 100e-6}

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])

    bad_channels, bad_epochs = _adaptive_egi(
        epochs, reject, n_epochs_bad_ch=n_epochs_bad_ch,
        n_channels_bad_epoch=n_channels_bad_epoch,
        zscore_thresh=zscore_thresh, max_iter=max_iter,
        summary=summary)
    epochs.info['bads'].extend(bad_channels)
    logger.info('found bad channels: {} {}'.format(
        len(bad_channels), str(bad_channels)))

    _check_min_events(epochs, min_events)
    _check_min_channels(epochs, min_channels)

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    epochs.interpolate_bads(reset_bads=True)
    out = epochs
    if summary is not None:
        out = out, summary
    return out


def _preprocess_lg_raw_egi(raw, config_params):
    n_jobs = config_params.get('n_jobs', 1)
    reject = config_params.get('reject', None)
    min_events = config_params.get('min_events', 0.3)
    min_channels = config_params.get('min_channels', 0.7)
    n_epochs_bad_ch = config_params.get('n_epochs_bad_ch', 0.5)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 4)
    max_iter = config_params.get('max_iter', 4)
    tmin = config_params.get('tmin', -.2)
    tmax = config_params.get('tmax', 1.34)
    # run_ica = config_params.get('ica', False)
    autoreject = config_params.get('autoreject', False)
    baseline = config_params.get('baseline', (None, 0))

    if reject is None:
        reject = {'eeg': 100e-6}

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])

    # Filter
    _egi_filter(raw, config_params, summary=summary, n_jobs=n_jobs)

    # Cut
    events = mne.find_events(raw, shortest_event=1)
    all_id = _icm_lg_event_id
    found_id = np.unique(events[:, 2])
    this_id = {k: v for k, v in all_id.items() if v in found_id}

    if summary is not None:
        summary['steps'].append(dict(step='Epochs',
                                params={'baseline': baseline}))
    epochs = mne.Epochs(raw, events, this_id, tmin=tmin, tmax=tmax,
                        preload=True, reject=None, picks=None,
                        baseline=baseline, verbose=False)

    ch_idx = epochs.ch_names.index('STI 014')
    concat_idx = []

    for ii, e in enumerate(epochs):
        if _icm_lg_concatenation_event in e[ch_idx]:  # constant
            concat_idx.append(ii)
    epochs.drop(concat_idx, reason='concatenation')
    epochs.drop_channels(['STI 014'])

    if autoreject is True:
        logger.info('Using autoreject')
        from autoreject import AutoReject
        ar = AutoReject()
        epochs_clean = ar.fit_transform(epochs)
        reject_log = ar.get_reject_log(epochs)
        if summary is not None:
            summary['autoreject'] = reject_log
            summary['steps'].append(
                dict(step='Autoreject',
                     params={'n_interpolate': ar.n_interpolate_['eeg'],
                             'consensus_perc': ar.consensus_['eeg']},
                     bad_epochs=np.where(reject_log.bad_epochs)[0]))
        _check_min_events(epochs, min_events)
        logger.info('found bad epochs: {} {}'.format(
            np.sum(reject_log.bad_epochs),
            np.where(reject_log.bad_epochs)[0]))
        epochs = epochs_clean

        ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
        epochs.add_proj(ref_proj)
        epochs.apply_proj()

    else:
        bad_channels, bad_epochs = _adaptive_egi(
            epochs, reject, n_epochs_bad_ch=n_epochs_bad_ch,
            n_channels_bad_epoch=n_channels_bad_epoch,
            zscore_thresh=zscore_thresh, max_iter=max_iter,
            summary=summary)
        epochs.info['bads'].extend(bad_channels)
        logger.info('found bad channels: {} {}'.format(
            len(bad_channels), str(bad_channels)))

        _check_min_channels(epochs, bad_channels, min_channels)
        _check_min_events(epochs, min_events)

        ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
        epochs.add_proj(ref_proj)
        epochs.apply_proj()

        epochs.interpolate_bads(reset_bads=True)

    # Resample to 250 since all the pipeline is prepared for that
    if raw.info['sfreq'] != 250:
        if summary is not None:
            summary['steps'].append(dict(step='Resample',
                                    params={'freq': 250, 'npad': 'auto'}))
        logger.info('Resampling to 250 Hz')
        raw.resample(250, npad='auto')
        logger.info('Resampling done')

    out = epochs
    if summary is not None:
        out = out, summary
    return out


def _preprocess_lg_raw_egi_late(instance, config_params):
    config_params.update(tmin=.6)
    config_params.update(tmax=1.4)
    out = _preprocess_lg_raw_egi(instance, config_params)
    if 'summary' in config_params:
        epochs = out[0]
    else:
        epochs = out
    epochs.times = epochs.times - 0.8
    return out


def _preprocess_lg_raw_bv(raw, config_params):
    n_jobs = config_params.get('n_jobs', 1)
    reject = config_params.get('reject', None)
    min_events = config_params.get('min_events', 0.3)
    min_channels = config_params.get('min_channels', 0.7)
    n_epochs_bad_ch = config_params.get('n_epochs_bad_ch', 0.5)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 6)
    max_iter = config_params.get('max_iter', 4)
    tmin = config_params.get('tmin', -.2)
    tmax = config_params.get('tmax', 1.34)

    if reject is None:
        reject = {'eeg': 150e-6}

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])

    # Filter
    _bv_filter(raw, config_params, summary=summary, n_jobs=n_jobs)

    # Cut
    events = mne.find_events(raw, shortest_event=1)
    all_id = _icm_lg_event_id
    found_id = np.unique(events[:, 2])
    this_id = {k: v for k, v in all_id.items() if v in found_id}

    baseline = (None, 0)
    epochs = mne.Epochs(raw, events, this_id, tmin=tmin, tmax=tmax,
                        preload=True, reject=None, picks=None,
                        baseline=baseline, verbose=False)

    epochs.drop_channels(['STI 014'])

    bad_channels, bad_epochs = _adaptive_egi(
        epochs, reject, n_epochs_bad_ch=n_epochs_bad_ch,
        n_channels_bad_epoch=n_channels_bad_epoch,
        zscore_thresh=zscore_thresh, max_iter=max_iter,
        summary=summary)
    epochs.info['bads'].extend(bad_channels)
    logger.info('found bad channels: {} {}'.format(
        len(bad_channels), str(bad_channels)))

    _check_min_channels(epochs, bad_channels, min_channels)
    _check_min_events(epochs, min_events)

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    epochs.interpolate_bads(reset_bads=True)

    # Resample to 250 since all the pipeline is prepared for that
    if epochs.info['sfreq'] != 250:
        logger.info('Resampling from {} to 250 Hz'.format(
            epochs.info['sfreq']))
        epochs.resample(250, npad='auto')
    out = epochs
    if summary is not None:
        out = out, summary
    return out


def _preprocess_lg_cnt_ant(raw, config_params):
    n_jobs = config_params.get('n_jobs', 1)
    reject = config_params.get('reject', None)
    min_events = config_params.get('min_events', 0.3)
    min_channels = config_params.get('min_channels', 0.7)
    n_epochs_bad_ch = config_params.get('n_epochs_bad_ch', 0.5)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 6)
    max_iter = config_params.get('max_iter', 4)
    tmin = config_params.get('tmin', -.2)
    tmax = config_params.get('tmax', 1.34)

    if reject is None:
        reject = {'eeg': 100e-6}

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])

    # Filter
    _bv_filter(raw, config_params, summary=summary, n_jobs=n_jobs)

    # Cut
    events = mne.find_events(raw, shortest_event=1)

    found_id = np.unique(events[:, 2])
    this_id = {k: v for k, v in _icm_lg_event_id.items() if v in found_id}

    baseline = (None, 0)
    epochs = mne.Epochs(raw, events, this_id, tmin=tmin, tmax=tmax,
                        preload=True, reject=None, picks=None,
                        baseline=baseline, verbose=False)

    epochs.drop_channels(['STI 014'])

    bad_channels, bad_epochs = _adaptive_egi(
        epochs, reject, n_epochs_bad_ch=n_epochs_bad_ch,
        n_channels_bad_epoch=n_channels_bad_epoch,
        zscore_thresh=zscore_thresh, max_iter=max_iter,
        summary=summary)
    epochs.info['bads'].extend(bad_channels)
    logger.info('found bad channels: {} {}'.format(
        len(bad_channels), str(bad_channels)))

    _check_min_channels(epochs, bad_channels, min_channels)
    _check_min_events(epochs, min_events)

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    epochs.interpolate_bads(reset_bads=True)

    # Go down to 250 since all the pipeline is prepared for that
    if epochs.info['sfreq'] != 250:
        logger.info('Resampling from {} to 250 Hz'.format(
            epochs.info['sfreq']))
        epochs.resample(250, npad='auto')
    out = epochs
    if summary is not None:
        out = out, summary
    return out


def _preprocess_lg_hdf5_gtec(raw, config_params):
    n_jobs = config_params.get('n_jobs', 1)
    reject = config_params.get('reject', None)
    min_events = config_params.get('min_events', 0.3)
    min_channels = config_params.get('min_channels', 0.7)
    n_epochs_bad_ch = config_params.get('n_epochs_bad_ch', 0.5)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 6)
    max_iter = config_params.get('max_iter', 4)
    tmin = config_params.get('tmin', -.2)
    tmax = config_params.get('tmax', 1.34)
    if reject is None:
        reject = {'eeg': 100e-6}

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])

    # Filter
    _bv_filter(raw, config_params, summary=summary, n_jobs=n_jobs)

    # Cut
    events = mne.find_events(raw, shortest_event=1)
    all_id = _icm_lg_event_id
    found_id = np.unique(events[:, 2])
    this_id = {k: v for k, v in all_id.items() if v in found_id}

    baseline = (None, 0)
    epochs = mne.Epochs(raw, events, this_id, tmin=tmin, tmax=tmax,
                        preload=True, reject=None, picks=None,
                        baseline=baseline, verbose=False)

    epochs.drop_channels(['STI 014'])

    bad_channels, bad_epochs = _adaptive_egi(
        epochs, reject, n_epochs_bad_ch=n_epochs_bad_ch,
        n_channels_bad_epoch=n_channels_bad_epoch,
        zscore_thresh=zscore_thresh, max_iter=max_iter,
        summary=summary)
    epochs.info['bads'].extend(bad_channels)
    logger.info('found bad channels: {} {}'.format(
        len(bad_channels), str(bad_channels)))

    _check_min_channels(epochs, bad_channels, min_channels)
    _check_min_events(epochs, min_events)

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    epochs.interpolate_bads(reset_bads=True)

    # Go down to 250 since all the pipeline is prepared for that
    if epochs.info['sfreq'] != 256:
        logger.info('Resampling from {} to 256 Hz'.format(
            epochs.info['sfreq']))
        epochs.resample(256, npad='auto')
    out = epochs
    if summary is not None:
        out = out, summary
    return out


def _preprocess_lg_bdf_bs(raw, config_params):
    n_jobs = config_params.get('n_jobs', 1)
    reject = config_params.get('reject', None)
    min_events = config_params.get('min_events', 0.3)
    min_channels = config_params.get('min_channels', 0.7)
    n_epochs_bad_ch = config_params.get('n_epochs_bad_ch', 0.5)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 4)
    max_iter = config_params.get('max_iter', 4)
    tmin = config_params.get('tmin', -.2)
    tmax = config_params.get('tmax', 1.34)

    if reject is None:
        reject = {'eeg': 100e-6}

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])

    # Filter
    _bs_filter(raw, config_params, summary=summary, n_jobs=n_jobs)

    # Cut
    events = mne.find_events(raw, shortest_event=1)
    all_id = _icm_lg_event_id
    found_id = np.unique(events[:, 2])
    this_id = {k: v for k, v in all_id.items() if v in found_id}

    baseline = (None, 0)
    epochs = mne.Epochs(raw, events, this_id, tmin=tmin, tmax=tmax,
                        preload=True, reject=None, picks=None,
                        baseline=baseline, verbose=False)

    epochs.drop_channels(['STI 014'])

    bad_channels, bad_epochs = _adaptive_egi(
        epochs, reject, n_epochs_bad_ch=n_epochs_bad_ch,
        n_channels_bad_epoch=n_channels_bad_epoch,
        zscore_thresh=zscore_thresh, max_iter=max_iter,
        summary=summary)
    epochs.info['bads'].extend(bad_channels)
    logger.info('found bad channels: {} {}'.format(
        len(bad_channels), str(bad_channels)))

    _check_min_channels(epochs, bad_channels, min_channels)
    _check_min_events(epochs, min_events)

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    epochs.interpolate_bads(reset_bads=True)

    # Go down to 250 since all the pipeline is prepared for that
    # epochs.resample(250, npad='auto')
    out = epochs
    if summary is not None:
        out = out, summary
    return out
