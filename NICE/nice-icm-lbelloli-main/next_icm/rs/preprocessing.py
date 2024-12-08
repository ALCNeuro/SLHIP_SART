import numpy as np

import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.equipments import _egi_filter
from nice_ext.algorithms.adaptive import _adaptive_egi, find_bad_components


def register():
    register_module('preprocess', 'icm/rs/raw/egi', _preprocess_rs_raw_egi)


def _preprocess_rs_raw_egi(raw, config_params):
    from nice_ext.api.preprocessing import (_check_min_events,
                                            _check_min_channels)

    # Cut 800 ms epochs (-200, 600 ms)
    t_cut = config_params.get('t_cut', 0.8)
    # between 550ms (550 + 800 = 1350 ms space between triggers)
    min_jitter = config_params.get('min_jitter', 0.55)
    # and 850ms (850 + 800 = 1650 ms space between triggers)
    max_jitter = config_params.get('max_jitter', 0.85)
    onset = config_params.get('onset', 0)
    reject = config_params.get('reject', None)
    tmin = config_params.get('tmin', -.2)
    baseline = config_params.get('baseline', None)
    n_jobs = config_params.get('n_jobs', 1)
    min_events = config_params.get('min_events', 200)
    min_channels = config_params.get('min_channels', 0.7)
    n_epochs_bad_ch = config_params.get('n_epochs_bad_ch', 0.5)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 4)
    max_iter = config_params.get('max_iter', 4)
    run_ica = config_params.get('ica', False)

    if reject is None:
        reject = {'eeg': 100e-6}

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])

    n_eeg = len(mne.pick_types(raw.info, eeg=True))

    good_channels = ['E{}'.format(x) for x in range(1, n_eeg + 1)]
    to_drop = [x for x in raw.ch_names if x not in good_channels]
    if len(to_drop) > 0:
        raw.drop_channels(to_drop)

    # Filter
    _egi_filter(raw, config_params, summary=summary, n_jobs=n_jobs)

    max_events = int(np.ceil(len(raw) /
                     (raw.info['sfreq'] * t_cut))) + 1
    evt_times = []
    if isinstance(min_jitter, float):
        min_jitter = int(np.ceil(min_jitter * raw.info['sfreq']))
    if isinstance(max_jitter, float):
        max_jitter = int(np.ceil(max_jitter * raw.info['sfreq']))
    if isinstance(onset, float):
        onset = int(np.ceil(onset * raw.info['sfreq']))
    jitters = np.random.random_integers(
        min_jitter, max_jitter, max_events)
    epoch_len = int(np.ceil(t_cut * raw.info['sfreq']))
    this_sample = onset
    this_jitter = 0
    while this_sample < len(raw):
        evt_times.append(this_sample)
        this_sample += epoch_len + jitters[this_jitter]
        this_jitter += 1
    evt_times = np.array(evt_times)
    events = np.concatenate((evt_times[:, None],
                             np.zeros((len(evt_times), 1), dtype=np.int),
                             np.ones((len(evt_times), 1), dtype=np.int)),
                            axis=1)
    event_id = 1

    # Cut
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=t_cut + tmin,
                        preload=True, reject=None, picks=None,
                        baseline=baseline, verbose=False)

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

    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    if run_ica:
        # Run ICA
        logger.warning('ICA preprocessing is outdated and experimental')
        ica = mne.preprocessing.ICA(
            n_components=0.99, max_pca_components=None, max_iter=512)
        ica.fit(epochs, picks=picks, decim=1, reject=reject)

        bad_comps = find_bad_components(
            ica, epochs, zscore_thresh=4, max_iter=4)

        ica.exclude += list(bad_comps)

        if len(ica.exclude) > 0:
            ica.apply(epochs)

    epochs.interpolate_bads(reset_bads=True)
    out = epochs
    if summary is not None:
        out = out, summary
    return out
