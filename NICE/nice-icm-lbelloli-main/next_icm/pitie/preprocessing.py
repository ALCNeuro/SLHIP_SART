import numpy as np
import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.algorithms.adaptive import _adaptive_pitie
from nice_ext.equipments import _egi_filter



def register():
    register_module('preprocess', 'pitie/rs/raw/edf', _preprocess_rs_raw_edf)


def _preprocess_rs_raw_edf(raw, config_params):
    from nice.api.preprocessing import _check_min_events

    t_cut = config_params.get('t_cut', 0.8)
    min_jitter = config_params.get('min_jitter', 0)
    max_jitter = config_params.get('max_jitter', min_jitter)
    onset = config_params.get('onset', 0)
    reject = config_params.get('reject', None)
    tmin = config_params.get('tmin', -.2)
    baseline = config_params.get('baseline', None)
    n_jobs = config_params.get('n_jobs', 1)
    min_events = config_params.get('min_events', 200)
    n_channels_bad_epoch = config_params.get('n_channels_bad_epoch', 0.1)
    zscore_thresh = config_params.get('zscore_thresh', 4)
    max_iter = config_params.get('max_iter', 4)

    if reject is None:
        reject = {'eeg': 150e-6}

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])

    max_events = int(np.ceil(len(raw) /
                     (raw.info['sfreq'] * t_cut))) + 1
    evt_times = []
    if isinstance(min_jitter, float):
                min_jitter = int(np.ceil(
                    min_jitter * raw.info['sfreq']))
    if isinstance(max_jitter, float):
                max_jitter = int(np.ceil(
                    max_jitter * raw.info['sfreq']))
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

    # Filter
    _egi_filter(raw, config_params, summary=summary, n_jobs=n_jobs)

    # Cut
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=t_cut - tmin,
                        preload=True, reject=None, picks=None,
                        baseline=baseline, verbose=False)

    bad_epochs = _adaptive_pitie(
        epochs, reject, n_channels_bad_epoch=n_channels_bad_epoch,
        zscore_thresh=zscore_thresh, max_iter=max_iter,
        summary=summary)
    _check_min_events(epochs, min_events)

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()
    out = epochs
    if summary is not None:
        out = out, summary
    return out
