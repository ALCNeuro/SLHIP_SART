# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from copy import deepcopy
import mne
from mne.utils import logger


def _egi_filter(raw, params=None, summary=None, n_jobs=1):
    if params is None:
        params = {}
    lpass = params.get('lpass', 45.)
    hpass = params.get('hpass', 0.5)
    picks = mne.pick_types(raw.info, eeg=True, meg=True, ecg=True, exclude=[])
    _filter_params = dict(method='iir',
                          l_trans_bandwidth=0.1,
                          iir_params=dict(ftype='butter', order=6))
    filter_params = [
        dict(l_freq=hpass, h_freq=None,
             iir_params=dict(ftype='butter', order=6)),
        dict(l_freq=None, h_freq=lpass,
             iir_params=dict(ftype='butter', order=8))
    ]

    for fp in filter_params:
        if fp['l_freq'] is None and fp['h_freq'] is None:
            continue
        _filter_params2 = deepcopy(_filter_params)
        if fp.get('method') == 'fft':
            _filter_params2.pop('iir_params')
        if isinstance(fp, dict):
            _filter_params2.update(fp)
        if summary is not None:
            summary['steps'].append(
                {'step': 'filter', 'params':
                    {'hpass': '{} Hz'.format(_filter_params2['l_freq']),
                     'lpass': '{} Hz'.format(_filter_params2['h_freq'])}})
        raw.filter(picks=picks, n_jobs=n_jobs, **_filter_params2)

    notches = params.get('notches', [50, 100])
    logger.info('Notch filters at {}'.format(notches))
    if summary is not None:
        params = {(k + 1): '{} Hz'.format(v) for k, v in enumerate(notches)}
        summary['steps'].append({'step': 'notches', 'params': params})
    raw.notch_filter(notches, method='fft', n_jobs=n_jobs)

    if raw.info['sfreq'] != 250:
        logger.info('Resampling to 250Hz')
        raw.resample(250)
    if summary is not None:
        summary['steps'].append({'step': 'resample',
                                 'params': {'sfreq': 250}})


def _eximia_filter(raw, params=None, summary=None, n_jobs=1):
    if params is None:
        params = {}
    lpass = params.get('lpass', 45.)
    hpass = params.get('hpass', 0.5)
    picks = mne.pick_types(raw.info, eeg=True, meg=True, ecg=True, exclude=[])
    _filter_params = dict(method='iir',
                          l_trans_bandwidth=0.1,
                          iir_params=dict(ftype='butter', order=4))
    filter_params = [
        dict(l_freq=hpass, h_freq=None,
             iir_params=dict(ftype='butter', order=4)),
        dict(l_freq=None, h_freq=lpass,
             iir_params=dict(ftype='butter', order=8))
    ]

    for fp in filter_params:
        _filter_params2 = deepcopy(_filter_params)
        if fp.get('method') == 'fft':
            _filter_params2.pop('iir_params')
        if isinstance(fp, dict):
            _filter_params2.update(fp)
        if summary is not None:
            summary['steps'].append(
                {'step': 'filter', 'params':
                    {'hpass': '{} Hz'.format(_filter_params2['l_freq']),
                     'lpass': '{} Hz'.format(_filter_params2['h_freq'])}})
        raw.filter(picks=picks, n_jobs=n_jobs, **_filter_params2)

    notches = [50, 100, 200, 400]
    logger.info('Notch filters at {}'.format(notches))
    if summary is not None:
        params = {(k + 1): '{} Hz'.format(v) for k, v in enumerate(notches)}
        summary['steps'].append({'step': 'notches', 'params': params})
    raw.notch_filter(notches, method='fft', n_jobs=n_jobs)


def _bv_filter(raw, params=None, summary=None, n_jobs=1):
    if params is None:
        params = {}
    lpass = params.get('lpass', 45.)
    hpass = params.get('hpass', 0.5)
    picks = mne.pick_types(raw.info, eeg=True, meg=True, ecg=True, exclude=[])
    _filter_params = dict(method='iir',
                          l_trans_bandwidth=0.1,
                          iir_params=dict(ftype='butter', order=4))
    filter_params = [
        dict(l_freq=hpass, h_freq=None,
             iir_params=dict(ftype='butter', order=4)),
        dict(l_freq=None, h_freq=lpass,
             iir_params=dict(ftype='butter', order=8))
    ]

    for fp in filter_params:
        _filter_params2 = deepcopy(_filter_params)
        if fp.get('method') == 'fft':
            _filter_params2.pop('iir_params')
        if isinstance(fp, dict):
            _filter_params2.update(fp)
        if summary is not None:
            summary['steps'].append(
                {'step': 'filter', 'params':
                    {'hpass': '{} Hz'.format(_filter_params2['l_freq']),
                     'lpass': '{} Hz'.format(_filter_params2['h_freq'])}})
        raw.filter(picks=picks, n_jobs=n_jobs, **_filter_params2)

    notches = [50, 100]
    if raw.info['sfreq'] > 400:
        notches.append(200)
    logger.info('Notch filters at {}'.format(notches))
    if summary is not None:
        params = {(k + 1): '{} Hz'.format(v) for k, v in enumerate(notches)}
        summary['steps'].append({'step': 'notches', 'params': params})
    raw.notch_filter(notches, method='fft', n_jobs=n_jobs)


def _bs_filter(raw, params=None, summary=None, n_jobs=1):
    if params is None:
        params = {}
    lpass = params.get('lpass', 45.)
    hpass = params.get('hpass', 0.5)
    picks = mne.pick_types(raw.info, eeg=True, meg=True, ecg=True, exclude=[])
    _filter_params = dict(method='iir',
                          l_trans_bandwidth=0.1,
                          iir_params=dict(ftype='butter', order=4))
    filter_params = [
        dict(l_freq=hpass, h_freq=None,
             iir_params=dict(ftype='butter', order=4)),
        dict(l_freq=None, h_freq=lpass,
             iir_params=dict(ftype='butter', order=8))
    ]

    for fp in filter_params:
        _filter_params2 = deepcopy(_filter_params)
        if fp.get('method') == 'fft':
            _filter_params2.pop('iir_params')
        if isinstance(fp, dict):
            _filter_params2.update(fp)
        if summary is not None:
            summary['steps'].append(
                {'step': 'filter', 'params':
                    {'hpass': '{} Hz'.format(_filter_params2['l_freq']),
                     'lpass': '{} Hz'.format(_filter_params2['h_freq'])}})
        raw.filter(picks=picks, n_jobs=n_jobs, **_filter_params2)

    notches = [50, 100, 200, 400]
    logger.info('Notch filters at {}'.format(notches))
    if summary is not None:
        params = {(k + 1): '{} Hz'.format(v) for k, v in enumerate(notches)}
        summary['steps'].append({'step': 'notches', 'params': params})
    raw.notch_filter(notches, method='fft', n_jobs=n_jobs)
