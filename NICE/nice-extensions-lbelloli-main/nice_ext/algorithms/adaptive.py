# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import math
import numpy as np

from scipy.signal import butter, filtfilt

import mne
from mne.utils import logger


def find_bads_channels_threshold(epochs, picks, reject, n_epochs_bad_ch=0.5):
    """Find bad channels based on the number of epochs where the values range 
    is over the reject threshold.

    The range of the values is defined as the difference between the maximum value
    and the minimum value of a single channel in a single epoch. This defines the 
    channel range per epoch. After the range is calculated for each epoch, the
    number of epochs where the channel range is over the reject threshold is
    calculated. If the number of epochs is over the n_epochs_bad_ch, then
    the channel is returned as bad.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object.
    picks : list of int
        The indices of the channels to be used.
    reject : dict
        The rejection threshold for each channel type (EEG, MEG, etc).
    n_epochs_bad_ch : int or float
        The minimum number of epochs over the threshold to consider a channel to
        be bad. If float, it is the fraction of epochs. Default is 0.5.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.
    
    """

    n_channels = len(picks)
    data = epochs.get_data()
    n_epochs = data.shape[0]

    if isinstance(n_epochs_bad_ch, (np.float, float)):
        n_epochs_bad_ch = math.floor(n_epochs_bad_ch * n_epochs)

    ch_types_inds = mne.io.pick.channel_indices_by_type(epochs.info)
    data = np.transpose(data, (1, 0, 2))
    bad_ch_idx = np.ndarray((0,), dtype=np.int)
    for key, reject_thresh in reject.items():
        idx = np.array([x for x in ch_types_inds[key] if x in picks])
        count_bad_epochs = np.zeros((n_channels), dtype=np.int)
        for i_ch, channel in enumerate(data[idx]):
            deltas = channel.max(axis=1) - channel.min(axis=1)
            idx_deltas = np.where(np.greater(deltas, reject_thresh))[0]
            count_bad_epochs[i_ch] = idx_deltas.shape[0]
        reject_bad_channels = np.where(count_bad_epochs > n_epochs_bad_ch)[0]
        logger.info('Reject by threshold %f on %s %d : bad_channels: %s' %
                    (reject_thresh, key.upper(), len(reject_bad_channels),
                     reject_bad_channels))
        bad_ch_idx = np.concatenate((bad_ch_idx, reject_bad_channels))

    bad_chs = list({epochs.ch_names[i] for i in bad_ch_idx})
    return bad_chs


def find_bads_channels_variance(inst, picks, zscore_thresh=4, max_iter=2):
    """Find bad channels based on iterated Z-scoring outliers of the channel variances.

    First, the channel variances are calculated. using numpy.var along the sample axis.
    Then, the channel variances are passed to mne.preprocessing.bads._find_outliers 
    to find bad channels based on iterated Z-scoring over the calculated variance.
    This procedure compares the absolute z-score of the variances against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.

    Parameters
    ----------
    inst : instance of mne.Epochs
        The data.
    picks : list of int
        The indices of the channels to be used.
    zscore_thresh : int
        The threshold for the z-score outliers. Default is 4 std.
    max_iter : int
        The maximum number of iterations of the iterative z-scoring. Default is 2.
    
    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.

    """


    logger.info('Looking for bad channels with variance')
    if isinstance(inst, mne.Epochs):
        data = inst.get_data()
    else:
        data = inst._data[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.NaN)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_var = np.ma.hstack(masked_data).var(axis=-1)
    bad_ch_var = mne.preprocessing.bads._find_outliers(
        ch_var, threshold=zscore_thresh, max_iter=max_iter)
    logger.info('Reject by variance: bad_channels: %s' % bad_ch_var)
    bad_chs = list({inst.ch_names[i] for i in bad_ch_var})
    return bad_chs


def find_bads_channels_high_frequency(inst, picks, zscore_thresh=4, max_iter=2):
    """Find bad channels based on iterated Z-scoring outliers of the channel high frequencies standard deviation.

    First, the channel high frequencies standard deviations are calculated.
    Then, the channel high frequencies standard deviations are passed to 
    mne.preprocessing.bads._find_outliers to find bad channels based on iterated
    Z-scoring over the calculated standard deviation of the channels high frequencies.
    This procedure compares the absolute z-score of the standard deviations against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.

    Parameters
    ----------
    inst : instance of mne.Epochs
        The data.
    picks : list of int
        The indices of the channels to be used.
    zscore_thresh : int
        The threshold for the z-score outliers. Default is 4 std.
    max_iter : int
        The maximum number of iterations of the iterative z-scoring. Default is 2.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.
    """
    logger.info('Looking for bad channels with high frequency variance')
    if isinstance(inst, mne.Epochs):
        data = inst.get_data()
    else:
        data = inst._data[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.NaN)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    filter_freq = 25
    b, a = butter(4, 2.0 * filter_freq / inst.info['sfreq'], 'highpass')
    filt_data = filtfilt(b, a, np.ma.hstack(masked_data))
    filt_masked_data = np.ma.masked_array(filt_data, fill_value=np.NaN)
    if len(exclude) > 0:
        filt_masked_data[exclude, :] = np.ma.masked
    bad_ch_hf = mne.preprocessing.bads._find_outliers(
        filt_masked_data.std(axis=-1), threshold=zscore_thresh,
        max_iter=max_iter)
    logger.info('Reject by high frequency std: bad_channels: %s' % bad_ch_hf)
    bad_chs = list({inst.ch_names[i] for i in bad_ch_hf})
    return bad_chs


def find_bads_epochs_threshold(epochs, picks, reject, n_channels_bad_epoch=0.1):
    """Find bad epochs based on the number of channels where the values range 
    is over the reject threshold.

    The range of the values is defined as the difference between the maximum value
    and the minimum value of a single channel in a single epoch. This defines the 
    channel range per epoch. After the range is calculated for each epoch, the
    number of channels where the range is over the reject threshold is
    calculated for each epoch. If the number of channels is over the 
    n_channels_bad_epoch for an epoch, then that epoch is returned as bad.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object.
    picks : list of int
        The indices of the channels to be used.
    reject : dict
        The rejection threshold for each channel type (EEG, MEG, etc).
    n_channels_bad_epoch : int or float
        The number of channels that have to be over the reject threshold for an
        epoch to be considered bad. Default is 0.1.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.
    
    """

    n_channels = len(picks)
    bad_ep_idx = np.ndarray((0,), dtype=np.int)
    if isinstance(n_channels_bad_epoch, (np.float, float)):
        n_channels_bad_epoch = math.floor(n_channels_bad_epoch * n_channels)

    data = epochs.get_data()
    masked_data = np.ma.masked_array(data, fill_value=np.NaN)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_types_inds = mne.io.pick.channel_indices_by_type(epochs.info)
    n_epochs = masked_data.shape[0]
    for key, reject_thresh in reject.items():
        idx = np.array([x for x in ch_types_inds[key] if x in picks])
        count_bad_chans = np.zeros((n_epochs), dtype=np.int)
        for i_ep, epoch in enumerate(masked_data[:, idx]):
            deltas = epoch.max(axis=1) - epoch.min(axis=1)
            idx_deltas = np.where(np.greater(deltas, reject_thresh))[0]
            count_bad_chans[i_ep] = idx_deltas.shape[0]
        reject_bad_epochs = np.where(count_bad_chans > n_channels_bad_epoch)[0]
        logger.info('Reject by threshold %f on %s : bad_epochs: %s' %
                    (reject_thresh, key.upper(), reject_bad_epochs))
    bad_epochs = np.unique(np.concatenate((bad_ep_idx, reject_bad_epochs)))

    return bad_epochs


def find_bad_components(ica, epochs, zscore_thresh=4, max_iter=2):
    variance = np.hstack(ica.get_sources(epochs).get_data()).var(1)
    var_inds = mne.preprocessing.bads._find_outliers(
        variance, threshold=zscore_thresh, max_iter=max_iter)
    return var_inds


def _adaptive_egi(epochs, reject, n_epochs_bad_ch=0.5,
                  n_channels_bad_epoch=0.1,
                  zscore_thresh=4, max_iter=4, summary=None):

    """Find bad channels and bad epochs based on 4 adaptative steps.

    The steps are:
    1. Find bad channels based on the number of epochs where the channel values range (max - min) is over the reject threshold (see: find_bads_channels_threshold).
    2. Find bad channels based on iterative z-score outlier detection over the channel variance (see: find_bads_channels_variance).
    3. Find bad epochs based on the number of channels where the values range is over the reject threshold (see: find_bads_epochs_threshold).
    4. Find bad channels based on iterative z-score outlier detection over the channel high frequency variance (see: find_bads_channels_high_freq).

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object.
    reject : dict
        The rejection threshold for each channel type (EEG, MEG, etc).
    n_epochs_bad_ch : int or float
        The number of epochs that have to be over the reject threshold for a channel to be considered bad. Default is 0.5.
    n_channels_bad_epoch : int or float
        The number of channels that have to be over the reject threshold for an epoch to be considered bad. Default is 0.1.
    zscore_thresh : int or float
        The z-score threshold for the z-scoring outlier detection. Default is 4.
    max_iter : int
        The maximum number of iterations for the z-scoring outlier detection. Default is 4.
    summary : dict
        The summary dictionary.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.
    bad_epochs : list of int
        The indices of the bad epochs.
    
    """

    if isinstance(reject, float):
        reject = {'eeg': reject}

    bad_channels = set()
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude='bads')

    # 1. Adaptive - threshold (Channels)
    method_params = {'reject': reject, 'n_epochs_bad_ch': n_epochs_bad_ch}
    bad_chs = find_bads_channels_threshold(epochs, picks, **method_params)
    bad_channels.update(bad_chs)

    if summary is not None:
        summary['steps'].append({
            'step': 'adaptive/threshold',
            'params':method_params,
            'bad_chs':bad_chs
        })
    

    # 2. Adaptive - variance (Channels)
    picks = mne.pick_channels(
        epochs.info['ch_names'], include=[], exclude=list(bad_channels)
    )
    method_params = {'zscore_thresh': zscore_thresh, 'max_iter': max_iter}
    bad_chs = find_bads_channels_variance(epochs, picks, **method_params)
    bad_channels.update(bad_chs)
    
    if summary is not None:
        summary['steps'].append({
            'step':'adaptive/variance',
            'params':method_params,
            'bad_chs':bad_chs
        })
    
    # 3. Adaptive - Threshold (Epochs)
    picks = mne.pick_channels(
        epochs.info['ch_names'], include=[], exclude=list(bad_channels)
    )
    method_params = {
        'reject': reject,
        'n_channels_bad_epoch': n_channels_bad_epoch
    }
    bad_epochs = find_bads_epochs_threshold(epochs, picks, **method_params)
    epochs.drop(bad_epochs, reason='artifacted')

    if summary is not None:
        summary['steps'].append({
            'step':'adaptive/threshold',
            'params':method_params,
            'bad_epochs':bad_epochs
        })

    logger.info(f'found bad epochs: {len(bad_epochs)} {str(bad_epochs)}')

    # 4. Adaptive - High frequency (Channels)
    picks = mne.pick_channels(
        epochs.info['ch_names'], include=[], exclude=list(bad_channels))
    method_params = {'zscore_thresh': zscore_thresh, 'max_iter': max_iter}
    bad_chs = find_bads_channels_high_frequency(epochs, picks, **method_params)
    bad_channels.update(bad_chs)

    if summary is not None:
        summary['steps'].append({
            'step': 'adaptive/highfreq',
            'params': method_params,
            'bad_chs': bad_chs
        })

    if summary is not None:
        method_params = {
            'zscore_thresh': zscore_thresh,
            'max_iter': max_iter,
            'reject': reject,
            'n_epochs_bad_ch': n_epochs_bad_ch,
            'n_channels_bad_epoch': n_channels_bad_epoch
        }
        summary['steps'].append({
            'step': 'adative (total)',
            'params': method_params,
            'bad_chs': bad_channels,
            'bad_epochs': bad_epochs
        })

    bad_channels = sorted(bad_channels)
    if summary is not None:
        summary['bad_channels'] = bad_channels
        summary['bad_epochs'] = bad_epochs
    return bad_channels, bad_epochs


def _adaptive_pitie(epochs, reject, n_channels_bad_epoch=0.2, zscore_thresh=4,
                    max_iter=4, summary=None):
    picks = mne.pick_types(epochs.info, meg=False, eeg=True)
    method = 'adaptive/threshold'
    method_params = {'reject': reject,
                     'n_channels_bad_epoch': n_channels_bad_epoch}
    bad_epochs = find_bads_epochs_threshold(epochs, picks, **method_params)
    if summary is not None:
        summary['steps'].append(dict(step=method, params=method_params,
                                     bad_epochs=bad_epochs))

    epochs.drop(bad_epochs, reason='artifacted')
    logger.info('found bad epochs: {} {}'.format(
        len(bad_epochs), str(bad_epochs)))
    return bad_epochs
