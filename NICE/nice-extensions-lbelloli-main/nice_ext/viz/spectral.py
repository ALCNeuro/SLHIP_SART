# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017


import mne
import numpy as np
from ..equipments import get_roi_ch_names, get_ch_names


def plot_psd_spectrum(marker, rois=None, ax=None, sns_kwargs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_style('white')
    sns.set_color_codes()

    psd = marker.estimator

    if rois is None:
        rois = []

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    labels = None
    if len(rois) == 0:
        data = psd.data_.mean(0).T
    else:
        labels = []
        rois_data = []
        for t_roi in rois:
            t_names = get_roi_ch_names(marker.ch_info_['description'], t_roi)
            t_idx = mne.pick_channels(marker.ch_info_['ch_names'],
                                      include=t_names,
                                      exclude=marker.ch_info_['bads'])
            t_data = psd.data_.mean(axis=0)[t_idx].mean(axis=0)
            rois_data.append(t_data)
            labels.append(t_roi)
        data = np.array(rois_data).T

    freqs = psd.freqs_
    lines = ax.plot(freqs, data, lw=0.7)
    ax.legend(lines, labels)
    ax.set_xlabel('Frequency')
    ax.set_ylabel(r'Power Spectral Density ($V^2/Hz$)')

    fmax = np.max(freqs)
    fmin = np.min(freqs)
    vlines = [x for x in [4, 8, 12, 30] if x < fmax and x > fmin]
    [ax.axvline(x, ls='--', lw=0.7, c='0.7') for x in vlines]
    fig.suptitle('Spectral Density plot')
    return fig


def plot_ch_freq_matrix(marker, ax=None, sns_kwargs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_style('white')
    sns.set_color_codes()

    psd = marker.estimator

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data = 10 * np.log10(psd.data_.mean(0))

    ch_names = get_ch_names(marker.ch_info_['description'])
    good_idx = mne.pick_channels(marker.ch_info_['ch_names'], include=[],
                                 exclude=marker.ch_info_['bads'])

    bad_idx = mne.pick_channels(marker.ch_info_['ch_names'],
                                include=marker.ch_info_['bads'])

    freqs = psd.freqs_

    all_data = np.zeros((len(ch_names), len(freqs)))
    all_data[good_idx, :] = data
    all_data[bad_idx, :] = np.nan

    extent = [0, np.max(freqs), 0, all_data.shape[0]]
    im = ax.imshow(all_data, cmap='viridis', aspect='auto', extent=extent)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(r'$10 * log_{10} ($V^2/Hz$)$')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Channel')

    fig.suptitle('Spectral Power Density matrix')
    return fig
