# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np
from scipy import stats
import mne


def _flatten(alist):
    if isinstance(alist[0], list):
        alist = [item for sublist in alist for item in sublist]
    return alist


def _create_evoked(epochs, data):
    tmin = epochs.tmin
    info = epochs.info.copy()
    return mne.EvokedArray(data, info=info, tmin=tmin)


def anova_two(epochs, factor_a, factor_b):
    """Compute a two by two anova.

    Parameters
    ----------
    epochs : MNE Epochs object
        The data.
    factor_a : list (size = 2)
        The two epochs types for factor A
    factor_b : list (size = 2)
        The two epochs types for factor B

    Returns
    -------
    f_stats : dict with MNE Evoked object
        The ANOVA F-scores.
    p_values : dict with MNE Evoked object
        The ANOVA p-values.
    """

    # Check the input

    if len(factor_a) != 2 or len(factor_b) != 2:
        raise ValueError("Factors should have two conditions")

    factors_a_flat = _flatten(factor_a)
    factors_b_flat = _flatten(factor_b)

    all_types = ['{}/{}'.format(x, y) for x in factors_a_flat
                 for y in factors_b_flat]

    if not np.all(x in epochs.event_id for x in all_types):
        raise ValueError('All the conditions in both factors should be'
                         'present in the epochs file')

    # Discard unwanted epochs
    t_epochs = epochs[all_types]

    data = t_epochs.get_data()

    N = data.shape[-1]
    df_a = len(np.unique(factors_a_flat)) - 1
    df_b = len(np.unique(factors_b_flat)) - 1
    df_axb = df_a * df_b
    df_w = N - (df_a + 1) * (df_b + 1)

    grand_mean = data.mean(axis=0)
    ssq_t = np.sum((data - grand_mean) ** 2, axis=0)

    ssq_a = np.zeros_like(grand_mean)
    ssq_b = np.zeros_like(grand_mean)
    ssq_w = np.zeros_like(grand_mean)

    for ep_type in factor_a:
        f_epochs = t_epochs[ep_type]
        ssq_a += (((f_epochs.get_data().mean(axis=0) - grand_mean) ** 2) *
                  len(f_epochs))

    for ep_type in factor_b:
        f_epochs = t_epochs[ep_type]
        ssq_b += (((f_epochs.get_data().mean(axis=0) - grand_mean) ** 2) *
                  len(f_epochs))

    for ep_type_a in factor_a:
        for ep_type_b in factor_b:
            f_epochs = t_epochs['{}/{}'.format(ep_type_a, ep_type_b)]
            t_data = f_epochs.get_data()
            ssq_w += np.sum((t_data - t_data.mean(axis=0)) ** 2, axis=0)

    ssq_axb = ssq_t - ssq_a - ssq_b - ssq_w
    ms_a = ssq_a / df_a
    ms_b = ssq_b / df_b
    ms_axb = ssq_axb / df_axb
    ms_w = ssq_w / df_w
    f_a = ms_a / ms_w
    f_b = ms_b / ms_w
    f_axb = ms_axb / ms_w

    p_a = stats.f.sf(f_a, df_a, df_w)
    p_b = stats.f.sf(f_b, df_b, df_w)
    p_axb = stats.f.sf(f_axb, df_axb, df_w)

    f_stats = dict()
    f_stats['Factor A'] = _create_evoked(epochs, f_a)
    f_stats['Factor B'] = _create_evoked(epochs, f_b)
    f_stats['Interaction'] = _create_evoked(epochs, f_axb)

    p_values = dict()
    p_values['Factor A'] = _create_evoked(epochs, p_a)
    p_values['Factor B'] = _create_evoked(epochs, p_b)
    p_values['Interaction'] = _create_evoked(epochs, p_axb)

    return f_stats, p_values
