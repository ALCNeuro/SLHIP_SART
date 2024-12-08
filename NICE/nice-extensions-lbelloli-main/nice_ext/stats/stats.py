# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np

from scipy.stats import chi2

import mne


def entropy(a, axis=0):
    return -np.nansum(a * np.log(a), axis=axis) / np.log(a.shape[axis])


def roi_fun(a, roi, fun, axis=0):
    return fun(a.take(roi - 1, axis), axis)


def compute_gfp(x, alpha=0.05, method='chi2', df='auto'):
    """Compute GFP (Global Field Power) with confidence intervals
    """
    if df == 'auto':
        # XXX scaling empricially determined,
        # improves rank estimate on our data
        df = mne.rank.estimate_rank(x * 1e12, norm=False)
    else:
        df = len(x) - 1
    std = x.std(axis=0, ddof=1)

    if method == 'chi2':
        ci_lower = np.sqrt(df * std ** 2 / chi2.ppf(alpha / 2, df))
        ci_upper = np.sqrt(df * std ** 2 / chi2.ppf(1 - (alpha / 2), df))
    else:
        raise NotImplementedError('Method {} is not supported'.format(method))

    return std, ci_lower, ci_upper
