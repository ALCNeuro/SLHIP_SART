# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from mne.utils import logger

from .modules import _get_module_func


def predict(markers, summary, config='default', config_params=None):
    """ Predict one subject (markers) agains a set of markers (summary)

    Parameters
    ----------
    markers : str or nice.Markers
        Markers to use. If str, will look for an HDF5 file inside the folder.
    summary : str or nice.api.summarize.Summary
        Summary to use. If str, will look for a summary inside the folder
    config : str
        configuration to use
    config_params : dict(str) or None
        Extra dictionary of parameters to pass to the config-specific function
    Returns
    -------
    out : prediction summary

    """
    logger.info(f'Predicting from {config} config')
    out = None
    if config_params is None:
        config_params = {}
    func = _get_module_func('predict', config)
    out = func(markers, summary, config_params=config_params)
    return out
