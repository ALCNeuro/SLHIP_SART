# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from mne.utils import logger

from .modules import _get_module_func


def fit(instance, config='default', config_params=None):
    logger.info('Processing markers from {} config'.format(config))
    out = None
    if config_params is None:
        config_params = {}
    func = _get_module_func('markers', config)
    out = func(config_params=config_params)
    out.fit(instance)
    return out
