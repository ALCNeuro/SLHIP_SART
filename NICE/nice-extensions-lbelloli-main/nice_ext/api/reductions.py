# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017
from typing import List

import numpy as np
from scipy.stats import trim_mean

from mne.utils import logger

from .modules import _get_module_func, register_module


def trim_mean90(a, axis=0):
    return trim_mean(a, proportiontocut=.05, axis=axis)


def trim_mean80(a, axis=0):
    return trim_mean(a, proportiontocut=.1, axis=axis)


_stats_functions = {
    'trim_mean80': {
        'description': 'Trim Mean with 80% of the center data',
        'func': trim_mean80
    },
    'trim_mean90': {
        'description': 'Trim Mean with 90% of the center data',
        'func': trim_mean90
    },
    'std': {
        'description': 'Standard Deviation',
        'func': np.std
    },
    'mean': {
        'description': 'Mean',
        'func': np.mean
    }
}


def get_avaialable_functions():
    return list(_stats_functions.keys())


def get_function_by_name(funname):
    if funname not in _stats_functions:
        raise ValueError(f'Function {funname} does not exist in stats')
    return _stats_functions[funname]['func']

def get_function_description(funname):
    if funname not in _stats_functions:
        raise ValueError(f'Function {funname} does not exist in stats')
    return _stats_functions[funname]['description']

def get_reductions(config='default', config_params=None):
    logger.info(f'Using reductions from {config} config')
    configs = config.split('/')
    config_fun = configs[-1]
    func = _get_module_func('reductions', config)
    out = func(config_fun, config_params=config_params)
    return out


# Decorator to register a io module
def next_reduction_module(module_name: str, estimators: List[str], module_description: str=''):

    estimators = {e: get_function_description(e) for e in estimators}

    def wrapper(module):
        module.__description__ = module_description
        module.__estimators__ = estimators
        for estimator in estimators.keys():
            register_module('reductions', f'{module_name}/{estimator}', module)

    return wrapper
