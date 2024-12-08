import numpy as np

from nice_ext.api.reductions import (
    get_function_by_name,
    get_avaialable_functions,
    next_reduction_module
)
from nice_ext.api.modules import register_module
from nice_ext.equipments import get_roi

from ..rs.reductions import _get_rs_montage_generic


def register():
    for f in get_avaialable_functions():
        register_module('reductions', 'icm/lg/egi128/{}'.format(f),
                        _get_lg_egi128)
        register_module('reductions', 'icm/lg/egi128gfp/{}'.format(f),
                        _get_lg_egi128gfp)

@next_reduction_module(
    'icm/lg/egi256',
    get_avaialable_functions(),
    module_description='ICM LG reduction for EGI 256 channels - $estimator'
)
def _get_lg_egi256(config, config_params):
    if len(config) == 0:
        config == 'mean'
    epochs_fun = get_function_by_name(config)
    channels_fun = np.mean
    return _get_lg_egi256_generic(epochs_fun, channels_fun)


@next_reduction_module(
    'icm/lg/egi256gfp',
    get_avaialable_functions(),
    module_description='ICM LG reduction for EGI 256 channels - $estimator (GFP)'
)
def _get_lg_egi256gfp(config, config_params):
    if len(config) == 0:
        config == 'mean'
    epochs_fun = get_function_by_name(config)
    channels_fun = np.std
    return _get_lg_egi256_generic(epochs_fun, channels_fun)


def _get_lg_egi128(config, config_params):
    if len(config) == 0:
        config == 'mean'
    epochs_fun = get_function_by_name(config)
    channels_fun = np.mean
    return _get_lg_egi128_generic(epochs_fun, channels_fun)


def _get_lg_egi128gfp(config, config_params):
    if len(config) == 0:
        config == 'mean'
    epochs_fun = get_function_by_name(config)
    channels_fun = np.std
    return _get_lg_egi128_generic(epochs_fun, channels_fun)


def _get_lg_egi256_generic(epochs_fun, channels_fun):
    return _get_lg_montage_generic(
        montage='egi/256', epochs_fun=epochs_fun, channels_fun=channels_fun)


def _get_lg_egi128_generic(epochs_fun, channels_fun):
    return _get_lg_montage_generic(
        montage='egi/128', epochs_fun=epochs_fun, channels_fun=channels_fun)


def _get_lg_montage_generic(montage, epochs_fun, channels_fun):
    reduction_params = _get_rs_montage_generic(
        montage, epochs_fun, channels_fun)

    scalp_roi = get_roi(config=montage, roi_name='scalp')
    cnv_roi = get_roi(config=montage, roi_name='cnv')
    mmn_roi = get_roi(config=montage, roi_name='mmn')
    p3b_roi = get_roi(config=montage, roi_name='p3b')
    p3a_roi = get_roi(config=montage, roi_name='p3a')

    reduction_params['ContingentNegativeVariation'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': None,
            'channels': cnv_roi}}

    reduction_params['TimeLockedTopography'] = {
        'reduction_func':
            [{'axis': 'times', 'function': np.mean},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi,
            'times': None}}

    reduction_params['TimeLockedContrast'] = {
        'reduction_func':
            [{'axis': 'times', 'function': np.mean},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi,
            'times': None}}

    reduction_params['TimeLockedContrast/mmn'] = {
        'reduction_func':
            [{'axis': 'times', 'function': np.mean},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': None,
            'channels': mmn_roi,
            'times': None}}

    reduction_params['TimeLockedContrast/p3b'] = {
        'reduction_func':
            [{'axis': 'times', 'function': np.mean},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': None,
            'channels': p3b_roi,
            'times': None}}

    reduction_params['TimeLockedContrast/p3a'] = {
        'reduction_func':
            [{'axis': 'times', 'function': np.mean},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': None,
            'channels': p3a_roi,
            'times': None}}

    reduction_params['WindowDecoding'] = {
        'reduction_func':
            [{'axis': 'folds', 'function': np.mean}],
        'picks': {'folds': None}}

    return reduction_params
