# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from mne.epochs import BaseEpochs, read_epochs
from mne.utils import logger
from . modules import register_module


def register():
    register_module('io', 'dummy', _dummy_io)
    register_module('preprocess', 'bypass', _preprocess_bypass)
    register_module('preprocess', 'skip', _skip_preprocess)
    register_module('markers', 'dummy', _dummy_markers)
    register_module('report', 'skip', _skip_report)


def _dummy_io(path, config_params):
    return 'Dummy Placeholder'


def _preprocess_bypass(instance, config_params):
    if not isinstance(instance, BaseEpochs):
        msg = 'Default preprocessing is only defined for epochs'
        logger.error(msg)
        raise ValueError(msg)
    out = instance
    if 'summary' in config_params and config_params['summary'] is True:
        summary = {}
        out = instance, summary
    return out


def _skip_preprocess(instance, config_params):
    out = instance
    if 'summary' in config_params and config_params['summary'] is True:
        summary = {}
        out = instance, summary
    return out


class DummyMarkers:
    def fit(self, instance):
        return 'Dummy Placeholder'


def _dummy_markers(config_params):
    return DummyMarkers()


def _skip_report(instance, report, config_params):
    return None
