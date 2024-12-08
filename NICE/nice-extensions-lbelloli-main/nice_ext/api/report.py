# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import mne

from .modules import _get_module_func


def create_report(instance, title='default', config='default', report=None,
                  config_params=None):

    """
    Renders an HTML report of the instance processings

    Args:
        instante (TODO: get correct type preprocess_function_output): The preprocessed data to report
        title (str): Optional. The title of the report (default is 'default')
        config (str): Optional. The key of the already registered create report 
        function
        report (instance of mne.report.Report): Optional. A custom report module
        to render the HTML report 
        config_params (dict(str,any)): A dictionary of parameters for the report
        function

    Returns:

    """
    if report is None:
        report = mne.report.Report(title=title)
    if config_params is None:
        config_params = {}
    func = _get_module_func('report', config)
    return func(instance, report=report, config_params=config_params)
