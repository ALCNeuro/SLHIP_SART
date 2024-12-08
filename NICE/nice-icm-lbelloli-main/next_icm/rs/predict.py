from nice_ext.api.modules import register_module
from nice_ext.api.summarize import _try_read_summary
from nice_ext.common import _predict


def register():
    register_module('predict', 'icm/rs', _predict_rs)


def _predict_rs(markers, summary, config_params=None):
    if config_params is None:
        config_params = {}

    if 'ml_classes' not in config_params:
        config_params['ml_classes'] = ['VS/UWS', 'MCS']

    if 'target' not in config_params:
        config_params['target'] = 'Diagnosis'

    if 'reductions' not in config_params:
        config_params['reductions'] = [
            'icm/rs/egi256/trim_mean80', 'icm/rs/egi256/std',
            'icm/rs/egi256gfp/trim_mean80', 'icm/rs/egi256gfp/std']

    if 'clf_type' not in config_params:
        config_params['clf_type'] = ['gssvm', 'et-reduced']
        config_params['clf_select'] = {'gssvm': 20., 'et-reduced': None}
    elif 'clf_select' not in config_params:
        config_params['clf_select'] = {'gssvm': 20., 'et-reduced': None}

    if isinstance(summary, str):
        summ = _try_read_summary(summary)
        if summ is False:
            raise ValueError('Can not get a summary from {}'.format(summary))
        summary = summ

    summary._scalars = summary.scalars().replace(
        {config_params['target']: {'MCS+': 'MCS', 'MCS-': 'MCS'}})
    return _predict(markers, summary, config_params)
