from nice.markers import (ContingentNegativeVariation,
                          TimeLockedTopography,
                          TimeLockedContrast,
                          WindowDecoding)

from nice_ext.api.modules import register_module

from ..rs.markers import _get_resting_state


def register():
    register_module('markers', 'icm/lg', _get_local_global)


def _get_local_global(config_params):
    f_list = [
        # Evokeds
        ContingentNegativeVariation(tmin=-0.004, tmax=0.596),

        # need conditions
        TimeLockedTopography(tmin=0.064, tmax=0.112, comment='p1'),
        TimeLockedTopography(tmin=0.876, tmax=0.936, comment='p3a'),
        TimeLockedTopography(tmin=0.996, tmax=1.196, comment='p3b'),

        TimeLockedContrast(tmin=None, tmax=None, condition_a='LSGS',
                           condition_b='LDGD', comment='LSGS-LDGD'),

        TimeLockedContrast(tmin=None, tmax=None, condition_a='LSGD',
                           condition_b='LDGS', comment='LSGD-LDGS'),

        TimeLockedContrast(tmin=None, tmax=None, condition_a=['LDGS', 'LDGD'],
                           condition_b=['LSGS', 'LSGD'], comment='LD-LS'),

        TimeLockedContrast(tmin=0.736, tmax=0.788, condition_a=['LDGS', 'LDGD'],
                           condition_b=['LSGS', 'LSGD'], comment='mmn'),

        TimeLockedContrast(tmin=0.876, tmax=0.936, condition_a=['LDGS', 'LDGD'],
                           condition_b=['LSGS', 'LSGD'], comment='p3a'),

        TimeLockedContrast(tmin=None, tmax=None, condition_a=['LSGD', 'LDGD'],
                           condition_b=['LSGS', 'LDGS'], comment='GD-GS'),

        TimeLockedContrast(tmin=0.996, tmax=1.196, condition_a=['LSGD', 'LDGD'],
                           condition_b=['LSGS', 'LDGS'], comment='p3b'),

        WindowDecoding(tmin=0.6, tmax=0.967, condition_a=['LDGS', 'LDGD'],
                       condition_b=['LSGS', 'LSGD'], comment='local'),

        WindowDecoding(tmin=0.967, tmax=1.336, condition_a=['LSGD', 'LDGD'],
                       condition_b=['LSGS', 'LDGS'], comment='global'),
    ]
    fc = _get_resting_state(config_params)
    fc.add_marker(f_list)
    return fc
