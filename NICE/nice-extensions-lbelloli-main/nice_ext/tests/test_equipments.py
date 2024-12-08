# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import next_base as nxt
from next_base.equipments import (get_montage, get_ch_names, get_roi_ch_names,
                                  get_roi)
from nose.tools import assert_true

montages = ['ant/32', 'biosemi/128', 'bv/32', 'egi/256', 'egi/64', 'gtec/12']


def test_montages():
    """Test montage"""
    for m_name in montages:
        print('Testing {}'.format(m_name))
        montage = get_montage(m_name)
        ch_names = get_ch_names(m_name)
        scalp_roi = get_roi_ch_names(m_name, 'scalp')
        non_scalp_roi = get_roi_ch_names(m_name, 'nonscalp')
        if non_scalp_roi is not None:
            scalp_roi += non_scalp_roi
        assert_true(all(x in montage.ch_names for x in ch_names))
        assert_true(all(x in ch_names for x in scalp_roi))


def test_rois():
    """Testing if ROIs are defined"""
    for m_name in montages:
        montage = get_montage(m_name)
        for roi_name in nxt.equipments.rois._roi_names:
            get_roi(m_name, roi_name)
            ch_names = get_roi_ch_names(m_name, roi_name)
            assert_true(all(x in montage.ch_names for x in ch_names))


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
