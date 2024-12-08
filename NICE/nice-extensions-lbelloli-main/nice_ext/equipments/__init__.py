from . filters import (_egi_filter, _bv_filter, _bs_filter, _eximia_filter)
from . rois import get_roi, get_roi_ch_names, define_rois
from . montages import (define_equipment, get_montage, get_ch_names,
                        prepare_layout, map_montage,
                        get_ch_adjacency_montage, get_ch_adjacency)

from . import ant
from . import biosemi
from . import brainvision
from . import egi
from . import gtec
from . import eximia

ant.register()
biosemi.register()
brainvision.register()
egi.register()
gtec.register()
eximia.register()