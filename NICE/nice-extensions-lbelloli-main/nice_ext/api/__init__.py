# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from . io import read
from . preprocessing import preprocess
from . markers import fit
from . reductions import get_reductions
from . report import create_report
from . summarize import summarize_subject, summarize_run, read_summary
from . predict import predict
from . import modules
from . default import register as _register_default
from . import utils
from . import pipelines

_register_default()

def _get_installed_distributions():
    import pkg_resources
    dists = [d for d in pkg_resources.working_set]  
    return dists

extensions = [x for x in map(lambda x: x.key, _get_installed_distributions())
              if x.startswith('next')]
for x in extensions:
    try:
        mod = __import__(x.replace('-', '_'))
        print('Using {}'.format(mod.__next_name__))
    except ImportError as e:
        print(f'[ERROR]: Error loading {x}')
        print(f'[ERROR]: {e}')
