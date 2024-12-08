#! /usr/bin/env python
#
# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017
#

import os
from os import path as op
from glob import glob

import setuptools # Needed to work with numpy.distutils.core.setup
from numpy.distutils.core import setup

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(os.path.join('nice_ext', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

descr = """NICE python project for MEG
and EEG data analysis (Extensions Package)."""

DISTNAME = 'nice_ext'
DESCRIPTION = descr
MAINTAINER = '@fraimondo'
MAINTAINER_EMAIL = 'federaimondo@gmail.com'
URL = 'http://github.com/fraimondo/nice-extensions'
LICENSE = 'Copyright'
DOWNLOAD_URL = 'http://github.com/fraimondo/nice-extensions'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=open('README.md').read(),
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=['Intended Audience :: Science/Research',
                    'Intended Audience :: Developers',
                    'Programming Language :: Python',
                    'Topic :: Software Development',
                    'Topic :: Scientific/Engineering',
                    'Operating System :: Microsoft :: Windows',
                    'Operating System :: POSIX',
                    'Operating System :: Unix',
                    'Operating System :: MacOS'],
        platforms='any',
        packages=setuptools.find_packages(
            exclude=[
                "*.tests",
                "*.tests.*",
                "tests.*",
                "tests"
            ]),
        package_data={
            'nice_ext': ['equipments/data/*']
        },
        scripts=[
            'scripts/nice_export_configs'
        ]
    )
