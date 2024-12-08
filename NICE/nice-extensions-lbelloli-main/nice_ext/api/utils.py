# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import sys
from os import path as op
import time
import subprocess
from distutils.version import LooseVersion
import logging

from mne.utils import logger
from mne.utils._logging import WrapStdOut


def _get_git_head(path):
    """Aux function to read HEAD from git"""
    if not isinstance(path, str):
        raise ValueError('path must be a string, you passed a {}'.format(
            type(path))
        )
    if not op.exists(path):
        raise ValueError('This path does not exist: {}'.format(path))
    command = ('cd {gitpath}; '
               'git rev-parse --verify HEAD').format(gitpath=path)
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               shell=True)
    proc_stdout = process.communicate()[0].strip()
    del process
    return proc_stdout


def get_versions(sys):
    """Import stuff and get versions if module

    Parameters
    ----------
    sys : module
        The sys module object.

    Returns
    -------
    module_versions : dict
        The module names and corresponding versions.
    """
    module_versions = {}
    for name, module in sys.modules.items():
        if '.' in name:
            continue
        if '_curses' == name:
            continue
        module_version = LooseVersion(getattr(module, '__version__', None))
        module_version = getattr(module_version, 'vstring', None)
        if module_version is None:
            module_version = None
        elif 'git' in module_version:
            git_path = op.dirname(op.realpath(module.__file__))
            head = _get_git_head(git_path)
            module_version += '-HEAD:{}'.format(head)

        module_versions[name] = module_version
    return module_versions


def log_versions():
    versions = get_versions(sys)

    logger.info('===== Lib Versions =====')
    logger.info('Numpy: {}'.format(versions['numpy']))
    logger.info('Scipy: {}'.format(versions['scipy']))
    logger.info('MNE: {}'.format(versions['mne']))
    logger.info('scikit-learn: {}'.format(versions['sklearn']))
    logger.info('nice: {}'.format(versions['nice']))
    # logger.info('nice-jsmf: {}'.format(versions['njsmf']))
    # TODO: Log nice extensions versions
    logger.info('========================')


def get_run_id():
    """Get the run id

    Returns
    -------
    run_id : str
        A hex hash.
    """
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())


def configure_logging(fname=None):
    """Set format to file logging and add stdout logging
       Log file messages will be: DATE - LEVEL - MESSAGE
    """
    remove_file_logging()
    file_output_format = '%(asctime)s %(levelname)s %(message)s'
    date_format = '%d/%m/%Y %H:%M:%S'
    formatter = logging.Formatter(file_output_format, datefmt=date_format)

    if fname is not None:
        fh = logging.FileHandler(fname, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler(WrapStdOut())
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def remove_file_logging():
    """Close and remove logging to file"""
    handlers = logger.handlers
    for h in handlers:
        h.close()
        logger.removeHandler(h)


def _parse_value(vstr):
    """
    Parses a string representation of a python value into an actual python value

    Args:
        vstr (str): A string representation of a python value
    
    Returns:
        v (None|bool|str|float|int): The parsed value
    """
    if vstr in ['None']:
        v = None

    elif vstr in ['True', 'true', 'False', 'false']:
        v = vstr in ['True', 'true']

    elif '"' in vstr:
        v = vstr.replace('"', '')

    else: # it's a number

        # using try to allow any string representation of floats (dot or cientific)
        try:
            # if it can be parsed as int, it's not float 
            v = int(vstr)
        except ValueError as _:
            v = float(vstr)
    
    return v

def _parse_iterable(vstr):
    """
    Parses a list or tuple of values from string to python objects

    Args:
        vstr (str): A string representation of python list
    """
    elems = vstr.replace('(', '').replace(')', '').split(',')
    values = [_parse_value(v) for v in elems]
    if vstr.startswith('('):
        values = tuple(values)
    return values

def parse_params_from_config(config):
    """
    Converts a GET formated query string into a config target and a dictionary
    of parameters.

    Args:
        config (str): A GET formated query string.

    Returns:
        params (dict(str,value)): all the parameters parsed as None, bool,
        str, float or integer or iterable of them
        config_name (str): the config target name
    
    Raises:
        ValueError: If the config query string is incorrect
    """

    # TODO: better handling here
    params = {}
    if '?' in config:
        try:
            query = config.split('?')[1]
            for param in query.split('&'):
                if len(param) == 0:
                    continue
                k, v = param.split('=')
                if v.startswith('[') or v.startswith('('):
                    v = _parse_iterable(v)
                else:
                    v = _parse_value(v)
                params[k] = v
        except Exception:
            raise ValueError('Malformed config query {}'.format(config))
    return params, config.split('?')[0]
