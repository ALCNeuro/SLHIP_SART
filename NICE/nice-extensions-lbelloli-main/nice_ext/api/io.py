# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from pathlib import Path

from mne.utils import logger

from .modules import _get_module_func, register_module

_io_config_suffix_map = {
    'default': 'default-epochs.fif',
}


def _check_path(path):
    """
    Checks whether the parameter is a Path, if not it convert the parameter 
    to a Path
    
    Args:
        path (Path | str | list): The Path or str or a list of path or string 
    
    Returns:
        A Path object of the path passed as parameter or a list of paths (recursively parsed)
    """
    if isinstance(path, list):
        path = [_check_path(p) for p in path]
    
    elif not isinstance(path, Path):
        path = Path(path)

    return path


def register_suffix(config, suffix):
    if config in _io_config_suffix_map:
        logger.warning('Overwriting IO suffix for {}'.format(config))
    _io_config_suffix_map[config] = suffix


def _check_io_suffix(path, config, multiple, missing_allowed=False):
    """
    Checks whether the suffix is registered and return all files in path with
    the given suffix
        
    Args:
        path (str): Root path where to look for files with the suffix pattern
        config (str): The suffix key in the map of registered suffixes
        multiple (bool): If true, multiple files can match the patter. 
        Otherwise raise error if multiple files matches the suffix
        missing_allowed (bool): Optional. If True, then the function can return 
        an empty list of files if no matches where available. Otherwise raises
        error if no file was found. Default is False
    
    Returns:
        Returns a list of files paths.

    Raises:
        ValueError: If missing_allowed is False and there is no matches
        ValueError: If multiple is False and there is more than one matches
    """

    # Comment (Lao): I added this check to avoid meaningless crashes and to make
    # sense with the function name
    if config not in _io_config_suffix_map:
        existent_suffixes = '\n\t'.join(['', *_io_config_suffix_map.keys()])
        raise KeyError(f'Suffix {config} no yet registered,'
                        'registered suffixes are:' + existent_suffixes)

    suffix = _io_config_suffix_map[config]
    path = _check_path(path)
    files = [x for x in path.glob(f'*{suffix}')]
    if missing_allowed is False and len(files) == 0:
        msg = f'No files for {config} suffix: {suffix}'
        logger.error(msg)
        raise ValueError(msg)
    if multiple is False and len(files) > 1 and missing_allowed is False:
        msg = f'Only one file must have the {config} suffix: {suffix}'
        logger.error(msg)
        raise ValueError(msg)
    elif multiple is True and len(files) == 0 and missing_allowed is False:
        msg = f'At least one file must have the {config} suffix: {suffix}'
        logger.error(msg)
        raise ValueError(msg)
    return files


def read(path, config='default', config_params=None):
    """
    Read the data from path using a specified read configuration/method

    Args:
        path (str | Path | list): The path of the data to read, if string, it must be 
        compatible to convert to pathlib.Path if list, then a list of data files to read.
        config (str): Optional. A configuration formated string: TODO. check what are the configuration formats
        config_params (dict): Optional. A dictionary of parameters for the 
        configuration. TODO: Check what are the available parameters and what rol dose they play

    Returns:
        The raw data processed by the function specified by the config parameter
    """
    logger.info(f'Reading data using {config} config')
    if config_params is None:
        config_params = {}
    out = None
    func = _get_module_func('io', config)
    path = _check_path(path)
    out = func(path, config_params=config_params)
    return out

# Decorator to register a io module
def next_io_module(module_name, module_description='', suffix=None):
    def wrapper(module):
        module.__description__ = module_description
        register_module('io', module_name, module)
        
        if suffix is not None:
            register_suffix(module_name, suffix)

    return wrapper