# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

_modules = dict()


def _get_module_func(section, config):
    """
    This function returns the function specified by the config parameter that is
    stored in the specified section (the section must be already registered)

    Args:
        section (str): The section where the function is located.
        config (str): The config inside the section where the function is located.

    Returns:
        The function specified by section and config. This function takes as parameter the path and the 
        config_params dictionary

    Raises:
        ValueError: If section is not yet registerd.
        ValueError: If the config is not in the section.
    """
    instance = None
    if '?' in config:
        config = config.split('?')[0]
    if section in _modules:
        this_section = _modules[section]
        if config in this_section:
            instance = this_section[config]
        else:
            options = '\n'.join(this_section.keys())
            raise ValueError('No {} for config {}. Options are:'
                             '\n{}'.format(section, config, options))
    else:
        raise ValueError('No section {} registered'.format(section))
    return instance


def register_module(section, module_name, module):
    """
    Registers a module (function) in a specified section with a specified module_name

    Args:
        section (str): The section where to store the module
        module_name (str): The name of the module to save
        module (Callable): The module to register
    """
    if section not in _modules:
        _modules[section] = dict()
    _modules[section][module_name] = module
 

def _split_configs(config):
    configs = config.split('/')
    module = configs[0]
    subconfig = ''
    if len(configs) > 1:
        subconfig = '/'.join(configs[1:])

    return module, subconfig


def check_config(section, config):
    """
    A public access to the internal _get_module_func.
    """
    _get_module_func(section, config)

# TODO (Lao): This function is a particular case of .utils.parse_params_from_config
# Maybe we could deprecate this method to use the other that is more general.
def split_config_params(config):
    """
    Converts a GET formated query string into a config target and a dictionary
    of parameters.

    Args:
        config (str): A GET formated query string.

    Returns:
        A tuple of the config target name and a dict with all the parameters 
        parsed as boolean, float or integer
    
    Raises:
        ValueError: If the config query string is incorrect
    """
    params = {}
    if '?' in config:
        try:
            query = config.split('?')[1]
            for param in query.split('&'):
                k, v = param.split('=')
                if v in ['True', 'true', 'False', 'false']:
                    v = v in ['True', 'true']
                elif '.' in v:
                    v = float(v)
                else:
                    v = int(v)
                params[k] = v
        except:
            raise ValueError('Malformed config query {}'.format(config))
    return config.split('?')[0], params


# Decorator to register modules
def next_module(section, module_name, module_description=''):
    def wrapper(module):
        module.__description__ = module_description
        register_module(section, module_name, module)

    return wrapper