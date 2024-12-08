# Copyright (C) Laouen Belloli - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Laouen Belloli <laouen.belloli@gmail.com>, May 2022

_pipelines = dict()

def get_pipeline_configs(config):
    """
    This function returns the pipeline config dict specifying the pipeline steps
    with their configs and parameters to use. 

    Args:
        config (str): The config name.

    Returns:
        The pipeline config dict specifying the pipeline steps
        with their configs and parameters to use.

    Raises:
        ValueError: If the config is not registered.
    """
    if config in _pipelines:
        return _pipelines[config]
    else:
        raise ValueError(f'No pipeline {config} registered')

def register_pipeline(pipeline, config):
    _pipelines[config] = pipeline
    print(f'Registered pipeline {config}')

# Decorator to register pipelines
def next_pipeline(config):
    def wrapper(pipeline):
        register_pipeline(pipeline, config)

    return wrapper