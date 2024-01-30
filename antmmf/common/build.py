# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.common import Configuration
from antmmf.common.registry import registry


def build_config(
    config_file: str,
    config_override: str = None,
    opts_override=None,
    specific_override=None,
    args=None,
) -> Configuration:
    """
    Build configuration from command line.
    """

    configuration = Configuration(config_file)
    # Update with the config override if passed
    configuration.override_with_cmd_config(config_override)
    # Now, update with opts args that were passed
    configuration.override_with_cmd_opts(opts_override)
    # Finally, update with args that were specifically passed
    # as arguments
    if specific_override is not None:
        # Temporary solution for update the variable of `local_rank` to configuration.training_parameters
        distributed_training_options = vars(specific_override)
        configuration.training_parameters.update(distributed_training_options)
    configuration.freeze()

    registry.register("config", configuration)
    registry.register("configuration", configuration)
    return configuration
