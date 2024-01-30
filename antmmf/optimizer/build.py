# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import warnings
from antmmf.common.registry import registry
from antmmf.utils.general import get_optimizer_parameters


def build_optimizer(model, config, overwrite_optimizer_attributes=None):
    optimizer_config = (
        config.optimizer_attributes
        if overwrite_optimizer_attributes is None
        else overwrite_optimizer_attributes
    )
    if not hasattr(optimizer_config, "type"):
        raise ValueError(
            "Optimizer attributes must have a 'type' key specifying the type of optimizer. (Custom or PyTorch)"
        )
    optimizer_type = optimizer_config.type

    if not hasattr(optimizer_config, "params"):
        warnings.warn("optimizer attributes has no params defined, defaulting to {}.")

    params = getattr(optimizer_config, "params", {})

    if str(optimizer_type) == "fusedAdam":
        assert registry.get_optimizer_class(optimizer_type) is not None

    if hasattr(torch.optim, optimizer_type):
        optimizer_class = getattr(torch.optim, optimizer_type)
    else:
        optimizer_class = registry.get_optimizer_class(optimizer_type)
        if optimizer_class is None:
            raise ValueError(
                "No optimizer class of type {} present in "
                "either torch or registered to registry".format(optimizer_type)
            )

    log_writer = registry.get("writer")
    log_writer.write(f"apply optimizer_type: {optimizer_type}", log_all=True)

    parameters = get_optimizer_parameters(model, config)
    optimizer = optimizer_class(parameters, **params)
    return optimizer
