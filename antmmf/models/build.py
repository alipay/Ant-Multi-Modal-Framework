# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.utils.general import is_method_override
from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel


def build_model(config, for_test=False):
    model_name = config.model

    model_class = registry.get_model_class(model_name)
    assert model_class is not None, "No model registered for name: {}".format(
        model_name
    )
    model = model_class(config)

    if for_test and is_method_override(model_class, BaseModel, "build_for_test"):
        model.build_for_test()

    elif is_method_override(model_class, BaseModel, "build"):
        model.build()

    model.init_losses_and_metrics()
    return model


def build_pretrained_model(config):
    assert (
        config.from_pretrained is not None
    ), "config.from_pretrained should not be None"
    model_name = config.from_pretrained.model_name
    model_key = model_name.split(".")[0]
    model_class = registry.get_model_class(model_key)
    assert model_class is not None, "No model registered for name: {}".format(model_key)
    model, config = model_class.from_pretrained(model_name, config)

    return model, config
