# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import os

from antmmf.common.build import build_config
from antmmf.common.registry import registry


def build_predictor(config):
    predictor_type = config.predictor_parameters.predictor
    predictor_cls = registry.get_predictor_class(predictor_type)
    assert predictor_cls is not None, f"cannot get {predictor_type} from registry"
    predictor_obj = predictor_cls(config)
    return predictor_obj


def build_predictor_from_args(args, *rest, **kwargs):
    config = build_config(
        args.config,
        config_override=args.config_override,
        opts_override=args.opts,
        specific_override=args,
    )
    predictor_obj = build_predictor(config)
    setattr(predictor_obj, "args", args)
    return predictor_obj


def build_online_predictor(model_dir=None, config_yaml=None):
    assert model_dir or config_yaml
    from antmmf.utils.flags import flags

    # if config_yaml not indicated, there must be a `config.yaml` file under `model_dir`
    config_path = config_yaml if config_yaml else os.path.join(model_dir, "config.yaml")

    input_args = ["--config", config_path]
    if model_dir is not None:
        input_args += ["predictor_parameters.model_dir", model_dir]
    parser = flags.get_parser()
    args = parser.parse_args(input_args)
    predictor = build_predictor_from_args(args)
    return predictor
