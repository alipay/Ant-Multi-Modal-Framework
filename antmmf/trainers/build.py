# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import Union
from argparse import Namespace
from antmmf.common import Configuration
from antmmf.common.registry import registry
from antmmf.optimizer.build import build_optimizer
from antmmf.trainers.remote_trainer import RemoteTrainer


def build_trainer(
    config: Configuration,
    args: Union[Namespace, Configuration],
):
    if args.remote:
        return RemoteTrainer(config)

    trainer_type = config.training_parameters.trainer
    trainer_cls = registry.get_trainer_class(trainer_type)
    trainer_obj = trainer_cls(config)

    # Set args as an attribute for future use
    setattr(trainer_obj, "args", args)

    return trainer_obj


def build_adversarial_training(config, model):
    adversarial_type = config.adversarial_parameters.adversarial
    adversarial_class = registry.get_adversarial_class(adversarial_type)

    assert adversarial_class is not None, f"cannot get {adversarial_type} from registry"
    adversarial_obj = adversarial_class(config, model)
    assert adversarial_obj is not None, f"cannot initiate {adversarial_class}"
    optimizer = build_optimizer(
        adversarial_obj, config, config.adversarial_optimizer_parameters
    )

    return adversarial_obj, optimizer
