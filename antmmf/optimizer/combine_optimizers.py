# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Dict, Optional, Union

from antmmf.common.registry import registry
from antmmf.common import Configuration
from torch.optim.optimizer import Optimizer


@registry.register_optimizer("combined_optimizer")
class CombinedOptimizer(Optimizer):
    """
    Combine multiple optimizers but behave like one optimizer, we illustrate its usage
    by an example. To use CombinedOptimizer during training, one should
    1. configure optimizer_attributes in config.yaml

    optimizer_attributes:
        type: combined_optimizer
        params:
          optimizers:
            - type: AdamW
              params:
                lr: 1e-4
                weight_decay: 0.01
            - type: sgd
              params:
                lr: 1e−2
                weight_decay: 5e-4

    2. Implement method ` get_optimizer_parameters(self, config) -> Dict` in Model definition,
    which returns the parameters corresponding to optimizer_type, eg.
    `{"AdamW": adamW_grouped_parameters, "sgd": sgd_grouped_parameters}`

    """

    def __init__(self, params, optimizers: List[Union[Dict, Configuration]], **configs):
        self._optimizer_instance = []
        self._optimizer_types = []

        for optimizer in optimizers:
            opt_class = registry.get_optimizer_class(optimizer.type)
            if isinstance(params, dict):  # get parameters to optimize
                opt_params = params[optimizer.type]
            else:
                assert len(optimizers) == 1
                opt_params = params  # support only one optimizer for compatibility
            opt_instance = opt_class(opt_params, **optimizer.params)
            self._optimizer_instance.append(opt_instance)
            self._optimizer_types.append(optimizer.type)

        # Access all optimizers param_groups, which means we apply the same LRScheduler
        # for all optimizers. This is not a good assumption for now. Fix it later
        self.param_groups = [
            group for opt in self._optimizer_instance for group in opt.param_groups
        ]

    def get_optimizers_lr_str(self):
        """
        display each optimizer's learning rate for logging purpose during training
        """
        display_lr = ["["]
        for instance, optimizer_type in zip(
            self._optimizer_instance, self._optimizer_types
        ):
            lr_str = "{}: {:.5f}".format(
                optimizer_type, instance.param_groups[0]["lr"]
            ).rstrip("0")
            display_lr.append(lr_str)
        display_lr.append("]")
        return " ".join(display_lr)

    def __getstate__(self):
        return {
            "defaults": [opt.defaults for opt in self._optimizer_instance],
            "state": [opt.state for opt in self._optimizer_instance],
            "param_groups": [opt.param_groups for opt in self._optimizer_instance],
        }

    def __setstate__(self, state):
        for st, opt in zip(state, self._optimizer_instance):
            opt.__setstate__(st)

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        return_dict = {
            "state": [],
            "param_groups": [],
        }
        for opt in self._optimizer_instance:
            state = opt.state_dict()
            return_dict["state"].append(state["state"])
            return_dict["param_groups"].append(state["param_groups"])
        return return_dict

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        for i, opt in enumerate(self._optimizer_instance):
            opt_state_dict = {
                "state": state_dict["state"][i],
                "param_groups": state_dict["param_groups"][i],
            }
            opt.load_state_dict(opt_state_dict)

    def step(self, closure=None):
        total_loss = 0.0 if closure is not None else None
        for optimizer in self._optimizer_instance:
            loss = optimizer.step(closure)
            if loss is not None:
                total_loss += loss
        return total_loss

    def zero_grad(self, set_to_none: Optional[bool] = ...) -> None:
        for optimizer in self._optimizer_instance:
            optimizer.zero_grad()
