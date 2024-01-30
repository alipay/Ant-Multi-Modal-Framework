# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

"""
Losses module contains implementations for various losses used generally
in vision and language space. One can register custom losses to be detected by
antmmf using the following example.

.. code::

   from antmmf.common.registry import registry
   from torch import nn


   @registry.register_loss("custom")
   class CustomLoss(nn.Module):
       ...

Then in your model's config you can specify ``losses`` attribute to use this loss
in the following way:

.. code::

   model_attributes:
       some_model:
           losses:
               - type: custom
               - params: {}
"""

import torch
from torch import nn
from typing import Union, Dict
from antmmf.common import Configuration
from antmmf.common.registry import registry


def _check_loss_parameters(params: Union[Dict, Configuration]):
    if "type" not in params:
        raise ValueError(
            "Parameters to loss must have 'type' field to specify type of loss to instantiate"
        )

    loss_type = params["type"]
    loss_class = registry.get_loss_class(loss_type)
    if loss_class is None:
        raise ValueError(f"No loss named {loss_type} is registered to registry")


class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_attributes`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (List[Configuration]): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instanttions of each loss
                                   passed in config
    """

    def __init__(self, loss_list):
        super().__init__()
        self.losses = []
        for loss in loss_list:
            self.losses.append(AntMMFLoss(loss))

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Takes in the original ``SampleList`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample_list (SampleList): SampleList given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}

        # no need to have the constraints of having a target
        # this occurs for unsupervised learning

        for loss in self.losses:
            computed_loss = loss(sample_list, model_output, *args, **kwargs)
            if computed_loss is not None:
                output.update(computed_loss)

        if len(output) > 0:
            registry_loss_key = "{}.{}.{}".format(
                "losses", sample_list.dataset_name, sample_list.dataset_type
            )
            # Register the losses to registry
            registry.register(registry_loss_key, output)

        return output


class AntMMFLoss(nn.Module):
    """Internal antmmf helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/logit_bce": 27.4}``, in case
    `logit_bce` is used and SampleList is from `val` set.

    Args:
        params (str, dict, Configuration): Description of parameter `params`.

    .. note::

        Since, ``AntMMFLoss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params: Union[str, Dict, Configuration]):
        super().__init__()
        self.writer = registry.get("writer")

        if isinstance(params, str):
            params = {"type": params}

        _check_loss_parameters(params)

        loss_type = params["type"]
        loss_class = registry.get_loss_class(loss_type)
        loss_params = params.get("params", {})
        self.loss_criterion = loss_class(**loss_params)

        self.name = params.pop("name", loss_type)

    def forward(self, sample_list, model_output, *args, **kwargs):
        loss = self.loss_criterion(sample_list, model_output, *args, **kwargs)

        if loss is None:
            return loss

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)

        if loss.dim() == 0:
            loss = loss.view(1)

        return {"{}/{}".format(sample_list.dataset_type, self.name): loss}
