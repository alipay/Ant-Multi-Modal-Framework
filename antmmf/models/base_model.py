# Copyright (c) 2023 Ant Group and its affiliates.
"""
Models built on top of antmmf need to inherit ``BaseModel`` class and adhere to
some format. To create a model for antmmf, follow this quick cheatsheet.

1. Inherit ``BaseModel`` class, make sure to call ``super().__init__()`` in your
   class's ``__init__`` function.
2. Implement `build` function for your model. If you build everything in ``__init__``,
   you can just return in this function.
3. Write a `forward` function which takes in a ``SampleList`` as an argument and
   returns a dict.
4. Register using ``@registry.register_model("key")`` decorator on top of the
   class.

If you are doing logits based predictions, the dict you return from your model
should contain a `logits` field. Losses and Metrics are automatically
calculated by the ``BaseModel`` class and added to this dict if not present.

Example::

    import torch

    from antmmf.common.registry import registry
    from antmmf.models.base_model import BaseModel


    @registry.register("antmmf")
    class AntMMF(BaseModel):
        # config is model_attributes from global config
        def __init__(self, config):
            super().__init__(config)

        def build(self):
            ....

        def forward(self, sample_list):
            scores = torch.rand(sample_list.get_batch_size(), 3127)
            return {"logits": scores}
"""


import collections
import warnings
from copy import deepcopy

import torch.nn as nn

from antmmf.common import constants
from antmmf.common.checkpoint import load_pretrained_model
from antmmf.common.registry import registry
from antmmf.modules.losses import Losses
from antmmf.modules.metrics import Metrics


class BaseModel(nn.Module):
    """For integration with antmmf's trainer, datasets and other feautures,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (Configuration): ``model_attributes`` configuration from global config.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.writer = registry.get("writer")
        self.global_config = registry.get("config", {})
        self._is_pretrained = False
        self._set_registry_for_model()

    def _set_registry_for_model(self):
        """
        Get values registered in datasets.
        The dataset builder should save keys of values that is expected to get accessed for model,
        say key_registry_for_model in registry with key 'constants.REGISTRY_FOR_MODEL'. This method
        will automatically load those values with given keys for model.
        """
        setattr(self, "dataset_name", registry.get(constants.DATASET_NAME))
        registry_keys = registry.get(constants.REGISTRY_FOR_MODEL, [])
        for key in registry_keys:
            setattr(self, key, registry.get(key))

    @property
    def is_pretrained(self):
        return self._is_pretrained

    @is_pretrained.setter
    def is_pretrained(self, x):
        self._is_pretrained = x

    def build(self):
        """Function to be implemented by the child class, in case they need to
        build their model separately than ``__init__`` for training phrase.
        All model related downloads should also happen here.
        """
        raise NotImplementedError(
            "Build method not implemented in the child model class."
        )

    def build_for_test(self):
        """Function to be implemented by the child class, in case they need to
        build their model for unittest phrase, where models should not load
        pre-trained weights and be randomly initialized. Model related downloads
        is **not** allowed.

        Note: For online serving, models should be built with randomly initialized
        first, then Predictor will load model weights provided by users. So this method
        is also needed to be implemented.
        """
        raise NotImplementedError(
            "Build_For_Test method not implemented in the child model class."
        )

    def forward_graph(self, *args):
        """
        Function used to wrap model graph that needs to be exported by ONNX.
        AntMMF provides `pri/maya/onnx_format/maya_exporter` utils to convert antmmf pytorch model to ONNX format,
        the maya_exporter utils will automatically use this method to find model graph  for exporting ONNX graph.
        This method must be implemented if you need to convert your torch model to ONNX format.

        The best practice is to call `forward_graph` in `forward` method. See examples in
        `antmmf.models.mmbt`

        :param args(Tuple[torch.Tensor] or torch.Tensor): args is needed by
                    ONNX converter:https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
        :return: Dict
        """
        raise NotImplementedError("forward_graph method not implemented")

    def init_losses_and_metrics(self):
        """Initializes loss and metrics for the model based ``losses`` key
        and ``metrics`` keys. Automatically called by antmmf internally after
        building the model.
        """
        losses = self.config.get("losses", [])
        metrics = self.config.get("metrics", [])
        if len(losses) == 0:
            warnings.warn(
                "No losses are defined in model configuration. You are expected "
                "to return loss in your return dict from forward."
            )

        if len(metrics) == 0:
            warnings.warn(
                "No metrics are defined in model configuration. You are expected "
                "to return metrics in your return dict from forward."
            )
        self.losses = Losses(losses)
        self.metrics = Metrics(metrics)

    @classmethod
    def init_args(cls, parser):
        return parser

    @classmethod
    def format_state_key(cls, key):
        """Can be implemented if something special needs to be done to the
        key when pretrained model is being loaded. This will adapt and return
        keys according to that. Useful for backwards compatibility. See
        updated load_state_dict below. For an example, see VisualBERT model's
        code.

        Args:
            key (string): key to be formatted

        Returns:
            string: formatted key
        """
        return key

    @classmethod
    def strip_head(cls):
        """strip the classifier head when pretrained model is being loaded.

        Args:
            key (string): key to be striped

        Returns:
            string: backbone keys
        """
        return []

    def load_state_dict(self, state_dict, *args, **kwargs):
        copied_state_dict = deepcopy(state_dict)
        for key in list(copied_state_dict.keys()):
            formatted_key = self.format_state_key(key)
            copied_state_dict[formatted_key] = copied_state_dict.pop(key)
        # strip classifer head if needed
        strip_head = kwargs.pop("strip_head", False)
        if strip_head:
            copied_state_dict = self.strip_head(copied_state_dict)

        return super().load_state_dict(copied_state_dict, *args, **kwargs)

    def forward(self, sample_list, *args, **kwargs):
        """To be implemented by child class. Takes in a ``SampleList`` and
        returns back a dict.

        Args:
            sample_list (SampleList): SampleList returned by the DataLoader for
            current iteration

        Returns:
            Dict: Dict containing logits object.

        """
        raise NotImplementedError(
            "Forward of the child model class needs to be implemented."
        )

    def __call__(self, sample_list, *args, **kwargs):
        model_output = super().__call__(sample_list, *args, **kwargs)

        # Make sure that the output from the model is a Mapping
        assert isinstance(
            model_output, collections.abc.Mapping
        ), "A dict must be returned from the forward of the model."

        # Disable losses & metrics calculation when online serving, evalai_inference
        # or synchronized loss calculation
        if (
            registry.get(constants.STATE, None) == constants.STATE_ONLINE_SERVING
            or registry.get(constants.EVALAI_INFERENCE, False, no_warning=True)
            or self.global_config.get("training_parameters", {}).get(
                "synchronized_loss", False
            )
        ):
            return model_output

        if "losses" in model_output:
            warnings.warn(
                "'losses' already present in model output. No calculation will be done in base model."
            )
            assert isinstance(
                model_output["losses"], collections.abc.Mapping
            ), "'losses' must be a dict."
        else:
            loss = self.losses(sample_list, model_output, *args, **kwargs)
            if loss is not None:
                model_output["losses"] = loss
            # in case there is no output from loss, ignore it.
            # this happens either due to errors in computation, which is for fault-tolerance
            # or loss is not computed for each mini-batch

        if "metrics" in model_output:
            warnings.warn(
                "'metrics' already present in model output. No calculation will be done in base model."
            )
            assert isinstance(
                model_output["metrics"], collections.abc.Mapping
            ), "'metrics' must be a dict."
        else:
            model_output["metrics"] = self.metrics(
                sample_list, model_output, *args, **kwargs
            )

        return model_output

    @classmethod
    def from_pretrained(cls, model_name, config, *args, **kwargs):
        writer = registry.get("writer")
        model_key = model_name.split(".")[0]
        model_cls = registry.get_model_class(model_key)
        assert (
            model_cls == cls
        ), f"Incorrect pretrained model key {model_key} for class {cls.__name__}"
        output = load_pretrained_model(model_name, *args, **kwargs)
        pretrained_config, model = output["config"], output["model"]
        # override pretrained model config
        config = pretrained_config.merge_with(config)

        instance = cls(config)
        instance.is_pretrained = True
        instance.build()
        instance.init_losses_and_metrics()
        strip_head = config.from_pretrained.get("strip_head", False)
        incompatible_keys = instance.load_state_dict(
            model,
            strip_head=strip_head,
            strict=False,
        )

        if len(incompatible_keys.missing_keys) != 0:
            writer.write(
                f"Missing keys {incompatible_keys.missing_keys} in the"
                + " checkpoint.\n"
                + f"Unexpected keys if any: {incompatible_keys.unexpected_keys}",
                level="warning",
            )

        if len(incompatible_keys.unexpected_keys) != 0:
            writer.write(
                "Unexpected keys in state dict: "
                + f"{incompatible_keys.unexpected_keys} \n"
                + "This is usually not a problem with pretrained models.",
                level="warning",
            )

        # If the pretrained model's head is striped, it need a new classifier head and
        # would be used in finetuning.
        # If it's not, which means it has its own classifier head, it only could be
        # used in evaluation/inference.

        if not strip_head:
            instance.eval()

        return instance, config
