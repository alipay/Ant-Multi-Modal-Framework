# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.modules.module_registry import ModuleRegistry


class ClassifierLayer(ModuleRegistry):
    """A classification layer for any model, all details can be seen from
    :class:`antmmf.modules.module_registry.ModuleRegistry`.

    We registered :class:`MLP <antmmf.modules.layers.mlp.MLP>` and :external:py:class:`Linear <torch.nn.Linear>` in
    `antmmf/modules/classifier/__init__.py` since they are simple, and you can call them by set the
    `classifier_type` to `MLP` and `Linear`, respectively.

    Args:
        classifier_type (str): classifier's type.
        in_dim (int): dimension of input features.
        out_dim (int): dimension of output logits, if not set this argument, then the `num_labels` must be passed.
        kwargs (dict): custom configurations.
    """

    def __init__(
        self, classifier_type: str, in_dim: int, out_dim: int = None, **kwargs
    ):
        # add alias `num_labels` for `out_dim`
        out_dim = out_dim or kwargs.get("num_labels")

        # compatible codes, and they will be removed in the future.
        type_mapping = {
            "weight_norm": "WeightNormClassifier",
            "logit": "LogitClassifier",
            "transformer": "TransformerDecoderForClassificationHead",
            "bert": "BertClassifierHead",
            "mlp": "MLP",
            "language_decoder": "LanguageDecoder",
            "linear": "Linear",
        }
        if classifier_type in type_mapping:
            classifier_type = type_mapping[classifier_type]

        arg_mapping = {"nheads": "nhead"}
        for key in list(kwargs.keys()):
            if key in arg_mapping:
                kwargs[arg_mapping[key]] = kwargs.pop(key)

        super(ClassifierLayer, self).__init__(
            classifier_type, in_dim, out_dim, **kwargs
        )
