# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn
from transformers.models.bert import BertConfig
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from antmmf.common import Configuration
from antmmf.modules.classifier import ClassifierLayer


@ClassifierLayer.register()
class BertClassifierHead(nn.Sequential):
    """
    .. |BPHTransform| replace:: :class:`transformers.modeling_bert.BertPredictionHeadTransform`

    Classifier layer which uses |BPHTransform| as the feature transform module,
    and projects the transformed feature into logits by a :external:py:class:`Linear <torch.nn.Linear>` layer.
    Moreover, its inputs are same as |BPHTransform|.

    Args:
        in_dim (int): dimension of input features.
        out_dim (int): dimension of output logits.
        config (Configuration): configuration of |BPHTransform|.
    """

    def __init__(
        self, in_dim: int = 768, out_dim: int = 2, config: Configuration = None
    ):
        if config is None:
            config = BertConfig.from_pretrained("bert-base-uncased")

        assert config.hidden_size == in_dim
        modules = [
            nn.Dropout(config.hidden_dropout_prob),
            BertPredictionHeadTransform(config),
            nn.Linear(config.hidden_size, out_dim),
        ]

        super(BertClassifierHead, self).__init__(*modules)
