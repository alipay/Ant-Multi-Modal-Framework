# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from antmmf.modules.layers.mlp import MLP
from antmmf.modules.classifier.classifier_layer import ClassifierLayer
from antmmf.modules.classifier.weight_norm_classifier import WeightNormClassifier
from antmmf.modules.classifier.bert_classifier_head import BertClassifierHead
from antmmf.modules.classifier.logit_classifier import LogitClassifier
from antmmf.modules.classifier.transformer_decoder import (
    TransformerDecoderForClassificationHead,
)

# register the Linear, since it is very simple
ClassifierLayer.register(torch.nn.Linear)
ClassifierLayer.register(MLP)

__all__ = [
    "ClassifierLayer",
    "WeightNormClassifier",
    "BertClassifierHead",
    "LogitClassifier",
    "TransformerDecoderForClassificationHead",
]
