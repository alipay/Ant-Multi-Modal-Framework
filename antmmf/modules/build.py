# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn
from antmmf.common import Configuration
from antmmf.common.registry import registry
from antmmf.modules.transformers import heads
from antmmf.modules.classifier import ClassifierLayer
from antmmf.modules.encoders import (
    GraphEncoder,
    VisualEncoder,
    ImageFeatureEncoder,
    MultimodalEncoder,
)
from antmmf.modules.decoders import Decoder
from antmmf.modules.embeddings import AntMMFEmbeddings


def build_transformer_head(config, *args, **kwargs):
    head_cls = getattr(heads, config.type)
    return head_cls(config.params, *args, **kwargs)


def build_fusioner(config):
    fusion_cls = registry.get_fusion_class(config.type)
    assert fusion_cls is not None, f"Fusion of type:{config.type} is not registered"
    return fusion_cls(**config.params)


def build_classifier_layer(config: Configuration) -> nn.Module:
    """
    TODO: add document here.
    """
    classifier = ClassifierLayer(config.type, **config.params)
    return classifier.module


def build_interpreter(config: Configuration, predictor):
    interpreter = config.get("interpreter_parameters", {}).get("interpreter", None)
    if interpreter is None:
        return None

    interpreter_class = registry.get_interpreter_class(interpreter)

    assert (
        interpreter_class is not None
    ), "No interpreter registered for name: {}".format(interpreter)
    interpreter = interpreter_class(
        predictor, config=config.get("interpreter_parameters", {}).get("params", {})
    )

    return interpreter


def build_graph_encoder(config: Configuration, **kwargs):
    graph_encoder = GraphEncoder(config.type, **config.get("params", {}), **kwargs)
    return graph_encoder


def build_visual_encoder(config: Configuration, **kwargs):
    visual_encoder = VisualEncoder(config, **kwargs)
    return visual_encoder.module


def build_image_feature_encoder(config: Configuration, **kwargs):
    image_feature_encoder = ImageFeatureEncoder(config.type, **config.params, **kwargs)
    return image_feature_encoder.module


def build_multimodal_encoder(config, *args, **kwargs):
    module = MultimodalEncoder(config, **kwargs)
    return module.module


def build_decoder(config, *args, **kwargs):
    module = Decoder(config, **kwargs)
    return module.module


def build_embedder(config, *args, **kwargs):
    return AntMMFEmbeddings(config, *args, **kwargs).module
