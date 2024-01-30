# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.modules.module_registry import ModuleRegistry
from antmmf.modules.transformers.base import TransformerDecoder


class Decoder(ModuleRegistry):
    def __init__(self, config, **kwargs):
        super().__init__(config.type, **config.params, **kwargs)


Decoder.register(TransformerDecoder)
