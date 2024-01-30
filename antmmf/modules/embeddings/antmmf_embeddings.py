# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.modules.module_registry import ModuleRegistry


class AntMMFEmbeddings(ModuleRegistry):
    def __init__(self, config, *args, **kwargs):
        super(AntMMFEmbeddings, self).__init__(
            config.type, *args, **kwargs, **config.params
        )
