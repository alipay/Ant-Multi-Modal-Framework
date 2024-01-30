# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.modules.module_registry import ModuleRegistry


class GraphDecoder(ModuleRegistry):
    """
    A graph encoder register for decoders which is used by graph neural network (GNN), all other details can be
    seen from :class:`antmmf.modules.module_registry.ModuleRegistry`.

    Args:
        graph_decoder_type (str): type of graph decoder.
    """

    def __init__(self, graph_decoder_type: str, **kwargs):
        super(GraphDecoder, self).__init__(graph_decoder_type, **kwargs)
