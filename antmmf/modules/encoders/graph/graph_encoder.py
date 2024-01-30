# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.modules.module_registry import ModuleRegistry


class GraphEncoder(ModuleRegistry):
    """
    A graph encoder register for encoders that composed by graph neural network (GNN), all other details can be
    seen from :class:`antmmf.modules.module_registry.ModuleRegistry`.

    Args:
        graph_encoder_type (str): type of graph encoder.
    """

    def __init__(self, graph_encoder_type: str, **kwargs):
        # compatible codes, and they will be removed in the future.
        type_mapping = {
            "GAT": "GATEncoder",
        }
        if graph_encoder_type in type_mapping:
            graph_encoder_type = type_mapping[graph_encoder_type]

        super(GraphEncoder, self).__init__(graph_encoder_type, **kwargs)
