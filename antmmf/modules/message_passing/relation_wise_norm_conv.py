# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from .delta_conv import DeltaConv


class RelationWiseNormConv(DeltaConv):
    def __init__(
        self,
        num_rel: int = None,
        **kwargs,
    ):
        from kgrl.models.pytorch.layers import RelationNorm

        super(RelationWiseNormConv, self).__init__(**kwargs)
        self.norm = RelationNorm(self.in_channels, num_rel)
        self.num_rel = num_rel
        self.edge_type = None

        assert self.residual_fusion_func == self.residual_fusion_add, (
            f"{self.__class__.__name__} only support residual_fusion_mode == 'add'"
            f" please check it again"
        )

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        edge_time_embed,
        edge_type,
        rel_embed,
        index,
        ptr,
        size_i,
    ):
        value = super(RelationWiseNormConv, self).message(
            x_i,
            x_j,
            edge_attr,
            edge_time_embed,
            edge_type,
            rel_embed,
            index,
            ptr,
            size_i,
        )
        return self.norm(value, edge_type)
