# -- coding: utf-8 --

import torch
from torch import nn
from typing import Dict
from .graph_decoder import GraphDecoder


@GraphDecoder.register()
class DeltaKGDecoder(nn.Module):
    r"""
    Use TransE/TransH/DTransE... as decoder for heterogeneous graph model。
    Input: encoder result stored in a dict
    Output: pos_score, neg_score_head, neg_score_tail, relation_embedding
    """

    def __init__(self, decoder_type: str = "DTransE"):
        from kgrl.models.pytorch.conf import DecoderType

        super(DeltaKGDecoder, self).__init__()
        self.decoder_type = decoder_type
        funcs = {
            DecoderType.KG_TRANSE: lambda h, t, r: torch.norm(h + r - t, p=1, dim=-1),
            DecoderType.KG_DTRANSE: lambda h, t, r: torch.norm(h * r - t, p=1, dim=-1),
            DecoderType.KG_PAIRRE: lambda h, t, r: torch.norm(
                h * r - t * r, p=1, dim=-1
            ),
        }
        self.delta_func = funcs[self.decoder_type]

    def forward(self, decoder_input: Dict[str, torch.Tensor]):
        pos_score, neg_head_score, neg_tail_score, r = None, None, None, None
        r = decoder_input["update_rel_embed"].index_select(
            0, decoder_input["edge_type"].flatten()
        )
        if "node1_encoder_result" in decoder_input:
            pos_score = self.delta_func(
                decoder_input["node1_encoder_result"],
                decoder_input["node2_encoder_result"],
                r,
            )
        if "head_neg_encoder_result" in decoder_input:
            neg_head_score = self.delta_func(
                decoder_input["head_neg_encoder_result"],
                decoder_input["node2_encoder_result"],
                r,
            )
        if "tail_neg_encoder_result" in decoder_input:
            neg_tail_score = self.delta_func(
                decoder_input["node1_encoder_result"],
                decoder_input["tail_neg_encoder_result"],
                r,
            )

        return pos_score, neg_head_score, neg_tail_score, r
