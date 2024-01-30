# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from torch.nn import Linear, BatchNorm1d
from .graph_decoder import GraphDecoder


@GraphDecoder.register()
class FFNDecoder(nn.Module):
    def __init__(
        self,
        node_embed_dim: int,
        edge_embed_dim: int,
        num_classes: int = 2,
        neg_self_adversarial: bool = False,
        num_layers: int = 3,
        norm_type: str = "bn",
    ):
        super(FFNDecoder, self).__init__()
        self.neg_self_adversarial = neg_self_adversarial

        assert norm_type in ["bn", "ln", "in", "xn"]
        if norm_type == "bn":
            norm = BatchNorm1d
        elif norm_type in ["ln", "xn"]:
            norm = nn.LayerNorm
        elif norm_type == "in":
            norm = nn.InstanceNorm1d
        else:
            raise NotImplementedError(
                f"FFNDecoder's norm only support bn, ln, in and xn, but got {norm_type}"
            )

        in_channels = 2 * node_embed_dim + edge_embed_dim

        channels = [in_channels] + [in_channels // 4] * (num_layers - 1) + [num_classes]
        layers = []
        for i in range(num_layers):
            layers.append(Linear(channels[i], channels[i + 1]))
            layers.append(norm(channels[i + 1]))
            layers.append(nn.ReLU())
        layers = layers[:-2]
        if len(layers) == 0:
            layers.append(nn.Identity())
        self.layers = nn.Sequential(*layers)

    def forward(self, decoder_input):
        p_h = decoder_input["node1_encoder_result"]
        p_t = decoder_input["node2_encoder_result"]
        n_h = decoder_input["head_neg_encoder_result"]
        n_t = decoder_input["tail_neg_encoder_result"]
        p_r = decoder_input["update_rel_embed"].index_select(
            0, decoder_input["edge_type"].flatten()
        )
        p_r = p_r.reshape(p_h.shape)

        inputs_p = torch.cat((p_h, p_r, p_t), dim=-1)
        inputs_all = inputs_p

        def construct_negative_feature(inputs, neg_node):
            if neg_node is not None:
                p_r_i = torch.repeat_interleave(p_r, neg_node.shape[0], dim=0)
                p_t_i = torch.repeat_interleave(p_t, neg_node.shape[0], dim=0)
                inputs_neg = torch.cat((neg_node, p_r_i, p_t_i), dim=-1)
                return torch.cat((inputs, inputs_neg), dim=0)
            return inputs

        inputs_all = construct_negative_feature(inputs_all, n_h)
        inputs_all = construct_negative_feature(inputs_all, n_t)

        inputs_all = inputs_all.view(-1, inputs_all.shape[-1])
        output = self.layers(inputs_all)

        output_weight = None
        if self.neg_self_adversarial and self.training:
            output_neg = output[inputs_p.shape[0] :, :, :]
            output_weight = torch.softmax(
                torch.softmax(output_neg, dim=-1)[:, :, 1], dim=0
            ).detach()
            output_weight = torch.cat(
                (torch.ones_like(inputs_p[:, :, 0]).to(inputs_p.device), output_weight),
                dim=0,
            )

        return output, output_weight
