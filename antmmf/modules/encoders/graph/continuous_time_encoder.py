# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import numpy as np
import torch
from torch import nn
from .graph_encoder import GraphEncoder


@GraphEncoder.register()
class ContinuousTimeEncoder(torch.jit.ScriptModule):
    r"""
    This is a trainable encoder to map continuous time value into a low-dimension time vector.
    Ref:
     https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/master/module.py

    The input of ts should be like [E, 1] with all time interval as values.

    Any question please contact 七青(180570)
    """

    def __init__(
        self,
        time_embed_dim: int = 128,
        expand_dim: int = None,
        factor: int = 5,
        max_basis_freq: int = 1,
        use_linear_trans: bool = False,
    ):
        super(ContinuousTimeEncoder, self).__init__()
        self.time_dim = time_embed_dim
        self.expand_dim: int = expand_dim if expand_dim is not None else time_embed_dim
        self.factor = factor
        self.max_basis_freq = max_basis_freq
        use_linear_trans = use_linear_trans

        self.basis_freq = nn.Parameter(
            (
                torch.from_numpy(
                    self.max_basis_freq / 10 ** np.linspace(0, 9, self.time_dim)
                )
            ).float()
        )
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())
        if use_linear_trans:
            self.dense = nn.Linear(self.time_dim, self.expand_dim, bias=False)
            nn.init.xavier_normal_(self.dense.weight)
        else:
            self.dense = nn.Identity()

    @torch.jit.script_method
    def forward(self, ts):
        # ts: [E, 1]
        edge_len, dim = ts.size()
        ts = ts.view(edge_len, dim)  # [E, 1]
        map_ts = ts * self.basis_freq.view(1, -1)  # [E, time_dim]
        map_ts = map_ts + self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        harmonic = self.dense(harmonic)
        return harmonic
