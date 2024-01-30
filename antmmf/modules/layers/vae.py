# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from typing import Union, Callable, Optional


class VAE(nn.Module):
    """Multiple Layer Perceptron.

    Args:
        in_dim (int): dimension of input.
        out_dim (int): dimension of output.
        hidden_dim (int): dimension of hidden layer(s), its default value is in_dim.
        num_layers (int): the number of hidden layer(s), default value is 0.
        dropout (float): dropout ratio, it works only when it is set to a float in the range from 0 to 1, default None.
        activation (Union[str, nn.Module, Callable]): activation function. If it is a string, then we will map it into
          the function of torch according to :func:`transformers.activations.get_activation`. Moreover, it can also be a
          :external:py:class:`Module <torch.nn.Module>`, callable function, or `None`, and its default value is "relu".
        batch_norm (bool): whether use BatchNorm1d or not, default value is True.
        weight_norm (bool): apply :external:py:func:`weight_norm <torch.nn.utils.weight_norm>` on the layer(s) Linear,
          default value is False.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        class_num: int,
        hidden_dim: int = None,
        dropout: float = 0.2,
        activation: Union[str, nn.Module, Callable] = "tanh",
        **kwargs,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        self.dropout = nn.Dropout(dropout)
        self.taskid2labelembedding = nn.Linear(
            class_num, hidden_dim
        )  # task1 label embedding
        self.hidden2hidden = nn.Linear(
            hidden_dim, hidden_dim
        )  # label embedding to hidden vector
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.Relu()

        self.posterior = nn.Linear(in_dim + hidden_dim, out_dim)
        self.posterior_mu = nn.Linear(out_dim, out_dim)
        self.posterior_log_Sigma = nn.Linear(out_dim, out_dim)

        self.prior = nn.Linear(in_dim, out_dim)
        self.prior_mu = nn.Linear(out_dim, out_dim)
        self.prior_log_Sigma = nn.Linear(out_dim, out_dim)

    def forward(
        self,
        input_emb: torch.Tensor,
        targets: Optional[torch.LongTensor] = None,
        prototype_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_emb: [bs, in_dim]
        targets: [bs, class_num]
        prototype_emb: [class_num, hidden_dim]
        """
        # prior:
        P_z = self.prior(input_emb)
        P_z = self.activation(P_z)
        P_mu = self.prior_mu(P_z)
        P_Sigma = torch.exp(self.prior_log_Sigma(P_z))

        # Reparameterize
        z = P_mu
        KL = None

        # posterior
        if targets is not None:
            targets = targets.float()
            if prototype_emb is not None:
                labelembedding = (
                    torch.matmul(targets, prototype_emb)
                    / torch.sum(targets, dim=-1, keepdim=True).float()
                )
            else:
                labelembedding = (
                    self.taskid2labelembedding(targets)
                    / torch.sum(targets, dim=-1, keepdim=True).float()
                )
                labelembedding = self.hidden2hidden(labelembedding)
            label_hidden = self.activation(labelembedding)
            label_hidden = self.dropout(label_hidden)
            Q_z = self.posterior(torch.cat((input_emb, label_hidden), dim=-1))

            Q_z = self.activation(Q_z)
            Q_mu = self.posterior_mu(Q_z)
            Q_Sigma = torch.exp(self.posterior_log_Sigma(Q_z))

            # Reparameterize
            Q_sigma = torch.sqrt(Q_Sigma)
            z = Q_mu + torch.randn_like(Q_sigma) * Q_sigma

            KL = (
                -0.5
                * torch.sum(
                    torch.log(Q_Sigma / P_Sigma)
                    - Q_Sigma / P_Sigma
                    - torch.square(Q_mu - P_mu) / P_Sigma
                    + 1,
                    axis=-1,
                ).mean()
            )

        return KL, z
