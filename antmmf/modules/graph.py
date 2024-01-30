# Copyright (c) 2023 Ant Group and its affiliates.

# pylint: disable= no-member, arguments-differ, invalid-name, cell-var-from-loop
import torch
import torch.nn.functional as F
from torch.nn import (
    Linear,
    Sequential,
    ReLU,
    BatchNorm1d,
    BatchNorm2d,
    Parameter,
    Dropout,
    Conv2d,
)
from antmmf.utils.glob import global_mean_pool
from typing import Callable
from torch import Tensor
from antmmf.modules.message_passing import MessagePassing
from antmmf.utils.init import reset, xavier_normal
from antmmf.utils.scatter import scatter_add
from antmmf.modules.utils import ccorr


"""
Implementation of Graph I
according to paper
https://arxiv.org/pdf/1810.00826.pdf

GRAPH ISOMORPHISM NETWORK (GIN)

modified from
https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py

More details on creating new graph neural networks are at
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

Citation:
If using the following code, need to cite the following paper:
```
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```
"""


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)
    or
    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),
    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`antmmf.modules.MessagePassing`.
    """

    def __init__(
        self, nn: Callable, eps: float = 0.0, train_eps: bool = False, **kwargs
    ):
        super(GINConv, self).__init__(aggr="add", **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, size=None) -> Tensor:
        """ """
        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)
    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`antmmf.modules.MessagePassing`.
    """

    def __init__(
        self, nn: Callable, eps: float = 0.0, train_eps: bool = False, **kwargs
    ):
        super(GINEConv, self).__init__(aggr="add", **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr=None, size=None) -> Tensor:
        """ """
        if isinstance(x, Tensor):
            x = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        else:
            raise Exception("unknown type")

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BatchNorm1d(hidden),
            ),
            train_eps=True,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BatchNorm1d(hidden),
                    ),
                    train_eps=True,
                )
            )
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


"""
Implementation of "Composition-Based Multi-Relational Graph Convolutional Networks"
according to paper https://arxiv.org/pdf/1911.03082.pdf

modified from https://github.com/malllabiisc/CompGCN
"""


class CompGCNConvBasis(MessagePassing):
    r"""
    CompGCN convolutional layer basis
    Args:
        in_channels (int): Input dimension
        out_channels (int): Output dimension
        num_rels (int): Number of relations
        num_bases (int): Number of basis relation vectors
        act (function, optional): Activation function
        cache (bool, optional): Cache intermediate results like in_norm, out_norm
        gcn_drop (float, optional): Dropout ratio
        opn (string, optional): Composition operation
        bias (bool, optional): use bias
        **kwargs (optional): Additional arguments of class `antmmf.modules.MessagePassing`
    Return:
        torch.FloatTensor: node embedding with shape (num_nodes, out_channels)
        torch.FloatTensor: rel embedding with shape (num_rel, out_channels)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_rels,
        num_bases,
        act=lambda x: x,
        cache=True,
        gcn_drop=0.5,
        opn="corr",
        bias=False,
        **kwargs
    ):
        super(self.__class__, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.cache = cache  # Should be False for graph classification tasks
        self.opn = opn
        self.bias = bias

        def get_param(shape):
            param = Parameter(Tensor(*shape))
            xavier_normal(param)
            return param

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.rel_basis = get_param((num_bases, in_channels))
        self.rel_wt = get_param((num_rels * 2, num_bases))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))
        self.drop = Dropout(gcn_drop)
        self.bn = BatchNorm1d(out_channels)

        (
            self.in_norm,
            self.out_norm,
            self.in_index,
            self.out_index,
            self.in_type,
            self.out_type,
            self.loop_index,
            self.loop_type,
        ) = (None, None, None, None, None, None, None, None)

        if bias:
            self.register_parameter("bias", Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):
        rel_embed = torch.mm(self.rel_wt, self.rel_basis)
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        num_edges = edge_index.size(1) // 2
        num_nodes = x.size(0)

        if not self.cache or self.in_norm is None:
            self.in_index, self.out_index = (
                edge_index[:, :num_edges],
                edge_index[:, num_edges:],
            )
            self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

            self.loop_index = torch.stack(
                [torch.arange(num_nodes), torch.arange(num_nodes)]
            )
            self.loop_type = torch.full(
                (num_nodes,), rel_embed.size(0) - 1, dtype=torch.long
            )

            self.in_norm = self.compute_norm(self.in_index, num_nodes)
            self.out_norm = self.compute_norm(self.out_index, num_nodes)

        in_res = self.propagate(
            self.in_index,
            x=x,
            edge_type=self.in_type,
            rel_embed=rel_embed,
            edge_norm=self.in_norm,
            mode="in",
        )
        loop_res = self.propagate(
            self.loop_index,
            x=x,
            edge_type=self.loop_type,
            rel_embed=rel_embed,
            edge_norm=None,
            mode="loop",
        )
        out_res = self.propagate(
            self.out_index,
            x=x,
            edge_type=self.out_type,
            rel_embed=rel_embed,
            edge_norm=self.out_norm,
            mode="out",
        )
        out = (
            self.drop(in_res) * (1 / 3)
            + self.drop(out_res) * (1 / 3)
            + loop_res * (1 / 3)
        )

        if self.bias:
            out = out + self.bias

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed):
        if self.opn == "corr":
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.opn == "sub":
            trans_embed = ent_embed - rel_embed
        elif self.opn == "mult":
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, "w_{}".format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, inputs):
        return inputs

    def compute_norm(self, edge_index, num_nodes):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        # Summing number of weights of the edges [Computing out-degree] [Should be equal to in-degree (undireted graph)]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float("inf")] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):
        return "{}({}, {}, num_rels={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels
        )


class CompGCNConv(MessagePassing):
    r"""
    CompGCN convolutional layer
    Args:
        in_channels (int): Input dimension
        out_channels (int): Output dimension
        num_rels (int): Number of relations
        act (function, optional): Activation function
        gcn_drop (float, optional): Dropout ratio
        opn (string, optional): Composition operation
        bias (bool, optional): use bias
        **kwargs (optional): Additional arguments of class `antmmf.modules.MessagePassing`
    Return:
        torch.FloatTensor: node embedding with shape (num_nodes, out_channels)
        torch.FloatTensor: rel embedding with shape (num_rel, out_channels)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_rels,
        act=lambda x: x,
        gcn_drop=0.5,
        opn="corr",
        bias=False,
        **kwargs
    ):
        super(self.__class__, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.opn = opn
        self.bias = bias

        def get_param(shape):
            param = Parameter(Tensor(*shape))
            xavier_normal(param)
            return param

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))

        self.drop = Dropout(gcn_drop)
        self.bn = BatchNorm1d(out_channels)

        if bias:
            self.register_parameter("bias", Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_nodes = x.size(0)

        self.in_index, self.out_index = (
            edge_index[:, :num_edges],
            edge_index[:, num_edges:],
        )
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        self.loop_index = torch.stack(
            [torch.arange(num_nodes), torch.arange(num_nodes)]
        )
        self.loop_type = torch.full(
            (num_nodes,), rel_embed.size(0) - 1, dtype=torch.long
        )

        self.in_norm = self.compute_norm(self.in_index, num_nodes)
        self.out_norm = self.compute_norm(self.out_index, num_nodes)

        in_res = self.propagate(
            self.in_index,
            x=x,
            edge_type=self.in_type,
            rel_embed=rel_embed,
            edge_norm=self.in_norm,
            mode="in",
        )
        loop_res = self.propagate(
            self.loop_index,
            x=x,
            edge_type=self.loop_type,
            rel_embed=rel_embed,
            edge_norm=None,
            mode="loop",
        )
        out_res = self.propagate(
            self.out_index,
            x=x,
            edge_type=self.out_type,
            rel_embed=rel_embed,
            edge_norm=self.out_norm,
            mode="out",
        )
        out = (
            self.drop(in_res) * (1 / 3)
            + self.drop(out_res) * (1 / 3)
            + loop_res * (1 / 3)
        )

        if self.bias:
            out = out + self.bias
        out = self.bn(out)

        return (
            self.act(out),
            torch.matmul(rel_embed, self.w_rel)[:-1],
        )  # Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):
        if self.opn == "corr":
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.opn == "sub":
            trans_embed = ent_embed - rel_embed
        elif self.opn == "mult":
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, "w_{}".format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, inputs):
        return inputs

    def compute_norm(self, edge_index, num_nodes):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(
            edge_weight, row, dim=0, dim_size=num_nodes
        )  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float("inf")] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):
        return "{}({}, {}, num_rels={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels
        )


class CompGCNBase(torch.nn.Module):
    def __init__(
        self,
        edge_index,
        edge_type,
        feature_dim,
        hidden_dim,
        num_nodes,
        num_rels,
        score_func,
        num_bases,
        gcn_layer,
        embed_dim,
    ):
        super().__init__()

        self.act = torch.tanh

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.gcn_layer = gcn_layer
        self.score_func = score_func

        if gcn_layer == 2:
            assert embed_dim is not None

        def get_param(shape):
            param = Parameter(Tensor(*shape))
            xavier_normal(param)
            return param

        self.init_embed = get_param((num_nodes, feature_dim))

        if num_bases > 0:
            self.init_rel = get_param((num_bases, feature_dim))
        else:
            if score_func == "transe":
                self.init_rel = get_param((num_rels, feature_dim))
            else:
                self.init_rel = get_param((num_rels * 2, feature_dim))

        if num_bases > 0:
            self.conv1 = CompGCNConvBasis(
                feature_dim, hidden_dim, num_rels, num_bases, act=self.act
            )
            self.conv2 = (
                CompGCNConv(hidden_dim, embed_dim, num_rels, act=self.act)
                if self.gcn_layer == 2
                else None
            )
        else:
            self.conv1 = CompGCNConv(feature_dim, hidden_dim, num_rels, act=self.act)
            self.conv2 = (
                CompGCNConv(hidden_dim, embed_dim, num_rels, act=self.act)
                if self.gcn_layer == 2
                else None
            )

        self.register_parameter("bias", Parameter(torch.zeros(num_nodes)))

    def forward_base(self, sub, rel, drop1, drop2):
        r = (
            self.init_rel
            if self.score_func != "transe"
            else torch.cat([self.init_rel, -self.init_rel], dim=0)
        )
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x = drop1(x)
        x, r = (
            self.conv2(x, self.edge_index, self.edge_type, rel_embed=r)
            if self.gcn_layer == 2
            else (x, r)
        )
        x = drop2(x) if self.gcn_layer == 2 else x

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
    def __init__(
        self,
        edge_index,
        edge_type,
        feature_dim,
        hidden_dim,
        num_nodes,
        num_rels,
        num_bases=-1,
        gcn_layer=1,
        embed_dim=None,
        hid_drop=0.3,
        gamma=40.0,
    ):
        super(self.__class__, self).__init__(
            edge_index,
            edge_type,
            feature_dim,
            hidden_dim,
            num_nodes,
            num_rels,
            "transe",
            num_bases,
            gcn_layer,
            embed_dim,
        )
        self.gamma = gamma
        self.drop = Dropout(hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return {"logits": score}


class CompGCN_DistMult(CompGCNBase):
    def __init__(
        self,
        edge_index,
        edge_type,
        feature_dim,
        hidden_dim,
        num_nodes,
        num_rels,
        num_bases=-1,
        gcn_layer=1,
        embed_dim=None,
        hid_drop=0.3,
    ):
        super(self.__class__, self).__init__(
            edge_index,
            edge_type,
            feature_dim,
            hidden_dim,
            num_nodes,
            num_rels,
            "distmult",
            num_bases,
            gcn_layer,
            embed_dim,
        )
        self.drop = Dropout(hid_drop)

    def forward(self, sub, rel):

        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return {"logits": score}


class CompGCN_ConvE(CompGCNBase):
    def __init__(
        self,
        edge_index,
        edge_type,
        feature_dim,
        hidden_dim,
        num_nodes,
        num_rels,
        embed_dim,
        k_w,
        k_h,
        num_bases=-1,
        gcn_layer=1,
        num_filt=200,
        hid_drop=0.3,
        hid_drop2=0.3,
        feat_drop=0.3,
        ker_sz=7,
        bias=False,
    ):
        super(self.__class__, self).__init__(
            edge_index,
            edge_type,
            feature_dim,
            hidden_dim,
            num_nodes,
            num_rels,
            "conve",
            num_bases,
            gcn_layer,
            embed_dim,
        )
        self.embed_dim = embed_dim
        self.k_w = k_w
        self.k_h = k_h

        self.bn0 = BatchNorm2d(1)
        self.bn1 = BatchNorm2d(num_filt)
        self.bn2 = BatchNorm1d(embed_dim)

        self.hidden_drop = Dropout(hid_drop)
        self.hidden_drop2 = Dropout(hid_drop2)
        self.feature_drop = Dropout(feat_drop)
        self.m_conv1 = Conv2d(
            1,
            out_channels=num_filt,
            kernel_size=(ker_sz, ker_sz),
            stride=1,
            padding=0,
            bias=bias,
        )

        flat_sz_h = int(2 * k_w) - ker_sz + 1
        flat_sz_w = k_h - ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * num_filt
        self.fc = Linear(self.flat_sz, embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(
            sub, rel, self.hidden_drop, self.feature_drop
        )
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return {"logits": score}
