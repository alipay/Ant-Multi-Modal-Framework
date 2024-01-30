# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.registry import registry


def normalize(x, axis=-1):
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def hard_sample_mining(feats, labels, is_normalize=False):
    if is_normalize:
        feats = normalize(feats)
    dist_mat = euclidean_dist(feats, feats)
    n = dist_mat.size(0)

    is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
    is_neg = labels.expand(n, n).ne(labels.expand(n, n).t())

    dist_ap, dist_an, dist_pn, dist_nn = [], [], [], []
    for i in range(n):
        d_ap = dist_mat[i][is_pos[i]].max()
        d_an = dist_mat[i][is_neg[i]].min()
        ind = torch.arange(0, len(dist_mat[i]))
        ind_p = torch.eq(dist_mat[i], d_ap)
        ind_p = ind[ind_p][0]
        ind_n = torch.eq(dist_mat[i], d_an)
        ind_n = ind[ind_n][0]
        d_pn = dist_mat[ind_p][ind_n]
        d_nn = dist_mat[ind_n][is_neg[ind_n]].min()

        dist_ap.append(d_ap.unsqueeze(0))
        dist_an.append(d_an.unsqueeze(0))
        dist_pn.append(d_pn.unsqueeze(0))
        dist_nn.append(d_nn.unsqueeze(0))
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    dist_pn = torch.cat(dist_pn)
    dist_nn = torch.cat(dist_nn)

    return dist_ap, dist_an, dist_pn, dist_nn


@registry.register_loss("eet")
class EETLoss(nn.Module):
    """
    EET: Equidistant and Equidistributed Triplet-based Loss, an improved triplet metric loss,
         can really pull matched pairs closer
    @param margin: distance difference  between matched pairs and mismatched pairs, 0.3|0.5
    @param is_normalize: whether to normalize l2-norm of the feature
    """

    def __init__(self, margin=0.3, is_normalize=False):
        super().__init__()
        self.margin = margin
        self.is_normalize = is_normalize

    def forward(self, sample_list, model_output, *args, **kwargs):
        feats = model_output["out_feat"]
        labels = sample_list.targets
        dist_ap, dist_an, dist_pn, _ = hard_sample_mining(
            feats, labels, self.is_normalize
        )
        device = dist_ap.device
        margin_loss = torch.mean(
            torch.max(
                torch.zeros(dist_ap.size(0)).to(device=device),
                dist_ap + self.margin - dist_an,
            )
        ) + torch.mean(
            torch.max(
                torch.zeros(dist_ap.size(0)).to(device=device),
                dist_ap + self.margin - dist_pn,
            )
        )
        ict_loss = torch.mean(torch.abs(dist_pn - dist_an))
        return margin_loss + ict_loss
