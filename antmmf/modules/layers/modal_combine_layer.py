# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn
from antmmf.common.registry import registry
from antmmf.modules.module_registry import ModuleRegistry
from .linear import Linear


class ModalCombineLayer(ModuleRegistry):
    def __init__(self, combine_type, img_feat_dim, txt_emb_dim, **kwargs):
        # compatible codes, and they will be removed in the future.
        type_mapping = {
            "non_linear_element_multiply": "NonLinearElementMultiply",
            "two_layer_element_multiply": "TwoLayerElementMultiply",
            "top_down_attention_lstm": "TopDownAttentionLSTM",
        }
        if combine_type in type_mapping:
            combine_type = type_mapping[combine_type]

        self.out_dim = self.module.out_dim
        super(ModalCombineLayer, self).__init__(
            combine_type, img_feat_dim, txt_emb_dim, **kwargs
        )


class MfbExpand(nn.Module):
    def __init__(self, img_feat_dim, txt_emb_dim, hidden_dim, dropout):
        super(MfbExpand, self).__init__()
        self.lc_image = nn.Linear(in_features=img_feat_dim, out_features=hidden_dim)
        self.lc_ques = nn.Linear(in_features=txt_emb_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, question_embed):
        image1 = self.lc_image(image_feat)
        ques1 = self.lc_ques(question_embed)
        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            ques1_expand = torch.unsqueeze(ques1, 1).expand(-1, num_location, -1)
        else:
            ques1_expand = ques1
        joint_feature = image1 * ques1_expand
        joint_feature = self.dropout(joint_feature)
        return joint_feature


@ModalCombineLayer.register()
class MFH(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(MFH, self).__init__()
        self.mfb_expand_list = nn.ModuleList()
        self.mfb_sqz_list = nn.ModuleList()
        self.relu = nn.ReLU()

        hidden_sizes = kwargs["hidden_sizes"]
        self.out_dim = int(sum(hidden_sizes) / kwargs["pool_size"])

        self.order = kwargs["order"]
        self.pool_size = kwargs["pool_size"]

        for i in range(self.order):
            mfb_exp_i = MfbExpand(
                img_feat_dim=image_feat_dim,
                txt_emb_dim=ques_emb_dim,
                hidden_dim=hidden_sizes[i],
                dropout=kwargs["dropout"],
            )
            self.mfb_expand_list.append(mfb_exp_i)
            self.mfb_sqz_list.append(self.mfb_squeeze)

    def forward(self, image_feat, question_embedding):
        feature_list = []
        prev_mfb_exp = 1

        for i in range(self.order):
            mfb_exp = self.mfb_expand_list[i]
            mfb_sqz = self.mfb_sqz_list[i]
            z_exp_i = mfb_exp(image_feat, question_embedding)
            if i > 0:
                z_exp_i = prev_mfb_exp * z_exp_i
            prev_mfb_exp = z_exp_i
            z = mfb_sqz(z_exp_i)
            feature_list.append(z)

        # append at last feature
        cat_dim = len(feature_list[0].size()) - 1
        feature = torch.cat(feature_list, dim=cat_dim)
        return feature

    def mfb_squeeze(self, joint_feature):
        # joint_feature dim: N x k x dim or N x dim

        orig_feature_size = len(joint_feature.size())

        if orig_feature_size == 2:
            joint_feature = torch.unsqueeze(joint_feature, dim=1)

        batch_size, num_loc, dim = joint_feature.size()

        if dim % self.pool_size != 0:
            exit(
                "the dim %d is not multiply of \
             pool_size %d"
                % (dim, self.pool_size)
            )

        joint_feature_reshape = joint_feature.view(
            batch_size, num_loc, int(dim / self.pool_size), self.pool_size
        )

        # N x 100 x 1000 x 1
        iatt_iq_sumpool = torch.sum(joint_feature_reshape, 3)

        iatt_iq_sqrt = torch.sqrt(self.relu(iatt_iq_sumpool)) - torch.sqrt(
            self.relu(-iatt_iq_sumpool)
        )

        iatt_iq_sqrt = iatt_iq_sqrt.view(batch_size, -1)  # N x 100000
        iatt_iq_l2 = F.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(batch_size, num_loc, int(dim / self.pool_size))

        if orig_feature_size == 2:
            iatt_iq_l2 = torch.squeeze(iatt_iq_l2, dim=1)

        return iatt_iq_l2


# need to handle two situations,
# first: image (N, K, i_dim), question (N, q_dim);
# second: image (N, i_dim), question (N, q_dim);
@ModalCombineLayer.register()
class NonLinearElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(NonLinearElementMultiply, self).__init__()
        self.fa_image = Linear(image_feat_dim, kwargs["hidden_dim"])
        self.fa_txt = Linear(ques_emb_dim, kwargs["hidden_dim"])

        context_dim = kwargs.get("context_dim", None)
        if context_dim is None:
            context_dim = ques_emb_dim

        self.fa_context = Linear(context_dim, kwargs["hidden_dim"])
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["hidden_dim"]

    def forward(self, image_feat, question_embedding, context_embedding=None):
        image_fa = F.relu_(self.fa_image(image_feat))
        question_fa = F.relu_(self.fa_txt(question_embedding))

        if len(image_feat.size()) == 3:
            question_fa_expand = question_fa.unsqueeze(1)
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand

        if context_embedding is not None:
            context_fa = F.relu_(self.fa_context(context_embedding))

            context_text_joint_feaure = context_fa * question_fa_expand
            joint_feature = torch.cat([joint_feature, context_text_joint_feaure], dim=1)

        joint_feature = self.dropout(joint_feature)

        return joint_feature


@ModalCombineLayer.register()
class TopDownAttentionLSTM(nn.Module):
    def __init__(self, image_feat_dim, embed_dim, **kwargs):
        super().__init__()
        self.fa_image = Linear(image_feat_dim, kwargs["attention_dim"])
        self.fa_hidden = Linear(kwargs["hidden_dim"], kwargs["attention_dim"])
        self.top_down_lstm = nn.LSTMCell(
            embed_dim + image_feat_dim + kwargs["hidden_dim"],
            kwargs["hidden_dim"],
            bias=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["attention_dim"]

    def forward(self, image_feat, embedding):
        image_feat_mean = image_feat.mean(1)

        # Get LSTM state
        state = registry.get("{}_lstm_state".format(image_feat.device))
        h1, c1 = state["td_hidden"]
        h2, c2 = state["lm_hidden"]

        h1, c1 = self.top_down_lstm(
            torch.cat([h2, image_feat_mean, embedding], dim=1), (h1, c1)
        )

        state["td_hidden"] = (h1, c1)

        image_fa = self.fa_image(image_feat)
        hidden_fa = self.fa_hidden(h1)

        joint_feature = self.relu(image_fa + hidden_fa.unsqueeze(1))
        joint_feature = self.dropout(joint_feature)

        return joint_feature


@ModalCombineLayer.register()
class TwoLayerElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(TwoLayerElementMultiply, self).__init__()

        self.fa_image1 = Linear(image_feat_dim, kwargs["hidden_dim"])
        self.fa_image2 = Linear(kwargs["hidden_dim"], kwargs["hidden_dim"])
        self.fa_txt1 = Linear(ques_emb_dim, kwargs["hidden_dim"])
        self.fa_txt2 = Linear(kwargs["hidden_dim"], kwargs["hidden_dim"])

        self.dropout = nn.Dropout(kwargs["dropout"])

        self.out_dim = kwargs["hidden_dim"]

    def forward(self, image_feat, question_embedding):
        image_fa = F.relu_(self.fa_image2(F.relu_(self.fa_image1(image_feat))))
        question_fa = F.relu_(self.fa_txt2(F.relu_(self.fa_txt1(question_embedding))))

        if len(image_feat.size()) == 3:
            num_location = image_feat.size(1)
            question_fa_expand = torch.unsqueeze(question_fa, 1).expand(
                -1, num_location, -1
            )
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)

        return joint_feature
