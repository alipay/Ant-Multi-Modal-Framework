# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn
from torch.nn import functional as F

from antmmf.common import Configuration
from antmmf.modules.decoders.hierarchical_classifier import HierarchicalClassifier
from antmmf.modules.layers import NormLinear
from .multimodal_encoder import MultimodalBertEncoder


class MultimodalBertClf(nn.Module):
    def __init__(self, config: Configuration, modal_encoder=MultimodalBertEncoder):
        super(MultimodalBertClf, self).__init__()
        self.config = config
        self.enc = modal_encoder(config)
        self.classifier_type = config.get("classifier_type")
        # for compatibility
        hidden_size = getattr(config, "hidden_size", getattr(config, "hidden_sz", None))

        if self.classifier_type in [None, "mlp_classifier"]:
            self.clf = nn.Linear(hidden_size, config.num_labels)
        elif self.classifier_type == "retrieval_classifier":
            embedding_dim = getattr(config, "embedding_dim", hidden_size)
            self.clf = nn.Sequential(
                nn.Linear(hidden_size, embedding_dim),
                nn.GELU(),
                nn.BatchNorm1d(embedding_dim),
                NormLinear(embedding_dim, config.num_labels),
            )
        else:
            assert self.classifier_type == "hier_classifier"
            self.clf = HierarchicalClassifier(
                hidden_size, config.hier_label_schema, config
            )

    def forward(self, txt, mask, segment, img, cls_id, sep_id, img_mask=None):
        x, rep_info, _, _ = self.enc(txt, mask, segment, img, cls_id, sep_id, img_mask)
        if self.classifier_type == "hier_classifier":
            ret = self.clf(x)
        else:
            if self.classifier_type == "retrieval_classifier":
                out_feat, logits = self.clf(x)
                prob = F.softmax(logits, dim=-1)
            else:  # mlp_classifier
                out_feat = rep_info.pop("bert_embedding", None)
                logits = self.clf(x)
                prob = F.softmax(logits, dim=-1)
            ret = dict(rep_info)
            ret["out_feat"] = out_feat
            ret["logits"] = logits
            ret["prob"] = prob
        return ret

    def get_adv_parameters(self):
        return self.enc.get_adv_parameters()
