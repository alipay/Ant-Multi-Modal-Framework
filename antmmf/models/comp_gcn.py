# Copyright (c) 2023 Ant Group and its affiliates.

from antmmf.models.base_model import BaseModel
from antmmf.common.registry import registry
from antmmf.modules.graph import CompGCN_TransE, CompGCN_DistMult, CompGCN_ConvE


@registry.register_model("comp_gcn")
class CompGCN(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        edge_index, edge_type = self._construct_adj()
        if self.config.score_func == "transe":
            self.model = CompGCN_TransE(
                edge_index,
                edge_type,
                self.config.params.feature_dim,
                self.config.params.hidden_dim,
                self.config.params.num_nodes,
                self.config.params.num_rels,
                self.config.params.num_bases,
                self.config.params.gcn_layer,
                self.config.params.embed_dim,
                self.config.params.hid_drop,
                self.config.params.gamma,
            )
        elif self.config.score_func == "distmult":
            self.model = CompGCN_DistMult(
                edge_index,
                edge_type,
                self.config.params.feature_dim,
                self.config.params.hidden_dim,
                self.config.params.num_nodes,
                self.config.params.num_rels,
                self.config.params.num_bases,
                self.config.params.gcn_layer,
                self.config.params.embed_dim,
                self.config.params.hid_drop,
            )
        elif self.config.score_func == "conve":
            self.model = CompGCN_ConvE(
                edge_index,
                edge_type,
                self.config.params.feature_dim,
                self.config.params.hidden_dim,
                self.config.params.num_nodes,
                self.config.params.num_rels,
                self.config.params.embed_dim,
                self.config.params.k_w,
                self.config.params.k_h,
                self.config.params.num_bases,
                self.config.params.gcn_layer,
                self.config.params.num_filt,
                self.config.params.hid_drop,
                self.config.params.hid_drop2,
                self.config.params.feat_drop,
                self.config.params.ker_sz,
                self.config.params.bias,
            )
        else:
            raise Exception("unknown score_func")

    def _construct_adj():
        pass

    def forward(self, sample_list):
        sub = sample_list.sub
        rel = sample_list.rel

        output = self.model(sub, rel)
        if "targets" in sample_list:
            if self.training:
                losses = self.losses(sample_list, output)
                output["losses"] = losses
            else:
                metrics = self.metrics(sample_list, output)
                output["metrics"] = metrics

        return output
