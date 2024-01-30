# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn

from .multimodal_encoder import MultimodalBertEncoder
from .utils import ModelForPretrainingMixin


class MultimodalBertForPretraining(nn.Module, ModelForPretrainingMixin):
    def __init__(self, config):
        super(MultimodalBertForPretraining, self).__init__()
        self.config = config
        self.enc = MultimodalBertEncoder(config)
        from transformers.models.bert.modeling_bert import BertPreTrainingHeads

        self.clf = BertPreTrainingHeads(config)
        self.init_weights(self.clf)
        self.clf.predictions.decoder.weight = (
            self.enc.txt_embeddings.word_embeddings.weight
        )

    def forward(self, txt, mask, segment, img, cls_id, sep_id, img_mask=None):
        pooled_output, rep_info, sequence_output, image_seq_length = self.enc(
            txt, mask, segment, img, cls_id, sep_id, img_mask
        )
        prediction_scores, seq_relationship_score = self.clf(
            sequence_output, pooled_output
        )
        ret_dict = {
            "prediction_scores": prediction_scores,
            "seq_relationship_score": seq_relationship_score,
            "image_seq_length": image_seq_length,
        }
        ret_dict.update(rep_info)
        return ret_dict

    def get_adv_parameters(self):
        return self.enc.get_adv_parameters()
