# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import Optional
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from antmmf.modules.transformers.heads.base import PredictableHead
from antmmf.common import Configuration, configurable


class MLM(PredictableHead):
    """Pretraining heads with bounded losses & metrics:
    https://github.com/facebookresearch/mmf/blob/master/mmf/models/transformers/heads/mlm.py
    """

    @configurable
    def __init__(
        self,
        vocab_size: int = 30522,
        in_dim: int = 768,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
        ignore_index: int = -1,
        loss_name: str = "masked_lm_loss",
        metric_name: str = "masked_lm_acc",
    ):
        super().__init__()

        self.cls = BertOnlyMLMHead(Configuration(locals()))

        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.loss_name = loss_name
        self.metric_name = metric_name

        if in_dim != hidden_size:
            self.linear = nn.Linear(in_dim, hidden_size)
        else:
            self.linear = nn.Identity()

        # Loss
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def tie_weights(self, module: Optional[torch.nn.Module] = None):
        self.cls.predictions.decoder.weight = module.weight

    def forward_head(self, encoder_output=None, decoder_output=None):
        """
        :param encoder_output: bsz, seq, hidden
        :return: bsz, seq, vocab_size
        """
        return self.cls(self.linear(encoder_output))

    def convert_id2text(self, tokenizer, generation_ids):
        # caption post-process, refer to:
        # https://github.com/microsoft/UniVL/blob/1a40788874460e1f17691e749af7951d0e872523/main_task_caption.py#L553-L563
        sentence = []
        for generate_ids in generation_ids:
            decode_text = tokenizer.convert_ids_to_tokens(generate_ids)
            if tokenizer.sep_token in decode_text:
                SEP_index = decode_text.index(tokenizer.sep_token)
                decode_text = decode_text[:SEP_index]
            if tokenizer.pad_token in decode_text:
                PAD_index = decode_text.index(tokenizer.pad_token)
                decode_text = decode_text[:PAD_index]
            decode_text = " ".join(decode_text)
            decode_text = decode_text.replace(" ##", "").strip("##").strip()
            sentence.append(decode_text)
        return sentence

    def get_loss_metric(self, predictions, targets: torch.Tensor):
        """
        :param predictions:
        :param targets: bsz, seq_length
        :return:
        """
        output_dict = dict()
        output_dict["logits"] = predictions

        masked_lm_logits = predictions.contiguous().view(-1, self.vocab_size)
        masked_lm_targets = targets.contiguous().view(-1)

        masked_lm_loss = self.ce_loss(masked_lm_logits, masked_lm_targets)

        output_dict["losses"] = {}
        output_dict["losses"][self.loss_name] = masked_lm_loss

        # MLM acc
        with torch.no_grad():
            output_dict["metrics"] = {}
            if masked_lm_logits.size(0) > 0:
                pred_res = (masked_lm_logits.argmax(-1) == masked_lm_targets)[
                    torch.where(masked_lm_targets != self.ignore_index)
                ]
                mlm_acc = pred_res.sum() / (pred_res.size(0) + 1e-6)
            else:
                mlm_acc = torch.tensor(1.0).to(masked_lm_logits.device)
            output_dict["metrics"][self.metric_name] = mlm_acc

        return output_dict
