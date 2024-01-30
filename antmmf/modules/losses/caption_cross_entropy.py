# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from antmmf.common.registry import registry


@registry.register_loss("caption_cross_entropy")
class CaptionCrossEntropyLoss(nn.Module):
    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["logits"]
        targets = sample_list["targets"]

        # If no captions(test dataset) then assume decode length to be uniform
        if hasattr(sample_list, "caption_len"):
            caption_lengths, _ = sample_list.caption_len.sort(dim=0, descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
        else:
            decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= "1.1":
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = F.cross_entropy(scores, targets)

        return loss
