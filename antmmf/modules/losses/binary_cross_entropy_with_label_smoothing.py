# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from typing import Union
from antmmf.common.registry import registry


def get_labels_tensor_from_indices(
    batch_size: int,
    num_embeddings: int,
    entity_ids: torch.Tensor,
    dtype: torch.dtype = torch.float,
    label_smoothing: Union[int, float] = None,
) -> torch.Tensor:
    # create a tensor of 0-1 that is shape (batch_size, num_embeddings)
    # entity_ids = (batch_size, max_num_positive_entities), type long
    # it contains the list of 1 label indices in (0, num_embeddings-1)
    labels = entity_ids.new_zeros(batch_size, num_embeddings, dtype=dtype)
    labels.scatter_add_(1, entity_ids, torch.ones_like(entity_ids, dtype=dtype))

    # remove the masking
    labels[:, 0] = 0.0

    # label smoothing
    if label_smoothing is not None:
        total_labels = labels.mean(dim=1, keepdim=True)
        labels = (1.0 - label_smoothing) * labels + label_smoothing * total_labels
    return labels


@registry.register_loss("bce_ls")
class BinaryCrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = torch.nn.BCELoss()

    def forward(self, sample_list, model_output, *args, **kwargs):
        num_entities = model_output["logits"].shape[1]
        # entity = (batch_size, 1) with e1 ids
        # entity2 = (batch_size, max_correct_entities)
        entity = sample_list.e1
        entity2 = sample_list["targets"]

        # run the prediction
        # (batch_size, num_entities)
        predicted_e2 = model_output["logits"]

        # create the array with 0-1 values with gold entity2
        batch_size = entity.shape[0]
        labels = get_labels_tensor_from_indices(
            batch_size, num_entities, entity2, label_smoothing=0.1
        )

        loss = self.loss(predicted_e2, labels)
        return loss
