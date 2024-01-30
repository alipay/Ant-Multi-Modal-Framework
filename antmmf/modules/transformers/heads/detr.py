# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import List, Dict

import torch
from torch import nn
from torch.nn import functional as F

from antmmf.common import configurable
from antmmf.modules.layers import MLP
from antmmf.modules.matcher import HungarianMatcher
from antmmf.modules.transformers.heads.base import PredictableHead
from antmmf.modules.functional.set_criterion import SetCriterion


class DETR(PredictableHead):
    """Head for DETR, process the output generated by Backbone.

    Reference: https://arxiv.org/abs/2005.12872

    TODO: enrich the document here.
    """

    @configurable
    def __init__(
        self,
        hidden_size: int = 768,  # transformer d_model
        num_classes: int = 80,  # number of object classes
        aux_loss: bool = True,  # True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        set_cost_class: float = 1.0,
        set_cost_bbox: float = 5.0,
        set_cost_giou: float = 2.0,
        dice_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        losses: List[str] = ("labels", "boxes", "cardinality"),
        eos_coef: float = 0.1,  # relative classification weight applied to the no-object category
        num_bbox_layers: int = 3,
        dec_layers: int = 3,
        loss_name: str = "detr_loss",
    ):
        super().__init__()
        self.aux_loss = aux_loss
        self.dec_layers = dec_layers
        self.loss_name = loss_name

        self.class_embed = nn.Linear(hidden_size, num_classes + 1)
        self.bbox_embed = MLP(
            hidden_size,
            4,
            num_layers=num_bbox_layers,
            batch_norm=False,
            dropout=0.0,
        )

        self.matcher = HungarianMatcher(
            cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou
        )
        weight_dict = {
            "loss_ce": dice_loss_coef,
            "loss_bbox": bbox_loss_coef,
            "loss_giou": giou_loss_coef,
        }

        if self.aux_loss:
            weight_dict.update(
                {
                    k + f"_{i}": v
                    for k, v in weight_dict.items()
                    for i in range(self.dec_layers - 1)
                }
            )

        self.set_loss = SetCriterion(
            num_classes, self.matcher, weight_dict, eos_coef, losses
        )

    def forward_head(self, encoder_output=None, decoder_output=None, **kwargs):
        """
        TODO: add document here.
        """
        assert decoder_output is not None
        outputs_class = self.class_embed(decoder_output)
        outputs_coord = self.bbox_embed(decoder_output).sigmoid()

        predictions = {}
        if decoder_output.ndim == 4:  # output decoder layers
            # last layer output
            predictions["pred_logits"] = outputs_class[-1]
            predictions["pred_boxes"] = outputs_coord[-1]
            if self.aux_loss:
                predictions["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
        else:  # only output last decode layer
            assert decoder_output.ndim == 3
            predictions["pred_logits"] = outputs_class
            predictions["pred_boxes"] = outputs_coord
        return predictions

    @torch.no_grad()
    def post_process(self, predictions: Dict, target_sizes: torch.Tensor):
        """Perform the computation
        Parameters:
            predictions: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = predictions["pred_logits"], predictions["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        cx, cy, w, h = out_bbox.unbind(-1)
        _w, _h = w / 2.0, h / 2.0
        boxes = torch.stack([cx - _w, cy - _h, cx + _w, cy + _h], dim=-1)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

    def get_loss_metric(self, predictions: Dict, targets: List):
        """
        :param targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target, not padded). containing the class labels, labels should be 1-based,
                           and num_classes should be set as max_label_id + 1.
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates. The target boxes
                           are expected in format (center_x, center_y, w, h), normalized by the image size.
        :return:
        """
        loss_dict = self.set_loss(predictions, targets)
        weight_dict = self.set_loss.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        output_dict = {
            "losses": {self.loss_name: losses},
        }
        with torch.no_grad():
            output_dict["metrics"] = {}
            for metric_key in [
                "loss_ce",
                "loss_bbox",
                "loss_giou",
                "class_error",
                "cardinality_error",
            ]:
                if metric_key in loss_dict:
                    output_dict["metrics"][metric_key] = (
                        loss_dict[metric_key].clone().detach()
                    )

        return output_dict
