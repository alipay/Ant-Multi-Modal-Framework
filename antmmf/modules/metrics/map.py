# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import os
import torch
import contextlib

from antmmf.utils.general import get_absolute_path
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric

from .evaluators import CocoEvaluator, format_coco_detection_result


@registry.register_metric("bbox_ap")
class MAP(BaseMetric):
    """Metric for calculating map on whole dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(name=kwargs.get("name", "bbox_ap"))
        from pycocotools.coco import COCO

        annotation_file = kwargs["coco_gt"]
        if isinstance(annotation_file, str):
            annotation_file = [annotation_file]
        self.anno_path = get_absolute_path(os.path.join(*annotation_file))

        self.coco_evaluator = CocoEvaluator(COCO(self.anno_path), ["bbox"])
        self.label_mapping = kwargs.get("label_mapping", None)
        self._summarized = None

    def reset(self):
        self.coco_evaluator.reset()
        self._summarized = None

    def collect(self, sample_list, model_output, *args, **kwargs):
        """
        Args:
            sample_list: must have 'image_id' key
            model_output (Dict): Dict returned by model, that contains key 'bbox_output',
                                 bbox_output should be Nx6 np.Array: with each box represented as
                                 (x1, y1, x2, y2, conf, cls), all coordinates are absolutely measured.
        Returns:
        """
        predictions = format_coco_detection_result(
            sample_list["image_id"],
            model_output["bbox_output"],
            label_mapping=self.label_mapping,
        )

        # convert from evalai_format
        # to coco eval format:{image_id: {boxes: Nx4, scores: N, labels: N}}
        ret_dict = dict()
        for det in predictions:
            image_id = det["image_id"]
            if len(det["labels"]) == 0:
                ret_dict[image_id] = {}
            else:
                ret_dict[image_id] = {
                    "boxes": torch.Tensor(det["boxes"]),
                    "labels": torch.Tensor(det["labels"]),
                    "scores": torch.Tensor(det["scores"]),
                }
        self.coco_evaluator.update(ret_dict)

    def summarize(self, *args, **kwargs):
        if self._summarized is None:
            # summarize may called many times, but only synchronize once
            self.coco_evaluator.synchronize_between_processes()

            # only run summarization by main process
            # suppress pycocotools prints
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.coco_evaluator.accumulate()
                    self.coco_evaluator.summarize()
            self._summarized = self.coco_evaluator.coco_eval["bbox"].stats.tolist()

        return {
            "map": self._summarized[0],
            "map@0.5": self._summarized[1],
            "map@small": self._summarized[3],
            "map@medium": self._summarized[4],
            "map@large": self._summarized[5],
        }
