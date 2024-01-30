# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import numpy as np
from typing import List
from deprecated import deprecated
from antmmf.common.registry import registry
from antmmf.common.constants import TEXT_MODALITY
from antmmf.modules.metrics.base_metric import BaseMetric


def _compute_recall_from_affine_closeness(text_embd, video_embd):
    """
    assume that the distance between text and video in the same pair is smaller than
    they each compared with other modality in other pairs
    """
    x = np.dot(text_embd, video_embd.T)
    # [[2.3000002 2.25      2.28      2.27     ]
    # [2.25      2.3000002 2.21      2.24     ]
    # [2.28      2.21      2.3000002 2.27     ]
    # [2.27      2.24      2.27      2.3      ]] w size (4, 4)
    if isinstance(x, np.float32):
        x = np.array([[x]])
    sx = np.sort(-x, axis=1)
    # sx [[-2.3000002 -2.28      -2.27      -2.25     ]
    #    [-2.3000002 -2.25      -2.24      -2.21     ]
    #    [-2.3000002 -2.28      -2.27      -2.21     ]
    #    [-2.3       -2.27      -2.27      -2.24     ]]
    d = np.diag(-x)
    #    d [-2.3000002 -2.3000002 -2.3000002 -2.3      ]
    d = d[:, np.newaxis]
    ind = sx - d
    #   ind [[0.         0.02000022 0.03000021 0.05000019]
    #        [0.         0.05000019 0.06000018 0.09000015]
    #        [0.         0.02000022 0.03000021 0.09000015]
    #        [0.         0.02999997 0.02999997 0.05999994]]
    ind = np.where(ind == 0)
    #   indp (array([0, 1, 2, 3]), array([0, 0, 0, 0]))
    ind = ind[1]
    #   indone [0 0 0 0]
    return ind


def _organize_for_two_modalities(model_output, modalities=List):
    text_embd = model_output[modalities[1]]
    video_embd = model_output[modalities[0]]
    text_embd = text_embd.detach().cpu().numpy()
    video_embd = video_embd.view(text_embd.shape[0], -1, text_embd.shape[1])
    video_embd = video_embd.mean(dim=1)
    video_embd = video_embd.detach().cpu().numpy()
    return video_embd, text_embd


@registry.register_metric("mm_retrieval_recall")
class MMRetrievalRecall(BaseMetric):
    """
    calculate similarity-based bi-modal retrieval performance
    This is used in some of the following senarios:
    e.g. the output of the last I3D (or S3D) model that
        maps the video to a space, and text is also mapped to this space.
        This measures the closeness betweent the mappings of these two modalities
    """

    def __init__(self, modalities, topk, *args, **kwargs):
        super().__init__(kwargs.get("name", "mm_retrieval_recall"))
        self._modalities = modalities
        assert len(self._modalities) == 2, "current support two modalities only"
        assert (
            self._modalities[1] == TEXT_MODALITY
        ), "text has to be the second in the modalities"
        self._topk = topk

    def _cal_recall(self, ind):
        epsilon = 1e-10
        if self._topk == "median-rank":
            return np.median(ind) + 1
        else:
            return float(np.sum(ind < int(self._topk)) / (len(ind) + epsilon))

    def reset(self):
        self._ind = np.array([])

    def collect(self, sample_list, model_output, *args, **kwargs):
        """
        Args:
            model_output (Dict): Dict returned by model, that contains two modalities
        Returns:
            torch.FloatTensor: Recall@1

        """

        video, text = _organize_for_two_modalities(model_output, self._modalities)
        ind = _compute_recall_from_affine_closeness(text, video)
        self._ind = np.concatenate((self._ind, ind))

    def calculate(self, sample_list, model_output, *args, **kwargs):
        video, text = _organize_for_two_modalities(model_output, self._modalities)
        ind = _compute_recall_from_affine_closeness(text, video)
        score = self._cal_recall(ind)
        return torch.tensor(score, dtype=torch.float64)

    def summarize(self, *args, **kwargs):
        score = self._cal_recall(self._ind)
        return torch.tensor(score, dtype=torch.float64)


@registry.register_metric("mm_retrieval_recall@1")
@deprecated(
    reason="mm_retrieval_recall@1 is deprecated, you can use mm_retrieval_recall by changing the type from "
    "`mm_retrieval_recall@1` to `mm_retrieval_recall` and changing the `topk` in params to `1`",
    version="1.3.7",
    action="default",
)
class MMRetrievalRecallAt1(MMRetrievalRecall):
    """
    calculate similarity-based bi-modal retrieval performance, measured in recall rate at top 1
    """

    def __init__(self, modalities, *args, **kwargs):
        super().__init__(
            modalities, topk=1, name=kwargs.get("name", "mm_retrieval_recall@1")
        )


@registry.register_metric("mm_retrieval_recall@5")
@deprecated(
    reason="mm_retrieval_recall@5 is deprecated, you can use mm_retrieval_recall by changing the type from "
    "`mm_retrieval_recall@5` to `mm_retrieval_recall` and changing the `topk` in params to `5`",
    version="1.3.7",
    action="default",
)
class MMRetrievalRecallAt5(MMRetrievalRecall):
    """
    calculate similarity-based bi-modal retrieval performance, measured in recall rate at top 1
    """

    def __init__(self, modalities, *args, **kwargs):
        super().__init__(
            modalities, topk=5, name=kwargs.get("name", "mm_retrieval_recall@5")
        )


@registry.register_metric("mm_retrieval_recall@10")
@deprecated(
    reason="mm_retrieval_recall@10 is deprecated, you can use mm_retrieval_recall by changing the type from "
    "`mm_retrieval_recall@10` to `mm_retrieval_recall` and changing the `topk` in params to `10`",
    version="1.3.7",
    action="default",
)
class MMRetrievalRecallAt10(MMRetrievalRecall):
    """
    calculate similarity-based bi-modal retrieval performance, measured in recall rate at top 1
    """

    def __init__(self, modalities, *args, **kwargs):
        super().__init__(
            modalities, topk=10, name=kwargs.get("name", "mm_retrieval_recall@10")
        )


@registry.register_metric("mm_retrieval_median_rank")
@deprecated(
    reason="mm_retrieval_median_rank is deprecated, you can use mm_retrieval_recall by changing the type from "
    "`mm_retrieval_median_rank` to `mm_retrieval_recall` and changing the `topk` in params to `median-rank`",
    version="1.3.7",
    action="default",
)
class MMRetrievalMedianRank(MMRetrievalRecall):
    """
    calculate similarity-based bi-modal retrieval performance, measured in recall rate at top 1
    """

    def __init__(self, modalities, *args, **kwargs):
        super().__init__(
            modalities,
            topk="median-rank",
            name=kwargs.get("name", "mm_retrieval_median_rank"),
        )
