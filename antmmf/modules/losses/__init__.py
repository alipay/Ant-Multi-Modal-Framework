# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from .losses import Losses, AntMMFLoss
from .binary_cross_entropy import BinaryCrossEntropyLoss
from .binary_cross_entropy_with_logits import BinaryCrossEntropyWithLogits
from .binary_cross_entropy_with_label_smoothing import (
    BinaryCrossEntropyLossWithLabelSmoothing,
)
from .caption_cross_entropy import CaptionCrossEntropyLoss
from .nll_loss import NLLLoss
from .multi_loss import MultiLoss
from .attention_supervision_loss import AttentionSupervisionLoss
from .weighted_softmax_loss import WeightedSoftmaxLoss
from .softmax_focal_loss import SoftmaxFocalLoss
from .softmax_kl_div_loss import SoftmaxKlDivLoss
from .wrong_loss import WrongLoss
from .combined_loss import CombinedLoss
from .m4c_decoding_bce_with_mask_loss import M4CDecodingBCEWithMaskLoss
from .cross_entropy_loss import CrossEntropyLoss
from .eet_loss import EETLoss
from .cos_arc_loss import CosArcLoss
from .nce_loss import NCELoss
from .mil_nce_loss import MILNCELoss
from .mil_margin_contrastive_loss import MILMarginContrastiveLoss
from .mse_loss import MSELoss
from .cos_ams_softmax_loss import CosAmsSoftmaxLoss
from .hierarchical_softmax_loss import HierarchicalSoftmaxLoss
from .hierarchical_multilabel_loss import HierarchicalMultilabelLoss
from .asymmetric_loss_optimized import AsymmetricLossOptimized
from .knowledge_distill_loss import KnowledgeDistillLoss
from .ordinal_loss import OrdinalLoss
from .multi_label_category_cross_entropy_loss import MultiLabelCategoryCrossEntropyLoss
from .label_smoothing_cross_entropy import LabelSmoothingCrossEntropy
from .pairwise_loss import PairwiseLoss
from .kg_margin_contrastive_loss import KGMarginContrastiveLoss
from .info_nce_loss import DInfoNCELoss
