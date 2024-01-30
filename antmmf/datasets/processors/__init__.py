# Copyright (c) 2023 Ant Group and its affiliates.
from .image_processors import (
    BBoxProcessor,
    TorchvisionTransforms,
    GrayScaleTo3Channels,
    NormImageProcessor,  # this one is not optimal,
    DetrProcessor,
)
from .mm_processors import VQAAnswerProcessor
from .processors import CopyProcessor
from .text_processors import (
    VocabProcessor,
    GloVeProcessor,
    FastTextProcessor,
    VQAAnswerProcessor,
    MultiHotAnswerFromVocabProcessor,
    SoftCopyAnswerProcessor,
    SimpleWordProcessor,
    SimpleSentenceProcessor,
    CaptionProcessor,
    PhocProcessor,
    BertTokenizerProcessor,
    MaskedTokenProcessor,
)
from .video_processors import FMpegProcessor
