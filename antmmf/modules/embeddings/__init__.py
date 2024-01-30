# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from .antmmf_embeddings import AntMMFEmbeddings
from .text_embedding import (
    TextEmbedding,
    PreExtractedEmbedding,
    AttentionTextEmbedding,
    BiLSTMTextEmbedding,
)
from .image_embedding import ImageEmbedding
from .bert_vision_linguistic_embeddings import BertVisionLinguisticEmbeddings
from .image_bert_embeddings import ImageBertEmbeddings
from .layout_lm_embeddings import LayoutLMEmbeddings
from .visual_layout_embeddings import VisualLayoutEmbeddings
from .detr_position_embedding_sine import DetrPositionEmbeddingSine
from .detr_position_embedding_learned import DetrPositionEmbeddingLearned
from .clip_visual_embedding import ClipVisualEmbedding
from .univl_layout_embedding import UnivlLayoutEmbedding

AntMMFEmbeddings.register(TextEmbedding)
AntMMFEmbeddings.register(PreExtractedEmbedding)
AntMMFEmbeddings.register(ImageEmbedding)
AntMMFEmbeddings.register(BertVisionLinguisticEmbeddings)
AntMMFEmbeddings.register(ImageBertEmbeddings)
AntMMFEmbeddings.register(LayoutLMEmbeddings)
AntMMFEmbeddings.register(VisualLayoutEmbeddings)
AntMMFEmbeddings.register(DetrPositionEmbeddingSine)
AntMMFEmbeddings.register(DetrPositionEmbeddingLearned)
AntMMFEmbeddings.register(ClipVisualEmbedding)
AntMMFEmbeddings.register(UnivlLayoutEmbedding)
