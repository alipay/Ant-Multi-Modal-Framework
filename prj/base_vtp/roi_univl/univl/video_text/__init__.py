# coding: utf-8
# Copyright (c) Ant Group. All rights reserved.

from .cls_builder import UnivlDatasetBuilder as ClsUnivlDatasetBuilder
from .mc_qa_builder import UnivlDatasetBuilder as McQAUnivlDatasetBuilder
from .ret_builder import UnivlDatasetBuilder as RetUnivlDatasetBuilder

__all__ = [
    "ClsUnivlDatasetBuilder",
    "McQAUnivlDatasetBuilder",
    "RetUnivlDatasetBuilder",
]
