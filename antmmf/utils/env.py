# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import multiprocessing
import os
import platform
import random
import sys
from datetime import datetime

import numpy as np
import torch
from antmmf.common import constants


def set_seed(seed):
    if seed is None:
        return
    if seed == -1:
        # From detectron2, automatic seed
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_env():
    """
    Set online environ variables, those envs will allow loading locally downloaded pre-trained models.
    we can indicate short-name instead of path for transformer & torchvision models in config.yml if those
    variables are correctly set.

    Currently used env variables:

    `TORCH_HOME`: torchvision pretrained models dir
    `PYTORCH_TRANSFORMERS_CACHE`: pratrained transformers and vocabs dir
    `USER_MODEL_HOME`: root dir placing online antmmf models, usually used in collie env

    Returns:

    """
    os.environ[constants.TORCHVISION_PRETAINED_MODELS_ENV_VAR] = "pretrained_models"
    os.environ[constants.BERT_PRETRAINED_MODELS_ENV_VAR] = "pretrained_models"
    os.environ[constants.USER_MODEL_HOME] = "models"


def setup_compatibility():
    """
    Python 3.8 changes the default mode of multiprocessing on MacOS to spawn instead of fork.
    This requires all parameters passed to Sanic worker processed to be picklable, and may cause some errors.
    """
    if (
        platform.system() == "Darwin"
        and sys.version_info.major == 3
        and sys.version_info.minor >= 8
    ):
        try:
            multiprocessing.set_start_method("fork")
        except RuntimeError:
            pass
