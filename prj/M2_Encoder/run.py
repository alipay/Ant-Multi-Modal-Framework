# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn4k.consts import NN_EXECUTOR_KEY
from nn4k.invoker import LLMInvoker

cfg = {
    'model_config': './configs/Encoder_0.4B.json',
    NN_EXECUTOR_KEY: 'm2_encoder.M2EncoderExecutor'
}

encoder = LLMInvoker.from_config(cfg)
encoder.warmup_local_model()

data = {
    'image_path': './pics/pokemon.jpeg',
    'texts': ['杰尼龟', '妙蛙种子', '小火龙', '皮卡丘']
}

res = encoder.local_inference(data)
print(res)
