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

# result of Encoder_0.4B.json
# [[3.3752596e-07 7.2387557e-08 2.4727814e-07 9.9999928e-01]]

# result of Encoder_1B.json
# [[1.5551204e-06 4.3255361e-07 7.5551941e-07 9.9999726e-01]]

# result of Encoder_10B.json
# [[6.2423642e-05 2.7531558e-05 6.2977582e-05 9.9984785e-01]]
