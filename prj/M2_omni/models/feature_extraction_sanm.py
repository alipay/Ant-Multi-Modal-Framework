# coding=utf-8
# Copyright 2024 The ANT Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for BailingAudio.
"""
import os.path

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import numpy as np
from typing import List, Optional, Union, Tuple, Dict

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)

__all__ = [
    'load_audio',
    'SANMFeatureExtractor'
]

audio_mean = [
    -8.311879, -8.600912, -9.615928, -10.43595, -11.21292,
    -11.88333, -12.36243, -12.63706, -12.8818, -12.83066,
    -12.89103, -12.95666, -13.19763, -13.40598, -13.49113,
    -13.5546, -13.55639, -13.51915, -13.68284, -13.53289,
    -13.42107, -13.65519, -13.50713, -13.75251, -13.76715,
    -13.87408, -13.73109, -13.70412, -13.56073, -13.53488,
    -13.54895, -13.56228, -13.59408, -13.62047, -13.64198,
    -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
    -13.53063, -13.48348, -13.61047, -13.64716, -13.71546,
    -13.79184, -13.90614, -14.03098, -14.18205, -14.35881,
    -14.48419, -14.60172, -14.70591, -14.83362, -14.92122,
    -15.00622, -15.05122, -15.03119, -14.99028, -14.92302,
    -14.86927, -14.82691, -14.7972, -14.76909, -14.71356,
    -14.61277, -14.51696, -14.42252, -14.36405, -14.30451,
    -14.23161, -14.19851, -14.16633, -14.15649, -14.10504,
    -13.99518, -13.79562, -13.3996, -12.7767, -11.71208,
    -8.311879, -8.600912, -9.615928, -10.43595, -11.21292,
    -11.88333, -12.36243, -12.63706, -12.8818, -12.83066,
    -12.89103, -12.95666, -13.19763, -13.40598, -13.49113,
    -13.5546, -13.55639, -13.51915, -13.68284, -13.53289,
    -13.42107, -13.65519, -13.50713, -13.75251, -13.76715,
    -13.87408, -13.73109, -13.70412, -13.56073, -13.53488,
    -13.54895, -13.56228, -13.59408, -13.62047, -13.64198,
    -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
    -13.53063, -13.48348, -13.61047, -13.64716, -13.71546,
    -13.79184, -13.90614, -14.03098, -14.18205, -14.35881,
    -14.48419, -14.60172, -14.70591, -14.83362, -14.92122,
    -15.00622, -15.05122, -15.03119, -14.99028, -14.92302,
    -14.86927, -14.82691, -14.7972, -14.76909, -14.71356,
    -14.61277, -14.51696, -14.42252, -14.36405, -14.30451,
    -14.23161, -14.19851, -14.16633, -14.15649, -14.10504,
    -13.99518, -13.79562, -13.3996, -12.7767, -11.71208,
    -8.311879, -8.600912, -9.615928, -10.43595, -11.21292,
    -11.88333, -12.36243, -12.63706, -12.8818, -12.83066,
    -12.89103, -12.95666, -13.19763, -13.40598, -13.49113,
    -13.5546, -13.55639, -13.51915, -13.68284, -13.53289,
    -13.42107, -13.65519, -13.50713, -13.75251, -13.76715,
    -13.87408, -13.73109, -13.70412, -13.56073, -13.53488,
    -13.54895, -13.56228, -13.59408, -13.62047, -13.64198,
    -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
    -13.53063, -13.48348, -13.61047, -13.64716, -13.71546,
    -13.79184, -13.90614, -14.03098, -14.18205, -14.35881,
    -14.48419, -14.60172, -14.70591, -14.83362, -14.92122,
    -15.00622, -15.05122, -15.03119, -14.99028, -14.92302,
    -14.86927, -14.82691, -14.7972, -14.76909, -14.71356,
    -14.61277, -14.51696, -14.42252, -14.36405, -14.30451,
    -14.23161, -14.19851, -14.16633, -14.15649, -14.10504,
    -13.99518, -13.79562, -13.3996, -12.7767, -11.71208,
    -8.311879, -8.600912, -9.615928, -10.43595, -11.21292,
    -11.88333, -12.36243, -12.63706, -12.8818, -12.83066,
    -12.89103, -12.95666, -13.19763, -13.40598, -13.49113,
    -13.5546, -13.55639, -13.51915, -13.68284, -13.53289,
    -13.42107, -13.65519, -13.50713, -13.75251, -13.76715,
    -13.87408, -13.73109, -13.70412, -13.56073, -13.53488,
    -13.54895, -13.56228, -13.59408, -13.62047, -13.64198,
    -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
    -13.53063, -13.48348, -13.61047, -13.64716, -13.71546,
    -13.79184, -13.90614, -14.03098, -14.18205, -14.35881,
    -14.48419, -14.60172, -14.70591, -14.83362, -14.92122,
    -15.00622, -15.05122, -15.03119, -14.99028, -14.92302,
    -14.86927, -14.82691, -14.7972, -14.76909, -14.71356,
    -14.61277, -14.51696, -14.42252, -14.36405, -14.30451,
    -14.23161, -14.19851, -14.16633, -14.15649, -14.10504,
    -13.99518, -13.79562, -13.3996, -12.7767, -11.71208,
    -8.311879, -8.600912, -9.615928, -10.43595, -11.21292,
    -11.88333, -12.36243, -12.63706, -12.8818, -12.83066,
    -12.89103, -12.95666, -13.19763, -13.40598, -13.49113,
    -13.5546, -13.55639, -13.51915, -13.68284, -13.53289,
    -13.42107, -13.65519, -13.50713, -13.75251, -13.76715,
    -13.87408, -13.73109, -13.70412, -13.56073, -13.53488,
    -13.54895, -13.56228, -13.59408, -13.62047, -13.64198,
    -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
    -13.53063, -13.48348, -13.61047, -13.64716, -13.71546,
    -13.79184, -13.90614, -14.03098, -14.18205, -14.35881,
    -14.48419, -14.60172, -14.70591, -14.83362, -14.92122,
    -15.00622, -15.05122, -15.03119, -14.99028, -14.92302,
    -14.86927, -14.82691, -14.7972, -14.76909, -14.71356,
    -14.61277, -14.51696, -14.42252, -14.36405, -14.30451,
    -14.23161, -14.19851, -14.16633, -14.15649, -14.10504,
    -13.99518, -13.79562, -13.3996, -12.7767, -11.71208,
    -8.311879, -8.600912, -9.615928, -10.43595, -11.21292,
    -11.88333, -12.36243, -12.63706, -12.8818, -12.83066,
    -12.89103, -12.95666, -13.19763, -13.40598, -13.49113,
    -13.5546, -13.55639, -13.51915, -13.68284, -13.53289,
    -13.42107, -13.65519, -13.50713, -13.75251, -13.76715,
    -13.87408, -13.73109, -13.70412, -13.56073, -13.53488,
    -13.54895, -13.56228, -13.59408, -13.62047, -13.64198,
    -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
    -13.53063, -13.48348, -13.61047, -13.64716, -13.71546,
    -13.79184, -13.90614, -14.03098, -14.18205, -14.35881,
    -14.48419, -14.60172, -14.70591, -14.83362, -14.92122,
    -15.00622, -15.05122, -15.03119, -14.99028, -14.92302,
    -14.86927, -14.82691, -14.7972, -14.76909, -14.71356,
    -14.61277, -14.51696, -14.42252, -14.36405, -14.30451,
    -14.23161, -14.19851, -14.16633, -14.15649, -14.10504,
    -13.99518, -13.79562, -13.3996, -12.7767, -11.71208,
    -8.311879, -8.600912, -9.615928, -10.43595, -11.21292,
    -11.88333, -12.36243, -12.63706, -12.8818, -12.83066,
    -12.89103, -12.95666, -13.19763, -13.40598, -13.49113,
    -13.5546, -13.55639, -13.51915, -13.68284, -13.53289,
    -13.42107, -13.65519, -13.50713, -13.75251, -13.76715,
    -13.87408, -13.73109, -13.70412, -13.56073, -13.53488,
    -13.54895, -13.56228, -13.59408, -13.62047, -13.64198,
    -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
    -13.53063, -13.48348, -13.61047, -13.64716, -13.71546,
    -13.79184, -13.90614, -14.03098, -14.18205, -14.35881,
    -14.48419, -14.60172, -14.70591, -14.83362, -14.92122,
    -15.00622, -15.05122, -15.03119, -14.99028, -14.92302,
    -14.86927, -14.82691, -14.7972, -14.76909, -14.71356,
    -14.61277, -14.51696, -14.42252, -14.36405, -14.30451,
    -14.23161, -14.19851, -14.16633, -14.15649, -14.10504,
    -13.99518, -13.79562, -13.3996, -12.7767, -11.71208]

audio_std = [
    0.155775, 0.154484, 0.1527379, 0.1518718, 0.1506028, 0.1489256,
    0.147067, 0.1447061, 0.1436307, 0.1443568, 0.1451849, 0.1455157,
    0.1452821, 0.1445717, 0.1439195, 0.1435867, 0.1436018, 0.1438781,
    0.1442086, 0.1448844, 0.1454756, 0.145663, 0.146268, 0.1467386,
    0.1472724, 0.147664, 0.1480913, 0.1483739, 0.1488841, 0.1493636,
    0.1497088, 0.1500379, 0.1502916, 0.1505389, 0.1506787, 0.1507102,
    0.1505992, 0.1505445, 0.1505938, 0.1508133, 0.1509569, 0.1512396,
    0.1514625, 0.1516195, 0.1516156, 0.1515561, 0.1514966, 0.1513976,
    0.1512612, 0.151076, 0.1510596, 0.1510431, 0.151077, 0.1511168,
    0.1511917, 0.151023, 0.1508045, 0.1505885, 0.1503493, 0.1502373,
    0.1501726, 0.1500762, 0.1500065, 0.1499782, 0.150057, 0.1502658,
    0.150469, 0.1505335, 0.1505505, 0.1505328, 0.1504275, 0.1502438,
    0.1499674, 0.1497118, 0.1494661, 0.1493102, 0.1493681, 0.1495501,
    0.1499738, 0.1509654, 0.155775, 0.154484, 0.1527379, 0.1518718,
    0.1506028, 0.1489256, 0.147067, 0.1447061, 0.1436307, 0.1443568,
    0.1451849, 0.1455157, 0.1452821, 0.1445717, 0.1439195, 0.1435867,
    0.1436018, 0.1438781, 0.1442086, 0.1448844, 0.1454756, 0.145663,
    0.146268, 0.1467386, 0.1472724, 0.147664, 0.1480913, 0.1483739,
    0.1488841, 0.1493636, 0.1497088, 0.1500379, 0.1502916, 0.1505389,
    0.1506787, 0.1507102, 0.1505992, 0.1505445, 0.1505938, 0.1508133,
    0.1509569, 0.1512396, 0.1514625, 0.1516195, 0.1516156, 0.1515561,
    0.1514966, 0.1513976, 0.1512612, 0.151076, 0.1510596, 0.1510431,
    0.151077, 0.1511168, 0.1511917, 0.151023, 0.1508045, 0.1505885,
    0.1503493, 0.1502373, 0.1501726, 0.1500762, 0.1500065, 0.1499782,
    0.150057, 0.1502658, 0.150469, 0.1505335, 0.1505505, 0.1505328,
    0.1504275, 0.1502438, 0.1499674, 0.1497118, 0.1494661, 0.1493102,
    0.1493681, 0.1495501, 0.1499738, 0.1509654, 0.155775, 0.154484,
    0.1527379, 0.1518718, 0.1506028, 0.1489256, 0.147067, 0.1447061,
    0.1436307, 0.1443568, 0.1451849, 0.1455157, 0.1452821, 0.1445717,
    0.1439195, 0.1435867, 0.1436018, 0.1438781, 0.1442086, 0.1448844,
    0.1454756, 0.145663, 0.146268, 0.1467386, 0.1472724, 0.147664,
    0.1480913, 0.1483739, 0.1488841, 0.1493636, 0.1497088, 0.1500379,
    0.1502916, 0.1505389, 0.1506787, 0.1507102, 0.1505992, 0.1505445,
    0.1505938, 0.1508133, 0.1509569, 0.1512396, 0.1514625, 0.1516195,
    0.1516156, 0.1515561, 0.1514966, 0.1513976, 0.1512612, 0.151076,
    0.1510596, 0.1510431, 0.151077, 0.1511168, 0.1511917, 0.151023,
    0.1508045, 0.1505885, 0.1503493, 0.1502373, 0.1501726, 0.1500762,
    0.1500065, 0.1499782, 0.150057, 0.1502658, 0.150469, 0.1505335,
    0.1505505, 0.1505328, 0.1504275, 0.1502438, 0.1499674, 0.1497118,
    0.1494661, 0.1493102, 0.1493681, 0.1495501, 0.1499738, 0.1509654,
    0.155775, 0.154484, 0.1527379, 0.1518718, 0.1506028, 0.1489256,
    0.147067, 0.1447061, 0.1436307, 0.1443568, 0.1451849, 0.1455157,
    0.1452821, 0.1445717, 0.1439195, 0.1435867, 0.1436018, 0.1438781,
    0.1442086, 0.1448844, 0.1454756, 0.145663, 0.146268, 0.1467386,
    0.1472724, 0.147664, 0.1480913, 0.1483739, 0.1488841, 0.1493636,
    0.1497088, 0.1500379, 0.1502916, 0.1505389, 0.1506787, 0.1507102,
    0.1505992, 0.1505445, 0.1505938, 0.1508133, 0.1509569, 0.1512396,
    0.1514625, 0.1516195, 0.1516156, 0.1515561, 0.1514966, 0.1513976,
    0.1512612, 0.151076, 0.1510596, 0.1510431, 0.151077, 0.1511168,
    0.1511917, 0.151023, 0.1508045, 0.1505885, 0.1503493, 0.1502373,
    0.1501726, 0.1500762, 0.1500065, 0.1499782, 0.150057, 0.1502658,
    0.150469, 0.1505335, 0.1505505, 0.1505328, 0.1504275, 0.1502438,
    0.1499674, 0.1497118, 0.1494661, 0.1493102, 0.1493681, 0.1495501,
    0.1499738, 0.1509654, 0.155775, 0.154484, 0.1527379, 0.1518718,
    0.1506028, 0.1489256, 0.147067, 0.1447061, 0.1436307, 0.1443568,
    0.1451849, 0.1455157, 0.1452821, 0.1445717, 0.1439195, 0.1435867,
    0.1436018, 0.1438781, 0.1442086, 0.1448844, 0.1454756, 0.145663,
    0.146268, 0.1467386, 0.1472724, 0.147664, 0.1480913, 0.1483739,
    0.1488841, 0.1493636, 0.1497088, 0.1500379, 0.1502916, 0.1505389,
    0.1506787, 0.1507102, 0.1505992, 0.1505445, 0.1505938, 0.1508133,
    0.1509569, 0.1512396, 0.1514625, 0.1516195, 0.1516156, 0.1515561,
    0.1514966, 0.1513976, 0.1512612, 0.151076, 0.1510596, 0.1510431,
    0.151077, 0.1511168, 0.1511917, 0.151023, 0.1508045, 0.1505885,
    0.1503493, 0.1502373, 0.1501726, 0.1500762, 0.1500065, 0.1499782,
    0.150057, 0.1502658, 0.150469, 0.1505335, 0.1505505, 0.1505328,
    0.1504275, 0.1502438, 0.1499674, 0.1497118, 0.1494661, 0.1493102,
    0.1493681, 0.1495501, 0.1499738, 0.1509654, 0.155775, 0.154484,
    0.1527379, 0.1518718, 0.1506028, 0.1489256, 0.147067, 0.1447061,
    0.1436307, 0.1443568, 0.1451849, 0.1455157, 0.1452821, 0.1445717,
    0.1439195, 0.1435867, 0.1436018, 0.1438781, 0.1442086, 0.1448844,
    0.1454756, 0.145663, 0.146268, 0.1467386, 0.1472724, 0.147664,
    0.1480913, 0.1483739, 0.1488841, 0.1493636, 0.1497088, 0.1500379,
    0.1502916, 0.1505389, 0.1506787, 0.1507102, 0.1505992, 0.1505445,
    0.1505938, 0.1508133, 0.1509569, 0.1512396, 0.1514625, 0.1516195,
    0.1516156, 0.1515561, 0.1514966, 0.1513976, 0.1512612, 0.151076,
    0.1510596, 0.1510431, 0.151077, 0.1511168, 0.1511917, 0.151023,
    0.1508045, 0.1505885, 0.1503493, 0.1502373, 0.1501726, 0.1500762,
    0.1500065, 0.1499782, 0.150057, 0.1502658, 0.150469, 0.1505335,
    0.1505505, 0.1505328, 0.1504275, 0.1502438, 0.1499674, 0.1497118,
    0.1494661, 0.1493102, 0.1493681, 0.1495501, 0.1499738, 0.1509654,
    0.155775, 0.154484, 0.1527379, 0.1518718, 0.1506028, 0.1489256,
    0.147067, 0.1447061, 0.1436307, 0.1443568, 0.1451849, 0.1455157,
    0.1452821, 0.1445717, 0.1439195, 0.1435867, 0.1436018, 0.1438781,
    0.1442086, 0.1448844, 0.1454756, 0.145663, 0.146268, 0.1467386,
    0.1472724, 0.147664, 0.1480913, 0.1483739, 0.1488841, 0.1493636,
    0.1497088, 0.1500379, 0.1502916, 0.1505389, 0.1506787, 0.1507102,
    0.1505992, 0.1505445, 0.1505938, 0.1508133, 0.1509569, 0.1512396,
    0.1514625, 0.1516195, 0.1516156, 0.1515561, 0.1514966, 0.1513976,
    0.1512612, 0.151076, 0.1510596, 0.1510431, 0.151077, 0.1511168,
    0.1511917, 0.151023, 0.1508045, 0.1505885, 0.1503493, 0.1502373,
    0.1501726, 0.1500762, 0.1500065, 0.1499782, 0.150057, 0.1502658,
    0.150469, 0.1505335, 0.1505505, 0.1505328, 0.1504275, 0.1502438,
    0.1499674, 0.1497118, 0.1494661, 0.1493102, 0.1493681, 0.1495501,
    0.1499738, 0.1509654
]

def load_audio(audio_file, sample_rate=16000):
    waveform, orig_freq = torchaudio.load(audio_file, normalize=True)

    NORM_FACTOR_FOR_DTYPE = {
        torch.int8: 2**7,
        torch.int16: 2**15,
        torch.int32: 2**31,
        torch.int64: 2**63,
        torch.float32: 1,
        torch.float64: 1,
    }
    assert waveform.dtype in NORM_FACTOR_FOR_DTYPE, f"Unsupported waveform dtype: {waveform.dtype}"
    norm_factor = NORM_FACTOR_FOR_DTYPE[waveform.dtype]
    waveform = waveform.to(torch.float32) / norm_factor

    if len(waveform.shape) > 1:
        waveform = waveform[0]
    if orig_freq != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=sample_rate)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
    return waveform

def make_batched_audios(audios) -> List[torch.Tensor]:
    """
    Accepts audios in list or nested list format, and makes a list of images for preprocessing.

    Args:
            audios (`torch.Tensor`, `List[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.

    Returns:
        list: A list of audios.
    """
    if isinstance(audios, (list, tuple)) and isinstance(audios[0], (list, tuple)):
        return [audio for audio_list in audios for audio in audio_list]

    elif isinstance(audios, (list, tuple)):
        return audios
    else:
        return [audios]

class SANMFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a SANM feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        frame_length (float, optional):
            Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional):
            Frame shift in milliseconds (Default: ``10.0``)
        up_sample (`bool`, *optional*, defaults to True):
            Whether to upsample values of mel_energies.
        inverse_norm (`bool`, *optional*, defaults to True):
            Whether to inverse normalization of mel_energies.
    """

    model_input_names = ["pixel_values_audios"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        chunk_length=30,
        padding_value: float = 0.0,

        frame_length: int = 25,
        frame_shift: int = 10,

        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 0.0,
        up_sample: bool = True,
        inverse_norm: bool = True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )

        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate

        self.frame_length = frame_length
        self.frame_shift = frame_shift

        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.dither = dither
        self.up_sample = up_sample
        self.inverse_norm = inverse_norm

    @staticmethod
    def de_normalize(inputs, audio_mean, audio_std):
        audio_mean = np.array(audio_mean, dtype=np.float32)
        audio_std = np.array(audio_std, dtype=np.float32)
        dim = inputs.shape[-1]
        inputs += audio_mean[:dim]
        inputs *= audio_std[:dim]
        return inputs.type(torch.float32)

    @staticmethod
    def apply_lfr(inputs, lfr_m, lfr_n):
        LFR_inputs = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
        inputs = torch.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n:i * lfr_n + lfr_m]).view(1, -1))
            else:  # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = (inputs[i * lfr_n:]).view(-1)
                for _ in range(num_padding):
                    frame = torch.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)
        LFR_outputs = torch.vstack(LFR_inputs)
        return LFR_outputs.type(torch.float32)

    def __call__(
        self,
        raw_audios: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        **kwargs,
    ) -> BatchFeature:
        if raw_audios is not None:
            raw_audios = make_batched_audios(raw_audios)

        input_feats = []
        for raw_audio in raw_audios:
            if self.up_sample:
                raw_audio = raw_audio * (1 << 15)

            raw_audio = raw_audio.unsqueeze(0)
            raw_audio = kaldi.fbank(
                raw_audio,
                num_mel_bins=self.feature_size,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type='hamming',
                sample_frequency=self.sampling_rate,
                snip_edges=True
            )

            if self.lfr_m != 1 or self.lfr_n != 1:
                raw_audio = self.apply_lfr(raw_audio, self.lfr_m, self.lfr_n)
            if self.inverse_norm:
                # from .audio_mean_std import audio_mean, audio_std
                raw_audio = self.de_normalize(raw_audio, audio_mean=audio_mean, audio_std=audio_std)

            if not isinstance(raw_audio, np.ndarray):
                raw_audio = raw_audio.numpy().astype(np.float32)
            input_feats.append(raw_audio)

        input_lengths = [len(input_feat) for input_feat in input_feats]
        batched_speech = BatchFeature({"pixel_values_audios": input_feats})

        # convert into correct format for padding
        max_length = min(max(input_lengths), self.n_samples)
        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        padded_inputs["pixel_values_audios"] = np.stack(padded_inputs["pixel_values_audios"], axis=0)
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
        return padded_inputs
