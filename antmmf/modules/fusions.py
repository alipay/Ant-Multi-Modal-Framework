# Copyright (c) 2023 Ant Group and its affiliates.
"""
The fusions module contains various Fusions techniques some based on BLOCK:
Bilinear Superdiagonal Fusion for VQA and VRD. For e.g. LinearSum, ConcatMLP
taken from https://github.com/Cadene/block.bootstrap.pytorch#fusions.

For implementing your own fusion technique, you need to follow these steps:

.. code::
    from torch import nn
    from antmmf.common.registry import registry
    from antmmf.modules.fusions import SimpleLinear

    @regitery.register_fusion("custom")
    class CustomFusion(nn.Module):
        def __init__(self, params=None):
            super().__init__("Custom")
"""

import torch
import torch.nn as nn
from antmmf.common.registry import registry


@registry.register_fusion("simple_linear")
class SimpleLinear(nn.Module):
    def __init__(self, in_dim, n_hidden_1, model_root_dir=None, pretrained=None):
        super(SimpleLinear, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.bn = nn.BatchNorm1d(n_hidden_1)

        nn.init.xavier_uniform_(self.layer1.weight)

        self.writer = registry.get("writer")

        if pretrained is not None:
            if model_root_dir is not None:
                pretrained = "{}/{}".format(model_root_dir, pretrained)
            model = torch.load(pretrained, map_location=torch.device("cpu"))
            if "model_dict" in model:
                # remove the unused key
                model = model["model_dict"]
            self.load_state_dict(model)
            self.writer.write(
                "loaded pretrained fusion model from {}".format(pretrained)
            )

    def forward(self, x, mask=None):
        x = x.view(x.size(0), -1)
        if mask is not None:
            repeats = int(x.size(-1) / mask.size(-1))
            mask = mask.repeat_interleave(repeats=repeats, dim=-1)
            x = x * (1 - mask.float())

        x = self.layer1(x)
        self.new_feature = x
        return nn.functional.normalize(self.new_feature), mask

    def get_adv_parameters(self):
        r"""return parameters for adversarial genration and training
        only do adversarial on the word-embeddings, not on others such as
        positional embedding
        """
        adv_param_names = ["layer1.weight"]
        params = []
        for n, v in self.layer1.named_parameters():
            if n in adv_param_names:
                params.append(v)
        ret = [
            {"params": p, "name": n, "modality": "image"}
            for n, p in zip(adv_param_names, params)
        ]
        return ret


@registry.register_fusion("encoder")
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        from antmmf.modules.encoders import TextEncoder

        encoder = TextEncoder(config)

        self._encoder = encoder.module

    def forward(self, x, mask=None):

        encoded, _ = self._encoder(x, src_key_padding_mask=mask)

        return encoded, mask


@registry.register_fusion("cosine_fusion")
class CosineFusion(nn.Module):
    r"""
    This class uses cosine distance between two vectors, and then normalize to a scalar in [-1,1]
    For matrix a of [bsz, nchn, dim] and b of [bsz, nchn, dim],
    the output is [bsz, nchn]
    """

    def __init__(self):
        super(CosineFusion, self).__init__()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=2)

    def forward(self, item, query):
        assert len(item.shape) == 3, "item needs to be in shape of [bsz, nchn, dim]"
        assert len(query.shape) == 3, "query needs to be in shape of [bsz, nchn, dim]"

        bsz, nchn, dim = item.shape
        assert query.shape == item.shape, "item and query needs to be in the same shape"

        sim = self.cosine_similarity(item, query)
        assert sim.size()[0] == bsz
        assert sim.size()[1] == nchn

        return sim
