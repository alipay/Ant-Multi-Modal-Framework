# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import os
import torch
from torch import nn
import torch.nn.functional as F

from antmmf.modules.encoders import VisualEncoder
from antmmf.modules.vision.backbone.clip.model import VisionTransformer
from antmmf.modules.vision.backbone.clip.cn_model import _MODELS as CNMODELS
from antmmf.modules.vision.backbone.clip.cn_model import _download


@VisualEncoder.register()
class VitImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        out_dim: int,
        head_width=64,
        pretrained=True,
        is_proj=True,
    ):
        super().__init__()
        heads = width // head_width
        self.visual = VisionTransformer(
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=out_dim,
        )
        if not is_proj:
            self.visual.proj = None

        self.out_dim = out_dim
        if pretrained:
            self.load_state_dict(model_name)

    def load_state_dict(self, name, download_root=None):
        download_root = download_root or os.path.expanduser(os.environ["TORCH_HOME"])
        if name in CNMODELS:
            model_path = _download(CNMODELS[name], download_root)
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(
                f"Model {name} not found; available models = {available_models()}"
            )

        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        new_state_dict = {}
        for k in state_dict.keys():
            if "visual" not in k:
                continue
            if "module." in k:
                new_state_dict[k[14:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        if new_state_dict is not None:
            for k in self.visual.state_dict():
                if "module." in k:
                    self.visual.state_dict()[k[7:]].copy_(new_state_dict[k[7:]])
                else:
                    self.visual.state_dict()[k].copy_(new_state_dict[k])

    def forward(self, image, image_mask):
        """
        :param image(torch.float32): [b, N, channels, h, w]
        :param image_mask(torch.bool): [b, N, h, w], with True indicating padding areas.
        :return:
            out_feats(torch.float32): [b, N, self.out_dim, h//32, w//32]
            out_pos(torch.float32): [b, N, 2*position_embedding.params.num_pos_feats, h//32, w//32]
            out_feat_masks(torch.bool): [b, N, h//32, w//32]
        """
        _B, _T, _C, _H, _W = image.shape
        _h, _w = _H // _H, _W // _W
        image = image.view(_B * _T, _C, _H, _W)
        img_feat = self.visual(image)
        img_feat = img_feat.view(_B, _T, self.out_dim).unsqueeze(-1).unsqueeze(-1)
        img_mask = F.interpolate(image_mask.float(), size=(_h, _w)).to(torch.bool)

        output_dict = dict(
            grid_feature=img_feat,
            grid_mask=img_mask,
            grid_feature_with_pos=None,
        )
        return output_dict
