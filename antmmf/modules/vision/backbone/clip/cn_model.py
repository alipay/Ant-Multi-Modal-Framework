# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.
# Code for Official Version Chinese CLIP
# https://github.com/OFA-Sys/Chinese-CLIP

from typing import Tuple, Union
import os
import urllib
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import ssl
from antmmf.common import configurable
from antmmf.modules.vision.backbone.clip.model import ModifiedResNet, VisionTransformer
from antmmf.modules.vision.backbone.clip.configuration_bert import BertConfig
from antmmf.modules.vision.backbone.clip.modeling_bert import BertModel
from antmmf.modules.vision.backbone.clip.cn_tokenizer import FullTokenizer

CONFIGS = {
    "RN50": {
        "embed_dim": 1024,
        "image_resolution": 224,
        "vision_layers": [3, 4, 6, 3],
        "vision_width": 64,
        "vision_patch_size": None,
        "vocab_size": 21128,
        "text_attention_probs_dropout_prob": 0.1,
        "text_hidden_act": "gelu",
        "text_hidden_dropout_prob": 0.1,
        "text_hidden_size": 768,
        "text_initializer_range": 0.02,
        "text_intermediate_size": 3072,
        "text_max_position_embeddings": 512,
        "text_num_attention_heads": 12,
        "text_num_hidden_layers": 3,
        "text_type_vocab_size": 2,
    },
    "ViT-B-16": {
        "embed_dim": 512,
        "image_resolution": 224,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 16,
        "vocab_size": 21128,
        "text_attention_probs_dropout_prob": 0.1,
        "text_hidden_act": "gelu",
        "text_hidden_dropout_prob": 0.1,
        "text_hidden_size": 768,
        "text_initializer_range": 0.02,
        "text_intermediate_size": 3072,
        "text_max_position_embeddings": 512,
        "text_num_attention_heads": 12,
        "text_num_hidden_layers": 12,
        "text_type_vocab_size": 2,
    },
    "ViT-L-14": {
        "embed_dim": 768,
        "image_resolution": 224,
        "vision_layers": 24,
        "vision_width": 1024,
        "vision_head_width": 64,
        "vision_patch_size": 14,
        "vocab_size": 21128,
        "text_attention_probs_dropout_prob": 0.1,
        "text_hidden_act": "gelu",
        "text_hidden_dropout_prob": 0.1,
        "text_hidden_size": 768,
        "text_initializer_range": 0.02,
        "text_intermediate_size": 3072,
        "text_max_position_embeddings": 512,
        "text_num_attention_heads": 12,
        "text_num_hidden_layers": 12,
        "text_type_vocab_size": 2,
    },
    "ViT-L-14-336": {
        "embed_dim": 768,
        "image_resolution": 336,
        "vision_layers": 24,
        "vision_width": 1024,
        "vision_head_width": 64,
        "vision_patch_size": 14,
        "vocab_size": 21128,
        "text_attention_probs_dropout_prob": 0.1,
        "text_hidden_act": "gelu",
        "text_hidden_dropout_prob": 0.1,
        "text_hidden_size": 768,
        "text_initializer_range": 0.02,
        "text_intermediate_size": 3072,
        "text_max_position_embeddings": 512,
        "text_num_attention_heads": 12,
        "text_num_hidden_layers": 12,
        "text_type_vocab_size": 2,
    },
    "ViT-H-14": {
        "embed_dim": 1024,
        "image_resolution": 224,
        "vision_layers": 32,
        "vision_width": 1280,
        "vision_head_width": 80,
        "vision_patch_size": 14,
        "vocab_size": 21128,
        "text_attention_probs_dropout_prob": 0.1,
        "text_hidden_act": "gelu",
        "text_hidden_dropout_prob": 0.1,
        "text_hidden_size": 1024,
        "text_initializer_range": 0.02,
        "text_intermediate_size": 4096,
        "text_max_position_embeddings": 512,
        "text_num_attention_heads": 16,
        "text_num_hidden_layers": 24,
        "text_type_vocab_size": 2,
    },
}
_MODELS = {
    "RN50": "YourUrl" "VLPT/cn_clip/models/clip_cn_rn50.pt",
    "ViT-B-16": "YourUrl/cn_clip/models/clip_cn_vit-b-16.pt",
    "ViT-L-14": "YourUrl/cn_clip/models/clip_cn_vit-l-14.pt",
    "ViT-L-14-336": "YourUrl/cn_clip/models/clip_cn_vit-l-14-336.pt",
    "ViT-H-14": "YourUrl/cn_clip/models/clip_cn_vit-h-14.pt",
}


class CNCLIP(nn.Module):
    @configurable
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        vocab_size: int,
        text_attention_probs_dropout_prob: float,
        text_hidden_act: str,
        text_hidden_dropout_prob: float,
        text_hidden_size: int,
        text_initializer_range: float,
        text_intermediate_size: int,
        text_max_position_embeddings: int,
        text_num_attention_heads: int,
        text_num_hidden_layers: int,
        text_type_vocab_size: int,
        # vision head width, added this param for ViT-H
        vision_head_width: int = 64,
        model_type: str = "all",
    ):
        super().__init__()

        if model_type in ["vision", "all"]:
            if isinstance(vision_layers, (tuple, list)):
                vision_heads = vision_width * 32 // vision_head_width
                self.visual = ModifiedResNet(
                    layers=vision_layers,
                    output_dim=embed_dim,
                    heads=vision_heads,
                    input_resolution=image_resolution,
                    width=vision_width,
                )
            else:
                vision_heads = vision_width // vision_head_width
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                )

        if model_type in ["language", "all"]:
            self.bert_config = BertConfig(
                vocab_size_or_config_json_file=vocab_size,
                hidden_size=text_hidden_size,
                num_hidden_layers=text_num_hidden_layers,
                num_attention_heads=text_num_attention_heads,
                intermediate_size=text_intermediate_size,
                hidden_act=text_hidden_act,
                hidden_dropout_prob=text_hidden_dropout_prob,
                attention_probs_dropout_prob=text_attention_probs_dropout_prob,
                max_position_embeddings=text_max_position_embeddings,
                type_vocab_size=text_type_vocab_size,
                initializer_range=text_initializer_range,
                layer_norm_eps=1e-12,
            )
            self.bert = BertModel(self.bert_config)

            self.text_projection = nn.Parameter(
                torch.empty(text_hidden_size, embed_dim)
            )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.tokenizer = FullTokenizer()

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        pad_index = self.tokenizer.vocab["[PAD]"]
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(
            self.dtype
        )  # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return image_features, text_features, logits_per_image, logits_per_text


class CNCLIPImageEncoder(nn.Module):
    """
    Encode a single image with CLIP image encoder
        type: VisionTransformerImageEncoder
          params:
            model_name: RN50, ViT-B-16, ViT-L-14, ViT-L-14-336, ViT-H-14 or or checkpoint path
            config: The config parameters to build CNCLIP
            pretrained: If load pretrained checkpoint
    """

    @configurable
    def __init__(self, model_name, config, pretrained=True):
        super().__init__()
        self.model = self.build_model(model_name, config, pretrained)

    def build_model(self, model_name, config, pretrained):
        return load(model_name, config, pretrained)

    def forward(self, x):
        image_features = self.model.encode_image(x)

        return image_features


class CNCLIPLanguageEncoder(nn.Module):
    """
    Encode text contents with CNCLIP language encoder:
        type: TextBertEncoder
          params:
            model_name: RN50, ViT-B-16, ViT-L-14, ViT-L-14-336, ViT-H-14 or or checkpoint path
            config: The config parameters to build CNCLIP
            pretrained: If load pretrained checkpoint
    """

    @configurable
    def __init__(self, model_name, config, pretrained=True):
        super().__init__()
        self.model = self.build_model(model_name, config, pretrained)

    def build_model(self, model_name, config, pretrained):
        return load(model_name, config, pretrained)

    def forward(self, text):
        text_features = self.model.encode_text(text)
        return text_features


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(l, BertModel):
            l.to(torch.half)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(config: dict, state_dict: dict = None):
    model = CNCLIP(**config)
    if state_dict is not None:
        for k in model.state_dict():
            if "module." in k:
                model.state_dict()[k[7:]].copy_(state_dict[k[7:]])
            else:
                model.state_dict()[k].copy_(state_dict[k])
    return model.eval()


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-1]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    ssl._create_default_https_context = ssl._create_unverified_context
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def available_models():
    """Returns the names of available CNCLIP models"""
    return list(_MODELS.keys())


def load(
    name: str,
    config: dict,
    pretrained: bool,
    device="cpu",
    download_root: str = None,
):
    """
    Load a CNCLIP model

    Parameters
    ----------
    config : name
        A model name listed by `_MODELS.keys()`,

    config : dict
        The config parameters to build CNCLIP

    pretrained : bool
        If load pretrained checkpoint

    device : Union[str, torch.device]
        The device to put the loaded model


    Returns
    -------
    model : torch.nn.Module
        The CNCLIP model

    """
    if pretrained:
        download_root = download_root or os.path.expanduser(os.environ["TORCH_HOME"])
        if name in _MODELS:
            model_path = _download(_MODELS[name], download_root)
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(
                f"Model {name} not found; available models = {available_models()}"
            )
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        new_state_dict = {}
        for k in state_dict.keys():
            if "module." in k:
                new_state_dict[k[7:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]

        model = build_model(config, new_state_dict).to(device)
        if str(device) == "cpu":
            model.float()
    else:
        model = build_model(config).to(device)
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "RN50"
    pretrained = True
    from torchvision.transforms import (
        Compose,
        ToTensor,
        Normalize,
        Resize,
        InterpolationMode,
    )
    from PIL import Image

    image_size = CONFIGS[model_name]["image_resolution"]
    preprocess = Compose(
        [
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    input_tensor = (
        preprocess(Image.open("tests/data/image/dog.jpg").convert("RGB"))
        .unsqueeze(0)
        .to(device)
    )

    vision_model = CNCLIPImageEncoder(
        model_name, CONFIGS[model_name], pretrained=pretrained
    ).to(device)
    language_model = CNCLIPLanguageEncoder(
        model_name, CONFIGS[model_name], pretrained=pretrained
    ).to(device)

    from antmmf.datasets.processors.text_processors import CNCLIPTokenizerProcessor

    _tokenizer = CNCLIPTokenizerProcessor(CONFIGS[model_name])

    text = _tokenizer({"text": "宠物狗"})["input_ids"].unsqueeze(0).to(device)
    print("input text ids: ", text.shape, text)
    with torch.no_grad():
        output = vision_model(input_tensor)
        print("vision_feature", output.shape, output)
        output = language_model(text)
        print("text_feature", output.shape, output)
