import math
from functools import partial

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch.utils.checkpoint import checkpoint
from antmmf.common import configurable


class PVTImageEncoder(nn.Module):
    """
    Encode a single image with PVT series:
        type: BatchPVTEncoder
          params:
            model_path: {model_path} or None
            model_name: pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_li
                        ,pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
            pretrained: false
            pool_type: avg
            num_output_features: 8
    """

    @configurable
    def __init__(
        self,
        model_name: str,
        model_path: str,
        pool_type: str,
        pretrained: bool,
        num_output_features: int,
        gradient_checkpointing: bool = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.out_dim = 256 if model_name in ["pvt_v2_b0"] else 512
        self.model_name = model_name
        self.model_path = model_path
        self.pretrained = pretrained
        self.gradient_checkpointing = gradient_checkpointing

        self.model = self.build_model()
        pool_func = nn.AdaptiveAvgPool2d if pool_type == "avg" else nn.AdaptiveMaxPool2d
        # -1 will keep the original feature size
        if num_output_features == -1:
            self.pool = nn.Identity()
        elif num_output_features in [1, 2, 3, 5, 7]:
            self.pool = pool_func((num_output_features, 1))
        elif num_output_features == 4:
            self.pool = pool_func((2, 2))
        elif num_output_features == 6:
            self.pool = pool_func((3, 2))
        elif num_output_features == 8:
            self.pool = pool_func((4, 2))
        elif num_output_features == 9:
            self.pool = pool_func((3, 3))
        else:
            raise Exception("num_output_features only has 1 to 9")

    def build_model(self):
        model = PyramidVisionTransformer.from_pretrained(
            model_name=self.model_name,
            weights_path=self.model_path,
            pretrained=self.pretrained,
        )
        return model

    def forward(self, x):
        _, y = self.model(x, gcs=self.gradient_checkpointing)
        out = self.pool(y)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx512


class BatchPVTEncoder(PVTImageEncoder):
    """
    Encode batch images with torchvision ResNet series:
    eg.
        type: BatchImageEncoder
          params:
            encoder_type: pvt-v2-b1
            pretrained: false
            pool_type: avg
            num_output_features: 8
    """

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        # BxNx3x224x224 -> BxNxconfig.num_output_featuresx2048
        bsz, num_imgs = x.size(0), x.size(1)
        x = x.view([-1] + list(x.shape)[2:])  # B*Nx3x224x224
        out = super().forward(x)  # B*Nxnum_featuresx2048
        return out.view([bsz, num_imgs] + list(out.shape[1:])).contiguous()


class PyramidVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dim = embed_dims[3]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                        linear=linear,
                    )
                    for j in range(depths[i])
                ]
            )
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = (
            nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    @classmethod
    def from_pretrained(self, model_name, weights_path=None, pretrained=True):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name of PVT model, e.g. pvt_v2_b0
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.

        Returns:
            A pretrained pvt model.
        """
        model = eval(model_name)()

        if pretrained:
            if isinstance(weights_path, str):
                state_dict = torch.load(weights_path)
            else:
                # down pretrained models
                state_dict = model_zoo.load_url(url_map[model_name])

            ret = model.load_state_dict(state_dict, strict=False)
            assert (
                not ret.missing_keys
            ), "Missing keys when loading pretrained weights: {}".format(
                ret.missing_keys
            )

        return model

    def forward_features(self, x, gcs=None):
        B = x.shape[0]
        for i in range(self.num_stages):

            def forward_stage(i, x):
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                block = getattr(self, f"block{i + 1}")
                norm = getattr(self, f"norm{i + 1}")
                x, H, W = patch_embed(x)
                for blk in block:
                    x = blk(x, H, W)
                x = norm(x)
                if i == self.num_stages - 1:
                    s = x
                else:
                    s = torch.tensor(0)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                return s, x

            if gcs is not None and gcs[i] is True:
                s, x = checkpoint(forward_stage, i, x)
            else:
                s, x = forward_stage(i, x)
        return s.mean(dim=1), x

    def forward(self, x, gcs=None):
        x, feat = self.forward_features(x, gcs)
        x = self.head(x)

        return x, feat


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        linear=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = (
                    self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            linear=linear,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            linear=linear,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


def _conv_filter(state_dict, patch_size=16):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def pvt_v2_b0(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


def pvt_v2_b1(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


def pvt_v2_b2(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


def pvt_v2_b3(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


def pvt_v2_b4(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


def pvt_v2_b5(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


def pvt_v2_b2_li(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        linear=True,
        **kwargs,
    )

    return model


# model download path
url_map = {
    "pvt_v2_b0": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth",
    "pvt_v2_b1": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth",
    "pvt_v2_b2": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth",
    "pvt_v2_b2_li": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2_li.pth",
    "pvt_v2_b3": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth",
    "pvt_v2_b4": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth",
    "pvt_v2_b5": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth",
}

if __name__ == "__main__":

    model_name = "pvt_v2_b2"

    model = PyramidVisionTransformer.from_pretrained(
        model_name=model_name, weights_path=None, pretrained=True
    )
    model.eval()

    sample = torch.rand((2, 3, 224, 224))
    output, feat = model(sample)
