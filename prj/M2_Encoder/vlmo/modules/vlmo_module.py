import math
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning.utilities.distributed import rank_zero_info
from timm.models import create_model
from transformers import AutoTokenizer, BertTokenizer, XLMRobertaTokenizer  # noqa
from vlmo.modules import heads, objectives, vlmo_utils
from vlmo.tokenizer.tokenization_glm import GLMChineseTokenizer  # noqa
from vlmo.torchscale.architecture.encoder import Encoder
from vlmo.torchscale.model.BEiT3 import BEiT3 as ts_backbone
from vlmo.transforms.utils import inception_normalize as img_norm

from .modeling_utils import _get_base_config, _get_large_config, _get_huge_config, trunc_normal_  # noqa


def convert_pl_ckpt(state_dict, num_visual_token=197):
    print("start convert_pl_ckpt!!!")
    new_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        if "visual_tokenizer" in key:
            continue
        elif "backbone.encoder.embed_positions.A.weight" in key:
            if value.shape[0] < num_visual_token + 2:
                N = value.shape[0] - 3
                dim = value.shape[-1]
                class_pos_embed = value[:3, ]
                patch_pos_embed = value[3:, ]
                w0, h0 = int(math.sqrt(num_visual_token - 1)), int(math.sqrt(num_visual_token - 1))
                patch_pos_embed = patch_pos_embed.float()
                patch_pos_embed = nn.functional.interpolate(
                    patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                    size=(w0, h0),
                    mode="area",
                )
                patch_pos_embed = patch_pos_embed.to(class_pos_embed.dtype)
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
                new_value = torch.cat((class_pos_embed, patch_pos_embed), dim=0)
                new_state_dict[key] = new_value
                print("reshape ", key, "raw shape: ", value.shape, "new shape: ", new_value.shape, num_visual_token)
            elif value.shape[0] > num_visual_token + 2:
                new_state_dict[key] = value[: num_visual_token + 2, :]
                print("first ", key, "raw shape: ", value.shape, new_state_dict[key].shape, num_visual_token)
            else:
                new_state_dict[key] = value
                print("raw shape")
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def convert_deepspeed_ckpt(state_dict, num_visual_token=197):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("_forward_module."):
            new_key = key[len("_forward_module."):]
            value = state_dict[key]
            new_state_dict[new_key] = value
            if "visual_tokenizer.encoder.pos_embed" in new_key or "visual_tokenizer.decoder.pos_embed" in new_key:
                if value.shape[1] != num_visual_token:
                    N = value.shape[1] - 1
                    dim = value.shape[-1]
                    class_pos_embed = value[:, 0]
                    patch_pos_embed = value[:, 1:]
                    w0, h0 = int(math.sqrt(num_visual_token - 1)), int(math.sqrt(num_visual_token - 1))
                    patch_pos_embed = patch_pos_embed.float()
                    patch_pos_embed = nn.functional.interpolate(
                        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                        size=(w0, h0),
                        mode="area",
                    )
                    patch_pos_embed = patch_pos_embed.to(class_pos_embed.dtype)
                    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
                    new_value = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
                    new_state_dict[new_key] = new_value
                    print("reshape ", new_key, "raw shape: ", value.shape, "new_shape: ", new_value.shape)
            if "backbone.encoder.embed_positions.A.weight" in new_key:
                if value.shape[1] != num_visual_token + 2:
                    N = value.shape[0] - 3
                    dim = value.shape[-1]
                    class_pos_embed = value[:3, ]
                    patch_pos_embed = value[3:, ]
                    w0, h0 = int(math.sqrt(num_visual_token - 1)), int(math.sqrt(num_visual_token - 1))
                    patch_pos_embed = patch_pos_embed.float()
                    patch_pos_embed = nn.functional.interpolate(
                        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                        size=(w0, h0),
                        mode="area",
                    )
                    patch_pos_embed = patch_pos_embed.to(class_pos_embed.dtype)
                    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
                    new_value = torch.cat((class_pos_embed, patch_pos_embed), dim=0)
                    new_state_dict[new_key] = new_value
                    print("reshape ", new_key, "raw shape: ", value.shape, "new_shape: ", new_value.shape)

        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def get_visual_tokenizer(config):
    tokenizer_name = config["tokenizer_model"]
    print(f"Creating visual tokenizer: {tokenizer_name}")
    model = create_model(
        config["tokenizer_model"],
        img_size=config["image_size"],
        n_code=config["codebook_size"],
        code_dim=config["codebook_dim"],
    ).eval()
    return model


def get_pretrained_tokenizer(tokenizer_type, from_pretrained):
    _Tokenizer = eval(f"{tokenizer_type}")
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            _Tokenizer.from_pretrained(from_pretrained)
        torch.distributed.barrier()
    return _Tokenizer.from_pretrained(from_pretrained)


class VLMo(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        s_t = time.time()

        # tokenizer & backbone
        self.img_size = config["image_size"]
        if not config["test_only"]:
            self.visual_tokenizer = get_visual_tokenizer(config)
        kwargs = {}
        if "encoder_attention_heads" in config:
            kwargs["encoder_attention_heads"] = config["encoder_attention_heads"]
        if "atorch_config" in config and config["atorch_config"]:
            checkpoint_activations = False  # ?
        else:
            checkpoint_activations = config["checkpoint_activations"]
        args = eval(f'_get_{config["beit_version"]}_config')(
            img_size=config["image_size"],
            patch_size=config["patch_size"],
            vocab_size=config["vocab_size"],
            encoder_layers=config["encoder_layers"],
            encoder_embed_dim=config["encoder_embed_dim"],
            checkpoint_activations=checkpoint_activations,
            share_layer=config["share_layer"],
            share_attn=config["share_attn"],
            deepnorm=config["deepnorm"],
            mask_ratio=config["mask_ratio"],
            max_text_len=config["max_text_len"],
            one_attn=config["one_attn"],
            **kwargs,
        )
        self.num_features = args.encoder_embed_dim
        self.out_features = config["out_embed_dim"]
        self.cap_onlytext = config["cap_onlytext"]
        self.lang = config["lang"]
        self.num_frames = config["num_frames"]
        self.tokenizer_type = config["tokenizer_type"]
        self.text_tokenizer = get_pretrained_tokenizer(self.tokenizer_type, from_pretrained=config["tokenizer"])  # noqa
        print("BEiT args", args.__dict__)
        self.backbone = ts_backbone(args)

        self.use_vl = config["beit3_vl_layers"] > 0
        if self.use_vl:
            args.encoder_layers = config["beit3_vl_layers"]
            self.backbone_vl = Encoder(args)

        self.norm = nn.LayerNorm(self.num_features, eps=1e-6)

        # task layers
        self.pooler = heads.Pooler(self.num_features)
        self.pooler.apply(objectives.init_weights)

        # contrastive loss (or sampling for global hard negative)
        if config["loss_names"]["itc"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features, self.out_features)
            self.itc_image_proj = heads.ITCHead(self.num_features, self.out_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.itc_vl_text_proj = heads.ITCHead(self.num_features, self.out_features)
            self.itc_vl_image_proj = heads.ITCHead(self.num_features, self.out_features)
            self.itc_vl_text_proj.apply(objectives.init_weights)
            self.itc_vl_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        lp_s_t = time.time()

        self.load_pretrained_weight()
        load_pretrain_time = time.time() - lp_s_t

        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")

            state_dict = None

            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict, self.backbone.vision_embed.num_position_embeddings())
            if state_dict_key == "state_dict":
                state_dict = convert_pl_ckpt(state_dict, self.backbone.vision_embed.num_position_embeddings())
            if state_dict is None:
                if list(ckpt.keys())[0].startswith('_forward_module.'):
                    rank_zero_info("Read state dict from ckpt with _forward_module prefix. ")
                    state_dict = convert_deepspeed_ckpt(ckpt, self.backbone.vision_embed.num_position_embeddings())
                else:
                    rank_zero_info("Read state dict from ckpt. ")
                    state_dict = ckpt

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            rank_zero_info("missing_keys: {}".format(missing_keys))
            rank_zero_info("unexpected_keys: {}".format(unexpected_keys))

        construct_time = time.time() - s_t
        print(
            f"Process {os.getpid()}. VLMo Constructor time: {construct_time}s;",
            f"load_pretrain_time: {load_pretrain_time}s",
            flush=True,
        )
        # coalesce backbone calls
        self._coalesce_backbone = config["coalesce_backbone"]
        self._mask_data = config["mask_data"]
        self._backbone_inputs = {}
        self._backbone_inputs_current_size = 0
        self._backbone_inputs_keys = {}
        self._backbone_outputs = None
        self._default_attn_masks = {}
        self._itc_group = None
        self._itc_aggregate_dict = None
        self._itc_mask = config["itc_mask"]
        self._local_loss = config["local_loss"]
        self._aggregate_nodes = config["aggregate_nodes"]
        self.accumulated_batches_reached = False
        vlmo_utils.set_task(self)
        self._only_itc_single_machine = (
                self._aggregate_nodes > 0 and len(self.current_tasks) == 1 and "itc" in self.current_tasks
        )
        self._split_data_for_imagemlm = config["split_data_for_imagemlm"]
        self.log_metric_steps = config["log_metric_steps"]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.backbone.encoder.layers):
            rescale(layer.self_attn.v_proj.A.weight.data, layer_id + 1)
            rescale(layer.self_attn.v_proj.B.weight.data, layer_id + 1)
            rescale(layer.self_attn.out_proj.A.weight.data, layer_id + 1)
            rescale(layer.self_attn.out_proj.B.weight.data, layer_id + 1)
            rescale(layer.ffn.A.fc2.weight.data, layer_id + 1)
            rescale(layer.ffn.B.fc2.weight.data, layer_id + 1)

        if self.use_vl:
            pre_layers = len(self.backbone.encoder.layers) + 1
            for layer_id, layer in enumerate(self.backbone_vl.layers):
                rescale(layer.self_attn.v_proj.A.weight.data, layer_id + pre_layers)
                rescale(layer.self_attn.v_proj.B.weight.data, layer_id + pre_layers)
                rescale(layer.self_attn.out_proj.A.weight.data, layer_id + pre_layers)
                rescale(layer.self_attn.out_proj.B.weight.data, layer_id + pre_layers)
                rescale(layer.ffn.A.fc2.weight.data, layer_id + pre_layers)
                rescale(layer.ffn.B.fc2.weight.data, layer_id + pre_layers)

    def load_pretrained_weight(self):
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            config = self.hparams.config
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))

            state_dict = None

            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict, self.backbone.vision_embed.num_position_embeddings())
            if state_dict_key == "state_dict":
                state_dict = convert_pl_ckpt(state_dict, self.backbone.vision_embed.num_position_embeddings())
            if state_dict is None:
                if list(ckpt.keys())[0].startswith('_forward_module.'):
                    rank_zero_info("Read state dict from ckpt with _forward_module prefix. ")
                    state_dict = convert_deepspeed_ckpt(ckpt,
                                                        self.backbone.vision_embed.num_position_embeddings())
                else:
                    rank_zero_info("Read state dict from ckpt. ")
                    state_dict = ckpt

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            missing_keys = [k for k in missing_keys if "itc_teacher" not in k]
            rank_zero_info("missing_keys: {}".format(missing_keys))
            rank_zero_info("unexpected_keys: {}".format(unexpected_keys))

    def infer_text(
            self,
            batch,
            mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embed = self.backbone.text_embed(text_ids)
        text_padding_position = 1 - text_masks
        lffn_hiddens = self.backbone(
            textual_tokens=text_ids,
            text_padding_position=text_padding_position,
        )["encoder_out"]
        vlffn_hiddens = self.backbone_vl(
            src_tokens=None,
            token_embeddings=lffn_hiddens,
            encoder_padding_mask=text_padding_position,
            multiway_split_position=-1,
        )["encoder_out"]

        cls_feats = self.itc_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        cls_vlffn_feats = self.itc_vl_text_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "text_embed": text_embed,
        }

        return ret

    def infer_image(
            self,
            batch,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        img = batch[imgkey][0]
        if mask_image:
            image_masks = batch[f"{imgkey}_masks"][0].flatten(1)

            with torch.no_grad():
                img = self.visual_tokenizer.pre_process(img)
                quantize, embed_ind, _ = self.visual_tokenizer.encode(img)
                image_ids = embed_ind.view(img.shape[0], -1)

                image_labels = torch.full_like(image_ids, -100)
                bool_masked_pos = image_masks.to(torch.bool)
                image_labels[bool_masked_pos] = image_ids[bool_masked_pos]

        img_tensor = img_norm(img)
        vffn_hiddens = self.backbone(visual_tokens=img_tensor)["encoder_out"]
        vlffn_hiddens = self.backbone_vl(
            src_tokens=None,
            token_embeddings=vffn_hiddens,
            multiway_split_position=-1,
        )["encoder_out"]

        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        cls_vlffn_feats = self.itc_vl_image_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "image_feats": vffn_hiddens,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
        }

        return ret
