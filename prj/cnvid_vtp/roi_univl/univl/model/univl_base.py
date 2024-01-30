# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import Dict, Any

import torch
from torch import nn

from antmmf.common.registry import registry
from antmmf.datasets.build import build_processors
from antmmf.modules.transformers.position_enhance import PositionEnhancedTransformer
from antmmf.modules.build import build_embedder
from antmmf.modules.encoders import VisualEncoder, TextEncoder
from antmmf.utils.general import check_required_keys


def split_encoder_output(encoder_embeds, position_range):
    """

    :param encoder_embeds: (b, sq_length, hidden)
    :param position_range:
    :return:
    """
    position_list = position_range.tolist()
    position_size = []
    last_i = None
    for i in position_list:
        if last_i is None:
            last_i = i
            position_size.append(i)
        else:
            position_size.append(i - last_i)
            last_i = i
    position_size.append(encoder_embeds.size(1) - position_list[-1])
    embed_splits = torch.split(encoder_embeds, position_size, dim=1)
    return embed_splits


class UniVlBase(nn.Module):
    """
    Modelling image, text and ocr with an E2E-VLP like model:
    E2E-VLP: End-to-End Vision-Language Pre-training Enhanced by Visual Learning

    Image, text and ocr  are projected to the same 2d-space with global_layout_embeddings
    before fed into transformer.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.writer = registry.get("writer")
        self.build_base()
        self.prune_model()

    def prune_model(self):
        self.ocr_processor = build_processors(self.config.processor).ocr_processor
        # prune params that are not used
        # textBert pooler's output currently not used
        self.text_encoder.pooler = nn.Identity()

    def build_base(self):
        # text encoder & embedding ===>
        self.text_encoder = TextEncoder(self.config.text_encoder).module
        self.txt_embeddings = self.text_encoder.embeddings

        # image encoder & embedding
        self.img_encoder = VisualEncoder(self.config.image_encoder).module

        # layout embedding: 1d & 2d
        self.global_layout_embeddings = build_embedder(
            self.config.global_layout_embeddings
        )
        # image & text input are already encoded by text_encoder and img_encoder
        del self.global_layout_embeddings.word_embeddings

        # cross-modal encoder
        self.transformer = PositionEnhancedTransformer(self.config.transformer_config)

        # transformer heads
        self.heads = nn.ModuleDict()
        if self.config.transformer_heads is not None:
            from antmmf.modules.build import build_transformer_head

            for head_name, head_config in self.config.transformer_heads.items():
                self.heads[head_name] = build_transformer_head(head_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer hidden dimension,
        # add a linear projection layer between the two
        if self.config.text_encoder.params.hidden_size != self.config.hidden_size:
            self.text_bert_out_linear = nn.Linear(
                self.config.text_encoder.params.hidden_size, self.config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def forward_img_encoder(self, img_input: Dict[str, Any]):
        assert (
            check_required_keys(
                img_input, required_keys=["image_data", "image_pad_mask"]
            )
            and self.img_encoder
        )
        output_dict = self.img_encoder(
            img_input["image_data"], image_mask=img_input["image_pad_mask"]
        )

        # grid_feats(torch.float32): [b, N, self.out_dim, h // 32, w // 32]
        # out_pos(torch.float32): [b, N, 2 * position_embedding.params.num_pos_feats, h // 32, w // 32]
        # grid_masks(torch.bool): [b, N, h // 32, w // 32]
        grid_feature, grid_mask, grid_pos = (
            output_dict["grid_feature"],
            output_dict["grid_mask"],
            output_dict["grid_pos"],
        )
        grid_shape = grid_feature.shape[-2:]

        assert grid_feature.size(1) == 1, "only support *ONE* image input"
        # assert grid_pos is not None

        visual_embed = grid_feature[:, 0].flatten(2).transpose(1, 2)  # b, seq_length, c
        visual_embed_pos = grid_pos and grid_pos[:, 0].flatten(2).transpose(
            1, 2
        )  # b, seq_length, c
        visual_mask = grid_mask[:, 0].flatten(
            1
        )  # b, seq_length:  false indicating pixel areas

        return_dict = dict(
            visual_embed=visual_embed,
            visual_embed_pos=visual_embed_pos,
            visual_mask=visual_mask,
            visual_grid_shape=grid_shape,
        )
        return return_dict

    def _build_transformer_input_for_caption(self, text_embed_dict):
        # caption text==>
        cap_embed = text_embed_dict["cap_text_enc_embed"]  # bsz, seq_length, hidden
        # caption text position_id, no 2d pos generated for caption.
        cap_boxes = torch.zeros(
            [cap_embed.size(0), cap_embed.size(1), 4], device=cap_embed.device
        ).long()
        # [0, 0, 0, 0] for padding bbox, [0, 0, 1000, 1000] for caption
        cap_boxes[:, :, 2:] = 1000
        cap_pos_ids = (
            torch.arange(cap_embed.size(1))
            .expand(cap_embed.shape[:2])
            .to(cap_embed.device)
        )
        cap_embed, (
            cap_embed_pos_1d,
            cap_embed_pos_2d,
        ) = self.global_layout_embeddings._calc_text_embeddings(
            cap_embed, bbox=cap_boxes, position_ids=cap_pos_ids, segment_id=0
        )

        # caption mask: need to convert to transformer' mask, False indicates remained tokens.
        cap_mask = (1 - text_embed_dict["caption_input_mask"]).bool()
        # caption text<==
        return cap_embed, cap_mask, cap_embed_pos_1d + cap_embed_pos_2d

    def _build_transformer_input_for_ocr(self, text_embed_dict):
        # ocr text==>
        ocr_embed = text_embed_dict["ocr_text_enc_embed"]  # bsz, seq_length, hidden
        ocr_pos_id = (
            torch.arange(ocr_embed.size(1))
            .expand(ocr_embed.shape[:2])
            .to(ocr_embed.device)
        )

        ocr_embed, (
            ocr_embed_pos_1d,
            ocr_embed_pos_2d,
        ) = self.global_layout_embeddings._calc_text_embeddings(
            ocr_embed,
            bbox=text_embed_dict["ocr_bboxes"].long(),
            position_ids=ocr_pos_id,
            segment_id=1,
        )
        ocr_mask = (1 - text_embed_dict["ocr_input_mask"]).bool()
        # ocr text<==
        return ocr_embed, ocr_mask, ocr_embed_pos_1d + ocr_embed_pos_2d

    def _build_transformer_input_for_visual(self, visual_embed_dict):
        # visual pixels==>
        # mask(bool): binary mask, ignore those positions whose value is True
        visual_embed, visual_embed_pos, visual_mask = (
            visual_embed_dict["visual_embed"],
            visual_embed_dict["visual_embed_pos"],
            visual_embed_dict["visual_mask"],
        )
        # visual_embed_pos shares same 2d position embedding space with ocr
        grid_h, grid_w = visual_embed_dict["visual_grid_shape"]
        bsz, device = visual_embed.size(0), visual_embed.device
        # 把batch中的每张在padding之前的图片划分为1000*1000的格子，并将下采样后的格子映射到1000*1000
        # 注意batch中每个样本图像格子大小都不一样
        # refer to LayoutLMV2:
        # https://huggingface.co/transformers/_modules/transformers/models/layoutlmv2/modeling_layoutlmv2.html#LayoutLMv2Model
        # step1: 获取每张图片下采样32倍后的大小
        v_mask = ~visual_mask.reshape([-1, grid_h, grid_w])
        b_grid_h, b_grid_w = v_mask.cumsum(1)[:, -1, 0], v_mask.cumsum(2)[:, 0, -1]

        # step2: 根据下采样真实大小计算映射系数
        grid_x_ratio, grid_y_ratio = 1000 / b_grid_w, 1000 / b_grid_h

        # step3: 将下采样后每个格子的坐标映射到 [0, 1000] 范围
        b_grid_y1 = torch.arange(grid_h).unsqueeze(0).repeat(bsz, 1).to(
            device
        ) * grid_y_ratio.unsqueeze(
            -1
        )  # bsz, grid_h
        b_grid_y2 = (torch.arange(grid_h) + 1).unsqueeze(0).repeat(bsz, 1).to(
            device
        ) * grid_y_ratio.unsqueeze(-1)

        b_grid_x1 = torch.arange(grid_w).unsqueeze(0).repeat(bsz, 1).to(
            device
        ) * grid_x_ratio.unsqueeze(
            -1
        )  # bsz, grid_h
        b_grid_x2 = (torch.arange(grid_w) + 1).unsqueeze(0).repeat(bsz, 1).to(
            device
        ) * grid_x_ratio.unsqueeze(-1)

        grid_bboxes = torch.stack(
            [
                b_grid_x1.unsqueeze(1)
                .repeat(1, grid_h, 1)
                .long(),  # b, grid_h, grid_w,
                b_grid_y1.unsqueeze(2)
                .repeat(1, 1, grid_w)
                .long(),  # b, grid_h, grid_w,
                b_grid_x2.unsqueeze(1).repeat(1, grid_h, 1).long(),  # grid_h, grid_w,
                b_grid_y2.unsqueeze(2).repeat(1, 1, grid_w).long(),  # grid_h, grid_w,
            ],
            -1,
        )  # b, grid_h, grid_w, 4
        grid_bboxes = grid_bboxes.flatten(1, 2)
        # step4: ignore positions where visual_mask is True, those positions exceed boundary
        grid_bboxes[visual_mask] = 0
        visual_pos_1d = (
            torch.arange(visual_embed.size(1))
            .expand(visual_embed.shape[:2])
            .to(visual_embed.device)
        )
        # directly use visual_embed_pos if it is not None
        visual_embed, (
            visual_embed_pos_1d,
            visual_embed_pos_2d,
        ) = self.global_layout_embeddings._calc_img_embeddings(
            visual_embed,
            position_2d_embeddings=visual_embed_pos,
            position_ids=visual_pos_1d,
            bbox=grid_bboxes,
        )

        # visual pixels<==
        return visual_embed, visual_mask, visual_embed_pos_1d + visual_embed_pos_2d

    def build_transformer_input(self, visual_embed_dict, text_embed_dict):
        # transformer input:
        # :param src(torch.float32): (b,seq_length, hidden_dim)
        # :param mask(torch.bool): (b, seq_length), used as src_key_padding_mask/memory_key_padding_mask,
        #                          When given a binary mask and a value is True, the corresponding value
        #                          on the attention layer will be ignored.
        # :param query_embed(torch.float32): (query_seq_length, hidden_dim)
        # :param pos_embed(torch.float32): (b, seq_length, hidden_dim), position embedding enhanced at attention.

        # Transformer's embed, position, mask for each input modality
        cap_embed, cap_mask, cap_embed_pos = self._build_transformer_input_for_caption(
            text_embed_dict
        )

        ocr_embed, ocr_mask, ocr_embed_pos = self._build_transformer_input_for_ocr(
            text_embed_dict
        )

        (
            visual_embed,
            visual_mask,
            visual_embed_pos,
        ) = self._build_transformer_input_for_visual(visual_embed_dict)

        embed = [cap_embed, ocr_embed, visual_embed]
        embed_pos = [cap_embed_pos, ocr_embed_pos, visual_embed_pos]
        mask = [cap_mask, ocr_mask, visual_mask]

        # modality range info
        position_range = torch.cumsum(
            torch.tensor([x.size(1) for x in embed[:-1]]), dim=0
        )
        return dict(
            embed=torch.cat(embed, 1),
            embed_pos=torch.cat(embed_pos, 1),
            mask=torch.cat(mask, 1),
            range=position_range,
        )

    def forward_text_encoder(self, caption_input=None, ocr_input=None):
        assert caption_input is not None or ocr_input is not None
        encoder_input_ids, encoder_input_mask = [], []
        encoder_position_ids, encoder_token_type_ids = [], []
        if caption_input is not None:
            encoder_input_ids.append(caption_input["caption_input_ids"])
            encoder_input_mask.append(caption_input["caption_input_mask"])
            caption_position_ids = torch.arange(
                caption_input["caption_input_ids"].size(1),
                dtype=torch.long,
                device=caption_input["caption_input_ids"].device,
            )
            encoder_position_ids.append(caption_position_ids)
            encoder_token_type_ids.append(
                torch.zeros_like(caption_input["caption_input_ids"])
            )
        if ocr_input is not None:
            encoder_input_ids.append(ocr_input["ocr_input_ids"])
            encoder_input_mask.append(ocr_input["ocr_input_mask"])

            # Individual position_ids for captions & ocr tokens, which is different with
            # BERT NSP pretraining task.
            ocr_position_ids = torch.arange(
                ocr_input["ocr_input_ids"].size(1),
                dtype=torch.long,
                device=ocr_input["ocr_input_ids"].device,
            )
            encoder_position_ids.append(ocr_position_ids)
            encoder_token_type_ids.append(torch.ones_like(ocr_input["ocr_input_ids"]))

        input_ids = torch.cat(encoder_input_ids, 1)
        input_mask = torch.cat(encoder_input_mask, 1)
        position_ids = (
            torch.cat(encoder_position_ids, 0).unsqueeze(0).expand_as(input_ids)
        )
        token_type_ids = torch.cat(encoder_token_type_ids, 1)

        # BERT attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions,
        # Note: we should use pooled_output here
        sequence_output, pooled_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        affined_sequence_output = self.text_bert_out_linear(sequence_output)
        affined_pooled_output = self.text_bert_out_linear(pooled_output)

        # split sequence_output for each modality
        caption_input_length = (
            0 if caption_input is None else caption_input["caption_input_ids"].size(1)
        )
        position_range = torch.tensor(
            [0, caption_input_length, caption_input_length],
            device=affined_sequence_output.device,
        )
        # text encoder output
        _, caption_text_embed, _, ocr_text_enc_embed = split_encoder_output(
            affined_sequence_output, position_range
        )

        return_dict = dict(
            cap_text_enc_embed=caption_text_embed,
            ocr_text_enc_embed=ocr_text_enc_embed,
            text_enc_embed=sequence_output,
            affined_sequence_output=affined_sequence_output,
            pooled_output=affined_pooled_output,
        )
        # additional info
        additional_info = dict()
        if caption_input is not None:
            additional_info["caption_input_mask"] = caption_input["caption_input_mask"]
        if ocr_input is not None:
            additional_info["ocr_input_mask"] = ocr_input["ocr_input_mask"]
            additional_info["ocr_bboxes"] = ocr_input["ocr_bboxes"]
        return_dict.update(additional_info)
        return return_dict

    def forward(
        self,
        img_input,
        caption_input,
        ocr_input=None,
        region_input=None,
        caption_output=None,
        sample_list=None,
    ):
        visual_embed_dict = self.forward_img_encoder(img_input)

        # text embedding for both ocr & caption
        text_embed_dict = self.forward_text_encoder(caption_input, ocr_input)

        # Note:  if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        # refer to:
        # https://github.com/microsoft/TAP/blob/352891f93c75ac5d6b9ba141bbe831477dcdd807/pythia/models/tap.py#L74-L83
        trans_input = self.build_transformer_input(visual_embed_dict, text_embed_dict)

        position_range = trans_input["range"]

        # Mask unrelated memory for object detection and caption generation.
        # =====>
        # if itm_label is 1: memory_key_padding_mask attends to all input sources(img & ocr & itm)
        # if itm_label is 0: memory_key_padding_mask only attend to image.
        itm_label = torch.ones([img_input["image_data"].size(0)], dtype=torch.long)
        if sample_list is not None:
            itm_label = sample_list.itm_label

        key_padding_mask_with_image = trans_input["mask"].clone().detach()
        key_padding_mask_with_image[:, : position_range[1]] = True

        # key_padding_mask_with_image_ocr = trans_input["mask"].clone().detach()
        # key_padding_mask_with_image_ocr[:, :position_range[0]] = True

        memory_key_padding_mask = trans_input["mask"].clone().detach()
        memory_key_padding_mask[itm_label == 0] = key_padding_mask_with_image[
            itm_label == 0
        ]
        # <=====

        # 'ModuleDict' object has no attribute 'get'
        generation_head = (
            self.heads["generation_head"] if "generation_head" in self.heads else None
        )
        model_output = self.transformer(
            trans_input["embed"],
            src_key_padding_mask=trans_input["mask"],
            memory_key_padding_mask=memory_key_padding_mask,
            src_pos=trans_input["embed_pos"],
            text_embedding=self.txt_embeddings,
            tokenizer=self.ocr_processor._tokenizer,
            generation_head=generation_head,
            dec_step_num=20,
            caption_output=caption_output,
        )

        # Encoder output
        # cap: CLS .. SEP <PAD> <PAD>
        # ocr: .....SEP <PAD> <PAD>
        # visual: .....
        cap_enc_embed, ocr_enc_embed, visual_enc_embed = split_encoder_output(
            model_output["memory"], position_range
        )
        enc_dec_output = dict(
            enc_output=(
                cap_enc_embed,
                ocr_enc_embed,
                visual_enc_embed,
                text_embed_dict["text_enc_embed"],
            ),
            dec_output=model_output["decoder"],
        )
        transformer_output = dict(head_output=self.forward_heads(**enc_dec_output))
        transformer_output.update(enc_dec_output)
        return transformer_output

    def forward_heads(self, enc_output, dec_output):
        heads_out = {}
        if "detr" in dec_output:  # have detr head
            heads_out["detr_head"] = self.heads["detr_head"].forward_head(
                decoder_output=dec_output["detr"]
            )
        if "generation" in dec_output:
            heads_out["generation_head"] = {}
            heads_out["generation_head"]["generation_ids"] = dec_output["generation"][
                "generation_ids"
            ]
            heads_out["generation_head"]["generation_text"] = dec_output["generation"][
                "generation_text"
            ]
            heads_out["generation_head"]["generation_logits"] = dec_output[
                "generation"
            ]["generation_logits"]
        return heads_out
