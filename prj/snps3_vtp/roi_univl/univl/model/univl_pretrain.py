# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import os
from typing import List

import cv2
import numpy as np
import torch
from torch import nn

from antmmf.modules.encoders import ModelForPretrainingMixin
from antmmf.utils.distributed import is_master
from .univl_base import UniVlBase


class UnivlPretrainHeads(nn.Module):
    """
    Pretraining head for univl model,
    Five pretraining tasks are involved:
    1. Masked language(ocr and caption) modelling - text-encoder
    2. Masked language(ocr and caption) modelling (with visual clues) - cross-modal
    3. Image Text matching - cross-modal encoder
    4. Object detection - cross-modal decoder
    5. Masked caption generation - cross-modal decoder
    """

    def __init__(self, config, detr_head=None, generation_head=None):
        super().__init__()

        self.detr_head = detr_head
        # Note: generation_head need to tie weights
        self.generation_head = generation_head
        # Note: mlm_head need to tie weights
        from antmmf.modules.build import build_transformer_head

        self.transformer_mlm = build_transformer_head(
            config.pretraining_heads.transformer_mlm
        )
        self.text_encoder_mlm = build_transformer_head(
            config.pretraining_heads.text_encoder_mlm
        )
        self.itm_head = build_transformer_head(config.pretraining_heads.itm)

    def forward(
        self,
        visual_enc_embed,
        ocr_enc_embed,
        cap_enc_embed,
        text_enc_embed,  # text encoder output
        dec_output,
        head_output,
        img_input,
        ocr_input,
        caption_output,
        sample_list,
    ):

        # task1: Hierarchical Masked Language Modelling for caption & ocr
        # task1-Hier1: text encoder mlm for caption & ocr
        text_encoder_mlm_targets = torch.cat(
            [sample_list.caption_lm_label_ids, sample_list.ocr_lm_label_ids], -1
        )
        text_encoder_mlm_output = self.text_encoder_mlm(
            encoder_output=text_enc_embed,
            targets=text_encoder_mlm_targets,
        )

        # task1-Hier2: multi-modal transformer encoder mlm for caption & ocr
        transformer_mlm_embed = torch.cat([cap_enc_embed, ocr_enc_embed], 1)
        # ignore mlm when image & text are not match
        transformer_mlm_targets = text_encoder_mlm_targets.clone().detach()
        transformer_mlm_targets[
            sample_list.itm_label == 0
        ] = transformer_mlm_targets.new_full((transformer_mlm_targets.size(1),), -1)
        transformer_mlm_output = self.transformer_mlm(
            encoder_output=transformer_mlm_embed,
            targets=transformer_mlm_targets,
        )

        # task2: Image & OCR Matching
        image_ocr_match_output = self.itm_head(
            encoder_output=cap_enc_embed, targets=sample_list.itm_label
        )

        # task3: detr image detection,
        # Skip images of no-bbox annotation
        if self.detr_head is not None:
            detr_predictions = head_output["detr_head"]
            device = detr_predictions["pred_logits"].device

            batch_num_boxes = sample_list.obj_target.num_box
            targets = []
            for batch_ind, num_box in enumerate(batch_num_boxes):
                if num_box.item() == 0:
                    b = torch.zeros([0, 4], dtype=torch.float32, device=device)
                    l = torch.zeros([0], dtype=torch.long, device=device)
                else:  # remove padding bboxes for matching
                    b = sample_list.obj_target.boxes[batch_ind][:num_box]
                    l = sample_list.obj_target.labels[batch_ind][:num_box]
                targets += [dict(boxes=b.to(device), labels=l.to(device))]

            detr_head_output = self.detr_head.get_loss_metric(detr_predictions, targets)
        else:
            detr_head_output = None

        # task4: caption generation,
        if self.generation_head is not None:
            # TODO： follow coco caption generation的处理, clipBERT
            gen_sentence_output = head_output["generation_head"]["generation_logits"]
            caption_ids = caption_output["generation_input_ids"]
            # -1 is ignored for mlm, SOS: [CLS]-101, EOS:[SEP]-102 PAD: 0
            shift_target_ids = torch.where(
                caption_output["generation_input_mask"].bool(),
                caption_ids.roll(-1, -1),
                -1,
            )
            # generation_source_len==2 means no caption provided, just ignore
            shift_target_ids[
                torch.where(
                    torch.tensor(caption_output["generation_source_len"]).long() == 2
                )
            ] = -1

            gen_head_output = self.generation_head.get_loss_metric(
                gen_sentence_output, shift_target_ids
            )
        else:
            gen_head_output = None

        output = [
            text_encoder_mlm_output,
            transformer_mlm_output,
            image_ocr_match_output,
            detr_head_output,
            gen_head_output,
        ]
        output = [out for out in output if out is not None]

        # for detr & generation valid visualization
        if (
            not self.training
            and self.detr_head is not None
            and self.generation_head is not None
        ):
            if not is_master():
                return output
            save_dir = "detr_results"
            vis_batch_ind = 0

            img_path = sample_list.image_abs_path[vis_batch_ind]
            img_name = os.path.basename(img_path)

            # detr result
            final_detr_result = self.detr_head.post_process(
                detr_predictions, sample_list.image_raw_size
            )
            drawed_img = cv2.imread(img_path)
            if len(final_detr_result) > 0:
                boxes = (
                    final_detr_result[vis_batch_ind]["boxes"].cpu().numpy()
                )  # x1y1x2y2
                labels = final_detr_result[vis_batch_ind]["labels"].cpu().numpy()
                scores = final_detr_result[vis_batch_ind]["scores"].cpu().numpy()

                keep = scores > 0.1
                drawed_img = self.vis_detr_results(
                    boxes[keep],
                    labels[keep],
                    scores[keep],
                    drawed_img,
                    img_size=(400, 400),
                )

            # generation
            if "generation_head" in head_output:
                generated_text = head_output["generation_head"]["generation_text"][
                    vis_batch_ind
                ]
                caption_text = caption_output["generation_text"][
                    vis_batch_ind
                ]  # TODO: get all captions
                drawed_img = self.add_text_results(
                    drawed_img, generated_text, ["".join(caption_text)]
                )

            save_path = self.get_vis_path(save_dir, img_name)
            cv2.imwrite(save_path, drawed_img)

        return output

    def get_vis_path(self, vis_dir, img_name):
        os.makedirs(vis_dir, exist_ok=True)
        meta_save_path = f"{vis_dir}/{img_name}"
        save_path = meta_save_path
        file_no = 0
        while os.path.exists(save_path):
            save_path = meta_save_path.rstrip(".jpg") + f"_{file_no}" + ".jpg"
            file_no += 1
        return save_path

    def add_text_results(
        self,
        input_canvas,
        generated_text: str,
        captions: List[str],
        text_color=(200, 200, 200),
    ):
        from PIL import Image

        canvas = Image.fromarray(np.zeros_like(input_canvas))
        generated_text = "generation:" + generated_text
        caption_text = "captions:" + ";".join(captions)
        from antmmf.utils.visual_utils.visualization_utils import (
            draw_multiple_line_text,
        )

        y_text = draw_multiple_line_text(canvas, generated_text, text_color, 0)
        draw_multiple_line_text(canvas, caption_text, text_color, y_text + 20)
        return np.concatenate([input_canvas, canvas], axis=1)

    def vis_detr_results(self, boxes, labels, scores, drawed_img, img_size=(400, 400)):
        from antmmf.utils.visual_utils.vis_utils import visualize

        drawed_img = visualize(
            drawed_img,
            boxes,
            classes=labels,
            confs=scores,
            label_format="x1y1x2y2",
            class_names=None,
            normalized=False,
            image_format="BGR",
            save_path=None,
        )
        drawed_img = cv2.resize(drawed_img, img_size, interpolation=cv2.INTER_AREA)
        return drawed_img


class UnivlForPretraining(nn.Module, ModelForPretrainingMixin):
    """
    Univl model that performs pretraining
    """

    def __init__(self, config):
        super(UnivlForPretraining, self).__init__()
        self.config = config
        self.enc = UniVlBase(config)
        self.pretrain_head = UnivlPretrainHeads(config, **self.enc.heads)

        # tie weights for MLM head
        if getattr(self.pretrain_head, "transformer_mlm", None):
            self.pretrain_head.transformer_mlm.tie_weights(
                self.enc.txt_embeddings.word_embeddings
            )
        if getattr(self.pretrain_head, "text_encoder_mlm", None):
            self.pretrain_head.text_encoder_mlm.tie_weights(
                self.enc.txt_embeddings.word_embeddings
            )
        # tie weights for generation
        if getattr(self.pretrain_head, "generation_head", None):
            self.pretrain_head.generation_head.tie_weights(
                self.enc.txt_embeddings.word_embeddings
            )

        self.init_weights(self.pretrain_head)

    def forward(
        self,
        img_input,
        caption_input,
        ocr_input=None,
        region_input=None,
        caption_output=None,
        sample_list=None,
    ):
        model_output = self.enc(
            img_input,
            caption_input,
            ocr_input,
            region_input,
            caption_output,
            sample_list=sample_list,
        )
        cap_enc_embed, ocr_enc_embed, visual_enc_embed, text_enc_embed = model_output[
            "enc_output"
        ]
        dec_output = model_output["dec_output"]
        head_output = model_output["head_output"]

        heads_output = self.pretrain_head(
            visual_enc_embed,
            ocr_enc_embed,
            cap_enc_embed,
            text_enc_embed,
            dec_output,
            head_output,
            img_input,
            ocr_input,
            caption_output,
            sample_list,
        )

        return heads_output
