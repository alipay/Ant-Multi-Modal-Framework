# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.
# roi_model: modelling relationship among modalities of Region-OCR-Image.

import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import KLDivLoss
from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import (
    BertLMPredictionHead,
    BertPredictionHeadTransform,
)

from antmmf.common import constants
from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from antmmf.modules.embeddings import (
    ImageBertEmbeddings,
    LayoutLMEmbeddings,
    VisualLayoutEmbeddings,
)
from antmmf.modules.encoders import ModelForPretrainingMixin
from antmmf.modules.encoders.multimodal_bert_clf import MultimodalBertClf
from antmmf.modules.encoders import VisualEncoder
from antmmf.utils.general import get_transformer_model_vocab_path


class ROIBertEncoder(nn.Module):
    """
    Modelling relationships among R(regions) O(OCR bbox) I(Image),
    ROIBertEncoder is an extension of MultimodalBertEncoder, which jointly models not only
    image and its captions, but also 2d-regions(detection and OCR regions) that are extracted
    from image in advance.  Regions' 2d-position using layoutLM encoding style.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.writer = registry.get("writer")

        # text encoder ===>
        # from antmmf.modules.encoders import TextEncoder
        # self.text_encoder = TextEncoder(self.config.text_encoder)

        # text encoder <===

        if self.config.get(constants.PRETRAINED_STR) is False:
            bert_config = AutoConfig.for_model(
                model_type=config.get("model_type", "bert"), **config
            )
            bert = AutoModel.from_config(bert_config)
            warnings.warn("random initialization for {}".format(config.bert_model_name))
        else:
            pretrained_model_path = get_transformer_model_vocab_path(
                config.bert_model_name
            )
            self.writer.write(
                "loaded pretrained model for {} from {}".format(
                    config.bert_model_name, pretrained_model_path
                )
            )
            bert = AutoModel.from_pretrained(pretrained_model_path)

        self.txt_embeddings = bert.embeddings

        self.img_encoder = VisualEncoder(config.image_encoder)
        self.img_embeddings = ImageBertEmbeddings(
            config,
            img_hidden_sz=self.img_encoder.module.out_dim,
            embeddings=self.txt_embeddings,
        )

        # separate-token between image tokens, default as [SEP]
        self.inter_token_id = config.get(constants.INTER_TOKEN_ID_STR, 102)

        # number of separate-tokens between image tokens, default as 1 for compatibility.
        self.img_token_interval = config.get(constants.IMG_TOKEN_INTERVAL_STR, 1)

        self.encoder = bert.encoder
        self.pooler = bert.pooler

        # build

        # shared embedding for word/position/type
        self.ocr_embeddings = LayoutLMEmbeddings(
            config.layoutlm_embedding_config, embeddings=self.txt_embeddings
        )

        if not self.config.get("use_grid_feature", False):
            # shared embedding for region embedding
            self.region_embeddings = VisualLayoutEmbeddings(
                config.region_embedding_config,
                embeddings=self.ocr_embeddings,
            )

    def build_multimodal_embedding_input(
        self,
        input_txt,
        attention_mask,
        text_token_type,
        cls_id,
        sep_id,
        input_img,
        img_mask=None,
        img_token_type=0,
    ):
        if input_img.ndim == 4:  # bsz, channel, height, width
            bsz, num_images = input_img.size(0), 1
        elif input_img.ndim == 5:  # bsz, num_images, channel, height, width
            bsz, num_images = input_img.size(0), input_img.size(1)
        else:
            raise Exception(f"unknown input image shape:{input_img.shape}")
        length = self.config.image_encoder.params.num_output_features
        image_seq_length = (
            num_images * length + (num_images - 1) * self.img_token_interval + 2
        )

        img_modality_mask = (
            torch.ones(bsz, image_seq_length).long().to(device=attention_mask.device)
        )
        if img_mask is not None:  # ignore padding visual tokens
            num_visual_tokens = img_mask.sum(axis=1)
            attend_visual_tokens = (
                num_visual_tokens * length
                + (num_visual_tokens - 1) * self.img_token_interval
                + 2
            )
            for idx, attend_visual in enumerate(attend_visual_tokens):
                img_modality_mask[idx, attend_visual:] = 0

        attention_mask = torch.cat(
            [
                img_modality_mask,
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # As of now, Pytorch doesn't support calling self.parameters() within DataParallel, which causes the current
        # issue. Even after fixing that, which was straightforward, Pytorch also doesn't support calling
        # self.ParameterList and self.ParameterDict, which will cause another issue. As Pytorch is moving people
        # away from DataParallel, they are unlikely to fix this anytime soon on their end.
        # In the meantime, we could use DistributedDataParallel instead.
        # see detail:
        # https://github.com/huggingface/transformers/issues/8145#issuecomment-721044942

        # add support for data parallel, and drop support for fp16
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (torch.LongTensor(bsz, image_seq_length).fill_(img_token_type)).to(
            device=input_img.device
        )

        self.img_encoder.check_input(input_img)
        img = self.img_encoder(input_img)
        # B x config.num_output_features x 2048 or B x num_images x
        # config.num_output_features x 2048
        assert len(img.size()) in [
            3,
            4,
        ], f" processed image feature dimension {img.shape}"
        if img.size() == 3:  # [BxNx2048]
            # B x num_images x config.num_output_features x 2048
            img = img.unsqueeze(1)
        img_embed_out = self.img_embeddings(
            img,
            img_tok,
            cls_id,
            sep_id,
            inter_token_id=self.inter_token_id,
            img_token_interval=self.img_token_interval,
        )

        if isinstance(text_token_type, int):
            text_token_type = torch.zeros_like(input_txt).fill_(text_token_type)
        txt_embed_out = self.txt_embeddings(input_txt, text_token_type)
        embedding_inputs = {
            "img_embed_out": img_embed_out,
            "txt_embed_out": txt_embed_out,
            "input_attention_mask": extended_attention_mask,
            "image_seq_length": image_seq_length,
            "attention_mask": attention_mask,
        }
        return embedding_inputs

    def forward(self, img_input, caption_input, ocr_input=None, region_input=None):
        """
        This combines image with text to have the following inputs
        observation: <SEP image SEP> <regions> SEP <OCR SEP> SEP <caption SEP>
        position:    <0,1,...,len(Image-Feature)> <0,1,..., len(regions)> <0,1,..., len(ocr)> <0,1,...len(caption)>
        Token types:        1             1             0              0
        attention_mask: <1,1,...1> <1,1,...,0,0> 1  <1,1,...,0,0> 1 <1,1,...,0>
        Regions, OCRs and captions are padded with zero on attention mask.
        """
        batch_size = caption_input["caption_input_ids"].shape[0]
        current_device = caption_input["caption_input_ids"].device

        sep_embed = self.txt_embeddings.word_embeddings(
            torch.LongTensor((batch_size))
            .fill_(self.inter_token_id)
            .to(device=current_device)
        ).unsqueeze(1)
        sep_mask = (
            torch.zeros((batch_size, 1), dtype=torch.long)
            .fill_(1)
            .to(device=current_device)
        )

        # image and caption input
        embedding_inputs = self.build_multimodal_embedding_input(
            caption_input["caption_input_ids"],
            caption_input["caption_input_mask"],
            0,  # text_token_type, 0
            torch.tensor(caption_input["caption_cls_id"], dtype=torch.long).to(
                device=current_device
            ),
            torch.tensor(caption_input["caption_sep_id"], dtype=torch.long).to(
                device=current_device
            ),
            img_input["image_data"],
            img_mask=img_input["image_mask"],
            img_token_type=1,
        )

        attention_mask, image_seq_length = (
            embedding_inputs["attention_mask"],
            embedding_inputs["image_seq_length"],
        )
        img_embed_out, txt_embed_out = (
            embedding_inputs["img_embed_out"],
            embedding_inputs["txt_embed_out"],
        )
        img_attention_mask, txt_attention_mask = (
            attention_mask[:, :image_seq_length],
            attention_mask[:, image_seq_length:],
        )

        # image input
        multi_embedding_inputs = [img_embed_out]
        multi_attention_masks = [img_attention_mask]

        if region_input:
            # region embeddings
            region_embed_output = self.region_embeddings(
                region_input["region_feature"],
                region_input["region_location"],
                token_type_ids=1,
            )
            multi_embedding_inputs += [region_embed_output]
            multi_attention_masks += [region_input["region_mask"].to(torch.long)]

        multi_embedding_inputs += [sep_embed]
        multi_attention_masks += [sep_mask]

        if ocr_input:
            # ocr embeddings
            ocr_embed_output = self.ocr_embeddings(
                ocr_input["ocr_input_ids"], ocr_input["ocr_bboxes"], token_type_ids=0
            )
            multi_embedding_inputs += [ocr_embed_output]
            multi_attention_masks += [ocr_input["ocr_input_mask"]]

        multi_embedding_inputs += [sep_embed]
        multi_attention_masks += [sep_mask]

        # caption input
        multi_embedding_inputs += [txt_embed_out]
        multi_attention_masks += [txt_attention_mask]

        """
        inputs for multimodal encoders:
        Embedding inputs: <SEP image SEP> <regions> SEP <OCR SEP> SEP <caption SEP>
        Token types:             1            1             0          0
        """

        encoder_input = torch.cat(multi_embedding_inputs, 1)  # Bx(TEXT+IMG)xHID

        attention_mask = torch.cat(multi_attention_masks, dim=1).to(torch.long)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # add support for data parallel, and drop support for fp16
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(
            encoder_input,
            extended_attention_mask,
            head_mask=[None] * len(self.encoder.layer),
        )

        pooled_out = self.pooler(encoded_layers[0])

        # calculate modality tokens' range: [,)
        image_start = 1
        image_end = img_embed_out.size(1) - 1

        region_start = image_end + 1  # +sep
        if region_input:
            region_end = region_start + region_embed_output.size(1)
        else:
            region_end = region_start

        ocr_start = region_end + 1  # +sep
        if ocr_input:
            ocr_end = ocr_start + ocr_embed_output.size(1)
        else:
            ocr_end = ocr_start

        caption_start = ocr_end + 1  # +sep
        caption_end = caption_start + txt_embed_out.size(1)

        modality_range = {
            "image": [image_start, image_end],
            "region": [region_start, region_end],
            "ocr": [ocr_start, ocr_end],
            "caption": [caption_start, caption_end],
        }

        outputs = (
            pooled_out,  # pooled output
            encoded_layers[0],  # sentence_out
            modality_range,
        )
        return outputs


# ROI components
class BertImagePredictionHead(nn.Module):
    """
    VilBERT style Region-Head that performs region classification task.
    This implementation refers to:
    https://github.com/e-bug/volta/blob/9e5202141920600d58a9c5c17519ca453795d65d/volta/encoders.py#L720
    """

    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.region_kl_fc_dim)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)


class ROIPreTrainingHeads(nn.Module):
    """
    Pretraining head for ROI model that models relationships among R(regions) O(OCR bbox) I(Image).
    Three tasks are involved:
    1. Image Text(ocr) matching
    2. Masked region(detection) classification( with linguistic clues)
    3. Masked language(ocr) modelling( with visual clues)
    """

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.image_predictions = BertImagePredictionHead(config)

    def forward(self, sequence_output, pooled_output, modality_range):
        # prediction for MLM task
        sequence_output_v = sequence_output[
            :, modality_range["region"][0] : modality_range["region"][1], :
        ]
        sequence_output_t = sequence_output[
            :, modality_range["ocr"][0] : modality_range["ocr"][1], :
        ]
        prediction_scores_t = self.predictions(sequence_output_t)
        # vision prediction
        prediction_scores_v = self.image_predictions(sequence_output_v)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores_t, prediction_scores_v, seq_relationship_score


class RoiBertForPretraining(nn.Module, ModelForPretrainingMixin):
    """
    ROI model that performs pretraining
    """

    def __init__(self, config):
        super(RoiBertForPretraining, self).__init__()
        self.config = config
        self.enc = ROIBertEncoder(config)
        self.clf = ROIPreTrainingHeads(config)
        self.init_weights(self.clf)
        self.clf.predictions.decoder.weight = (
            self.enc.txt_embeddings.word_embeddings.weight
        )

    def forward(self, img_input, caption_input, ocr_input=None, region_input=None):
        pooled_output, sequence_output, modality_range = self.enc(
            img_input, caption_input, ocr_input, region_input
        )
        prediction_scores_t, prediction_scores_v, seq_relationship_score = self.clf(
            sequence_output, pooled_output, modality_range
        )
        ret_dict = {
            "prediction_scores_t": prediction_scores_t,
            "prediction_scores_v": prediction_scores_v,
            "seq_relationship_score": seq_relationship_score,
            "modality_range": modality_range,
        }
        return ret_dict


class RoiBertClf(MultimodalBertClf):
    """
    ROI model that performs classification task
    """

    def __init__(self, config):
        super(RoiBertClf, self).__init__(config, modal_encoder=ROIBertEncoder)

    def forward(self, img_input, caption_input, ocr_input=None, region_input=None):
        pooled_output, sequence_output, modality_range = self.enc(
            img_input, caption_input, ocr_input, region_input
        )
        return {"logits": self.clf(pooled_output)}


@registry.register_model("roi_model")
class ROIModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = RoiBertForPretraining(self.config)
            # unmasked token labels are set as -1
            self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
        else:
            self.model = RoiBertClf(self.config)

    def group_inputs(self, sample_list):
        key_input_group = {"ocr": {}, "caption": {}, "region": {}, "image": {}}
        for sample_key in sample_list.keys():
            for input_key in key_input_group.keys():
                if sample_key.startswith(input_key):
                    key_input_group[input_key][sample_key] = sample_list[sample_key]
        return key_input_group

    def forward(self, sample_list, *args, **kwargs):
        key_input_pairs = self.group_inputs(sample_list)

        model_output = self.model(
            key_input_pairs["image"],
            key_input_pairs["caption"],
            key_input_pairs["ocr"],
            key_input_pairs["region"],
        )

        if self.config.training_head_type == "pretraining":
            output = {"losses": {}, "metrics": {}}
            # step1: calculate itm loss & metrics
            if key_input_pairs.get("image") != {} and key_input_pairs.get("ocr") != {}:
                itm_loss, itm_acc = self.cal_itm_loss(sample_list, model_output)
                output["losses"]["itm_loss"] = itm_loss
                output["metrics"]["itm_acc"] = itm_acc

            # step2: calculate masked_labels for OCR input
            if key_input_pairs.get("ocr") != {}:
                masked_lm_loss, masked_lm_acc = self.cal_mlm(sample_list, model_output)
                output["losses"]["masked_lm_loss"] = masked_lm_loss
                output["metrics"]["masked_lm_acc"] = masked_lm_acc

            # step3: calculate region loss
            if key_input_pairs.get("region") != {}:
                region_loss, mrc_acc = self.cal_region_loss(sample_list, model_output)
                output["losses"]["region_loss"] = region_loss
                output["metrics"]["mrc_acc"] = mrc_acc
        else:
            output = (
                {"logits": model_output}
                if isinstance(model_output, (torch.Tensor,))
                else model_output
            )

        return output

    def cal_region_loss(self, sample_list, model_output):
        # refer to: https://github.com/e-bug/volta/blob/9e5202141920600d58a9c5c17519ca453795d65d/volta/losses.py#L16
        image_target = sample_list["region_cls"]
        region_to_predict = sample_list["region_to_predict"]
        prediction_scores_v = model_output["prediction_scores_v"]
        loss = KLDivLoss(reduction="none")(
            F.log_softmax(prediction_scores_v, dim=2), image_target
        )
        region_loss = torch.sum(
            loss * (region_to_predict == 1).unsqueeze(2).float()
        ) / max(torch.sum((region_to_predict == 1)), 1)

        # MRC  acc
        with torch.no_grad():
            mrc_res = (
                (image_target.argmax(-1) == prediction_scores_v.argmax(-1)).to(
                    torch.float
                )
                * (region_to_predict == 1)
            ).sum()

            mrc_acc = mrc_res / max(torch.sum((region_to_predict == 1)), 1)

        return region_loss, mrc_acc

    def cal_itm_loss(self, sample_list, model_output):
        itm_label = sample_list.itm_label
        itm_logits = model_output["seq_relationship_score"]  # N, 2
        batch_size = itm_logits.size(0)
        itm_loss = self.loss_fct(itm_logits, itm_label).sum() / batch_size
        # itm acc
        with torch.no_grad():
            itm_res = itm_logits.argmax(-1) == itm_label
            itm_acc = itm_res.sum() / (itm_res.size(0) + 1e-6)
        return itm_loss, itm_acc

    def cal_mlm(self, sample_list, model_output):
        ocr_masked_labels = sample_list["ocr_" + constants.LM_LABEL_IDS_STR].view(-1)
        masked_lm_logits = model_output[
            "prediction_scores_t"
        ]  # N, seq_length, vocab_size
        pred_ocr_logit = masked_lm_logits.reshape(-1, self.config.vocab_size)
        mlm_loss = self.loss_fct(pred_ocr_logit, ocr_masked_labels).sum() / max(
            torch.sum(ocr_masked_labels != -1), 1
        )
        # MLM acc
        with torch.no_grad():
            pred_res = (pred_ocr_logit.argmax(-1) == ocr_masked_labels)[
                torch.where(ocr_masked_labels != -1)
            ]
            mlm_acc = pred_res.sum() / (pred_res.size(0) + 1e-6)
        return mlm_loss, mlm_acc
