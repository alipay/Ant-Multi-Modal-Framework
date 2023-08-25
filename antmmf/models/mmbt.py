# This is imported from
# https://github.com/facebookresearch/mmbt/
# to the AntMMF framework

import torch
from torch.nn import CrossEntropyLoss

from antmmf.common import constants
from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from antmmf.modules.encoders.multimodal_bert_clf import MultimodalBertClf
from antmmf.modules.encoders.multimodal_bert_for_pretraining import (
    MultimodalBertForPretraining,
)


@registry.register_model("mmbt")
class MMBT(BaseModel):
    # 所有模型的初始化，需要这个config
    # 这个config应该包含所有模型需要用到的参数设置
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def strip_head(cls, state_dict):
        head_key = ["model.clf.weight", "model.clf.bias"]
        for key in head_key:
            if key in state_dict:
                del state_dict[key]
        return state_dict

    # 在这里定义所需要的模块
    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = MultimodalBertForPretraining(self.config)
            # unmasked token labels are set as -1
            self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
        else:
            self.model = MultimodalBertClf(self.config)

    def build_for_test(self):
        # Avoid loading pretrained models for inference, just loading
        # model weights provided by user
        self.config.pretrained = False
        self.config.image_encoder.params.pretrained = False
        self.model = MultimodalBertClf(self.config)

    def forward_graph(self, text, mask, segment, image, cls_id, sep_id, image_mask):
        output_dict = self.model(text, mask, segment, image, cls_id, sep_id, image_mask)
        return output_dict

    # MMF模型的forward函数的输入是一个词典，包含了有关输入特征的信息
    def forward(self, sample_list, *args, **kwargs):
        text = sample_list["text"]  # [batch_size, text_fields*max_length]
        mask = sample_list["mask"]  # [batch_size, text_fields*max_length]

        # [batch_size, num_images, c, h, w] or [batch_size, c, h, w]
        image = sample_list["image"]

        # [batch_size, num_images]
        image_mask = sample_list.get("image_mask", None)
        # [batch_size, text_fields*max_length]
        segment = sample_list["segment"]
        cls_id = sample_list["cls_id"]  # [batch_size,]
        sep_id = sample_list["sep_id"]  # [batch_size,]

        #  support normal classification or hierarchical classification
        #  Required model config for hierarchical classification:
        #     classifier_type: hier_classifier
        #     hier_label_schema: *hier_label_schema
        #
        #  Required model config for normal classification:
        #     classifier_type: mlp_classifier
        #     num_labels: 80 # num of total labels
        output = self.forward_graph(
            text, mask, segment, image, cls_id, sep_id, image_mask
        )

        if self.config.training_head_type == "pretraining":

            masked_lm_logits = output["prediction_scores"]  # N, seq_length, vocab_size
            itm_logits = output["seq_relationship_score"]  # N, 2
            image_seq_length = output["image_seq_length"]
            batch_size = text.size(0)

            # step1: calculate masked_labels and itm_label loss
            text_masked_labels = sample_list[constants.LM_LABEL_IDS_STR]
            itm_label = sample_list.itm_label
            pred_mlm_logit = masked_lm_logits[:, image_seq_length:, :].reshape(
                -1, self.config.vocab_size
            )
            target_mlm_label = text_masked_labels.view(-1)
            mlm_loss = self.loss_fct(pred_mlm_logit, target_mlm_label).sum() / max(
                torch.sum(target_mlm_label != -1), 1
            )
            itm_loss = self.loss_fct(itm_logits, itm_label).sum() / batch_size
            # step2: compute accuracy for pre-training monitoring
            with torch.no_grad():
                # MLM acc
                pred_res = (pred_mlm_logit.argmax(-1) == target_mlm_label)[
                    torch.where(target_mlm_label != -1)
                ]
                mlm_acc = pred_res.sum() / (pred_res.size(0) + 1e-6)
                # itm acc
                itm_res = itm_logits.argmax(-1) == itm_label
                itm_acc = itm_res.sum() / (itm_res.size(0) + 1e-6)

            output = {
                "losses": {"masked_lm_loss": mlm_loss, "itm_loss": itm_loss},
                "metrics": {"masked_lm_acc": mlm_acc, "itm_acc": itm_acc},
            }
        return output

    def get_adv_parameters(self):
        return self.model.get_adv_parameters()

    def init_adv_train(self):
        r"""
        during adversarial training, the model needs to be in eval mode
        to avoid randomness such as drop-out
        """
        self.eval()


@registry.register_model("AttriMMBT")
class AttriMMBT(MMBT):
    def get_optimizer_parameters(self, config):
        # code adapted from
        # https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L313
        no_decay = ["bias", "LayerNorm.weight"]
        cnn_params, bert_params = [], []
        for name, param in self.named_parameters():
            if "img_encoder" in name:  # cnn params
                cnn_params.append((name, param))
            else:  # bert params
                bert_params.append((name, param))

        adamW_grouped_parameters = [
            {
                "params": [
                    p for n, p in bert_params if not any(nd in n for nd in no_decay)
                ],
                "name": "adamW_grouped_parameters",
            },
            {
                "params": [
                    p for n, p in bert_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "name": "adamW_grouped_parameters",
            },
        ]
        sgd_grouped_parameters = [
            {
                "params": [
                    p for n, p in cnn_params if not any(nd in n for nd in no_decay)
                ],
                "name": "sgd_grouped_parameters",
            }
        ]
        return {"AdamW": adamW_grouped_parameters, "sgd": sgd_grouped_parameters}
