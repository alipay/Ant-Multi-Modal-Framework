# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torchvision.models as models
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
)

from antmmf.common import Configuration
from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from antmmf.modules.embeddings import LayoutLMEmbeddings
from antmmf.utils.file_io import load_yaml
from antmmf.utils.general import get_transformer_model_vocab_path
from antmmf.utils.general import get_user_model_resource_path
from antmmf.utils.timer import Timer


class LayoutLMModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.

    """

    def __init__(self, config):
        super(LayoutLMModel, self).__init__(config)

        self.embeddings = LayoutLMEmbeddings(
            Configuration(config.to_dict())
        )  # 对应bert中embedding
        """
        pytorch 1.5 has a bug with transformer library, see issue at:
        https://github.com/huggingface/transformers/issues/3936
        downgrade to pytorch 1.4 if you encounter error
        """
        self.encoder = BertEncoder(
            config
        )  # 对应transformer_model   12层layer 每层12个attention head
        self.pooler = BertPooler(
            config
        )  # see issue: https://github.com/google-research/bert/issues/196

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if attention_mask is None:  # attention_mask = torch.Size([9, 512])
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:  # token_type_ids = torch.Size([9, 512])
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension
        # here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # extended_attention_mask = torch.Size([9, 1, 1, 512])
        # attention_mask = torch.Size([9, 512])
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (
            1.0 - extended_attention_mask
        ) * -10000.0  # 0->-10000, will be 0 after softmax

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x
        # num_heads x seq_length x seq_length]
        if head_mask is not None:  # head_mask = None
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            # self.config.num_hidden_layers = 12
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(  # input_ids= torch.Size([9, 512])    bbox = torch.Size([9, 512, 4])
            input_ids,
            bbox,
            position_ids=position_ids,
            token_type_ids=token_type_ids,  # 输出 torch.Size([9, 512, 768])
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,  # [torch.Size([9, 512, 768])]
        )
        sequence_output = encoder_outputs[0]  # torch.Size([9, 512, 768])
        pooled_output = self.pooler(
            sequence_output
        )  # 取512中的第0个作为输出 进行dense和activation   # torch.Size([9, 768])

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here

        # sequence_output, pooled_output, (hidden_states),
        # (attentions) (torch.Size([9, 512, 768]), torch.Size([9, 768]))
        return outputs


class LayoutLMForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.

    """

    def __init__(self, config):
        super(LayoutLMForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels  # 53

        self.bert = LayoutLMModel(config)
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )  # hidden_dropout_prob =0.1
        self.classifier = nn.Linear(
            config.hidden_size, config.num_labels
        )  # hidden_size=768  num_labels = 53

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]  # torch.Size([16, 512, 768])

        sequence_output = self.dropout(sequence_output)
        # logits = torch.Size([16, 512, 13])
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class LayoutLMForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.

    """

    def __init__(self, config):
        super(LayoutLMForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = LayoutLMModel(config)
        self.bert_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.img_hidden_size = config.img_hidden_size

        if self.img_hidden_size > 0:
            self.resnet50 = models.resnet50(
                pretrained=config.to_dict().get("pretrained_resnet", True)
            )
            self.resnet50.fc = nn.Linear(
                self.resnet50.fc.in_features, self.config.num_labels
            )
            self.extract_result = FeatureExtractor(self.resnet50, ["avgpool", "fc"])
            self.concat_classifier = nn.Linear(
                config.hidden_size + config.img_hidden_size, self.config.num_labels
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(
            config.hidden_size, self.config.num_labels
        )  # 新增参数img_hidden_size = 2048
        self.activation = nn.Softmax(dim=-1)

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        image_data=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,  # torch.Size([9, 512])
            bbox=bbox,  # bbox
            attention_mask=attention_mask,
            # torch.Size([9, 512])  padding的全部为0
            token_type_ids=token_type_ids,  # torch.Size([9, 512]) 全是0
            position_ids=position_ids,  # None
            head_mask=head_mask,  # None
        )  # output = (torch.Size([9, 512, 768]), torch.Size([9, 768]))
        # 其中output[1] 是output[0] 取512中的第0个作为输出 进行dense和activation

        # when fine-tuned on downstream tasks, bert use [cls] for sentence
        # encoding
        bert_pooled_output = outputs[1]

        bert_pooled_output = self.bert_dropout(bert_pooled_output)

        # torch.Size([9, 53])
        bert_logits = self.classifier(bert_pooled_output)

        if self.img_hidden_size > 0:
            img_features, img_logits = self.extract_result(image_data)
            img_features = img_features.view(img_features.size(0), -1)
            concat_pooled_output = torch.cat((bert_pooled_output, img_features), -1)
            concat_pooled_output = self.dropout(concat_pooled_output)
            concat_logits = self.concat_classifier(concat_pooled_output)

            # ----------------------图像不存在，则用bert_logits代替-----------------------
            img_modality_mask = torch.eq(
                torch.sum(image_data.view(image_data.size(0), -1), axis=1), 0
            )
            concat_logits[img_modality_mask] = bert_logits[img_modality_mask]
            img_logits[img_modality_mask] = bert_logits[img_modality_mask]
            # -------------------------------------------------------------------------
            outputs = (self.activation(concat_logits),) + outputs[2:]
            return {
                "bert_logits": bert_logits,
                "img_logits": img_logits,
                "logits": concat_logits,
                "prob": self.activation(concat_logits),
                "outputs": outputs,
            }
        else:
            outputs = (self.activation(bert_logits),) + outputs[
                2:
            ]  # add hidden states and attention if they are here
            return {
                "logits": bert_logits,
                "prob": self.activation(bert_logits),
                "outputs": outputs,
            }


class CascadeLayoutlmForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, label_map=None, pretrained_roberta_path=None):
        assert label_map is not None, "label_map not set"
        assert pretrained_roberta_path is not None, "pretrained_roberta_path not set"

        super(CascadeLayoutlmForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = LayoutLMModel(config)
        self.bert_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.img_hidden_size = config.img_hidden_size

        # load roberta model from indicated model_dir
        self.roberta_model = BertModel.from_pretrained(pretrained_roberta_path)
        self.roberta_config = self.roberta_model.config
        self.roberta_dropout = nn.Dropout(self.roberta_config.hidden_dropout_prob)

        if self.img_hidden_size > 0:
            # will automatically load from $TORCH_HOME/checkpoints
            self.resnet50 = models.resnet50(pretrained=True)
            self.resnet50.fc = nn.Linear(
                self.resnet50.fc.in_features, self.config.num_labels
            )
            self.extract_result = FeatureExtractor(self.resnet50, ["avgpool", "fc"])
            self.concat_classifier = nn.Linear(
                config.hidden_size
                + config.img_hidden_size
                + self.roberta_config.hidden_size,
                self.config.num_labels,
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.img_hidden_size > 0:
            self.classifier = nn.Linear(
                config.hidden_size, self.config.num_labels
            )  # 新增参数img_hidden_size = 2048
        else:  # no use
            self.classifier = nn.Linear(
                config.hidden_size + self.roberta_config.hidden_size,
                self.config.num_labels,
            )  # 新增参数img_hidden_size = 2048

        self.roberta_classifier = nn.Linear(
            self.roberta_config.hidden_size, self.config.num_labels
        )

        self.activation = nn.Softmax(dim=-1)

        # ---------------------增加二级分类器-------------------
        label_map_path = get_user_model_resource_path(label_map)
        self.label_map = load_yaml(label_map_path)["second_level_label"]

        self.sec_num_labels = (
            max([self.label_map[x] for x in list(self.label_map.keys())]) + 1
        )

        if self.img_hidden_size > 0:
            self.sec_img_classifier = nn.Linear(
                self.img_hidden_size, self.sec_num_labels
            )
            self.sec_concat_classifier = nn.Linear(
                config.hidden_size
                + config.img_hidden_size
                + self.roberta_config.hidden_size,
                self.sec_num_labels,
            )
        self.sec_bert_classifier = nn.Linear(config.hidden_size, self.sec_num_labels)
        self.sec_roberta_classifier = nn.Linear(
            self.roberta_config.hidden_size, self.sec_num_labels
        )
        # ----------------------------------------------------

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        image_data=None,
    ):

        # roberta encoding
        roberta_outputs = self.roberta_model(
            input_ids=input_ids,  # torch.Size([9, 512])
            attention_mask=attention_mask,
            # torch.Size([9, 512])  padding的全部为0
            token_type_ids=token_type_ids,  # torch.Size([9, 512]) 全是0
            position_ids=position_ids,  # None
            head_mask=head_mask,  # None
        )

        roberta_pooled_output = roberta_outputs[1]
        roberta_pooled_output = self.roberta_dropout(roberta_pooled_output)

        # bert encoding
        outputs = self.bert(
            input_ids=input_ids,  # torch.Size([9, 512])
            bbox=bbox,  # bbox
            attention_mask=attention_mask,
            # torch.Size([9, 512])  padding的全部为0
            token_type_ids=token_type_ids,  # torch.Size([9, 512]) 全是0
            position_ids=position_ids,  # None
            head_mask=head_mask,  # None
        )  # output = (torch.Size([9, 512, 768]), torch.Size([9, 768]))
        # 其中output[1] 是output[0] 取512中的第0个作为输出 进行dense和activation

        # when fine-tuned on downstream tasks, bert use [cls] for sentence
        # encoding
        bert_pooled_output = outputs[1]
        bert_pooled_output = self.bert_dropout(bert_pooled_output)

        if self.img_hidden_size > 0:
            bert_logits = self.classifier(bert_pooled_output)
            roberta_logits = self.roberta_classifier(roberta_pooled_output)

            img_features, img_fc = self.extract_result(
                image_data
            )  # image_data.shape   ([16, 3, 224, 224])
            img_features = img_features.view(img_features.size(0), -1)

            concat_pooled_output = torch.cat(
                (bert_pooled_output, img_features, roberta_pooled_output), -1
            )

            concat_pooled_output = self.dropout(concat_pooled_output)
            concat_logits = self.concat_classifier(concat_pooled_output)

            # ----------------------图像不存在，则用bert_logits代替-----------------------
            mask_img = torch.eq(
                torch.sum(image_data.view(image_data.size(0), -1), axis=1), 0
            )
            concat_logits[mask_img] = bert_logits[mask_img]
            img_fc[mask_img] = bert_logits[mask_img]
            # -------------------------------------------------------------------------

            # ------------------------增加二级类目结果约束-----------------------
            if labels is not None:
                sec_labels = torch.tensor(
                    list(map(lambda x: self.label_map[x.int().item()], labels))
                )
                sec_labels = sec_labels.to(labels.device)

            sec_bert_logits = self.sec_bert_classifier(bert_pooled_output)
            sec_roberta_logits = self.sec_roberta_classifier(roberta_pooled_output)
            sec_img_logits = self.sec_img_classifier(img_features)
            sec_concat_logits = self.sec_concat_classifier(
                torch.cat((bert_pooled_output, img_features, roberta_pooled_output), -1)
            )
            # ----------------------------------------------------------------
            outputs = (self.activation(concat_logits),) + outputs[2:]
            output_dict = {
                "bert_logits": bert_logits,
                "img_logits": img_fc,
                "roberta_logits": roberta_logits,
                "logits": concat_logits,
                "prob": self.activation(concat_logits),
                "outputs": outputs,
            }
        else:
            bert_logits = self.classifier(
                torch.cat((bert_pooled_output, roberta_pooled_output), -1)
            )
            outputs = (self.activation(bert_logits),) + outputs[  # outputs[2:] 为空
                2:
            ]  # add hidden states and attention if they are here
            output_dict = {
                "logits": bert_logits,
                "prob": self.activation(bert_logits),
                "outputs": outputs,
            }

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(concat_logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss(reduce=False)   #图像模态缺失，则只用bert输出
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(bert_logits.view(-1, self.num_labels), labels.view(-1))

                # ------------------------增加二级类目结果约束-----------------------
                sec_bert_loss = loss_fct(
                    sec_bert_logits.view(-1, self.sec_num_labels), sec_labels.view(-1)
                )
                sec_roberta_loss = loss_fct(
                    sec_roberta_logits.view(-1, self.sec_num_labels),
                    sec_labels.view(-1),
                )
                loss += sec_bert_loss + sec_roberta_loss
                # ----------------------------------------------------------------

                if self.img_hidden_size > 0:
                    # ------------------------增加二级类目结果约束-----------------------
                    sec_img_loss = loss_fct(
                        sec_img_logits.view(-1, self.sec_num_labels),
                        sec_labels.view(-1),
                    )
                    sec_concat_loss = loss_fct(
                        sec_concat_logits.view(-1, self.sec_num_labels),
                        sec_labels.view(-1),
                    )
                    loss += sec_img_loss + sec_concat_loss
                    # ----------------------------------------------------------------

                    roberta_loss = loss_fct(
                        roberta_logits.view(-1, self.num_labels), labels.view(-1)
                    )
                    bert_loss = loss_fct(
                        concat_logits.view(-1, self.num_labels), labels.view(-1)
                    )
                    img_loss = loss_fct(
                        img_fc.view(-1, self.num_labels), labels.view(-1)
                    )
                    loss += roberta_loss
                    loss += img_loss
                    loss += bert_loss

                # loss_dict.update({'sec_level_loss': sec_level_loss})

            output_dict.update({"total_loss": loss})

        return output_dict


@registry.register_model("AntmmfLayoutLM")
class AntmmfLayoutLM(BaseModel):
    MODEL_CLASSES = {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        "distilbert": (
            DistilBertConfig,
            DistilBertForSequenceClassification,
            DistilBertTokenizer,
        ),
        "layoutlm": (BertConfig, LayoutLMForSequenceClassification, BertTokenizer),
        "cascadelayoutlm": (
            BertConfig,
            CascadeLayoutlmForSequenceClassification,
            BertTokenizer,
        ),
    }

    def __init__(self, config):
        super().__init__(config)
        self.profiler = Timer()
        self.not_debug = False

    def profile(self, text):
        if self.not_debug:
            return
        self.writer.write(
            "AntmmfLayoutLM:" + text + ": " + self.profiler.get_time_since_start(),
            "debug",
        )
        self.profiler.reset()

    def build_for_test(self):
        # random initialization for test only
        model_config, model_class, tokenizer = AntmmfLayoutLM.MODEL_CLASSES[
            self.config.model_type
        ]
        # override default bert configs
        bert_config = model_config(**self.config.bert_config, pretrained_resnet=False)
        # build layoutlm with random initialization
        self.model = model_class(bert_config)

    def build(self):
        model_config, model_class, tokenizer = AntmmfLayoutLM.MODEL_CLASSES[
            self.config.model_type
        ]

        bert_config = model_config.from_pretrained(  # bert-base-chinese
            get_transformer_model_vocab_path(self.config.model_name_or_path),
            num_labels=self.config.num_labels,
        )
        kwargs = {}
        if "label_map" in self.config:
            kwargs["label_map"] = self.config.label_map
        if "pretrained_roberta_path" in self.config:
            kwargs["pretrained_roberta_path"] = get_transformer_model_vocab_path(
                self.config.pretrained_roberta_path
            )

        # bert-base-chinese-pytorch_model.bin
        self.model = model_class.from_pretrained(
            get_transformer_model_vocab_path(self.config.model_name_or_path),
            config=bert_config,
            **kwargs,
        )

    def get_optimizer_parameters(self, config):
        # code adapted from
        # https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L313
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.optimizer_attributes.params.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def get_custom_scheduler(self, trainer):
        # code adapted from:
        # https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py#L70
        optimizer = trainer.optimizer
        num_training_steps = trainer.config.training_parameters.max_iterations
        num_warmup_steps = trainer.config.training_parameters.num_warmup_steps

        if "train" in trainer.run_type:
            if num_training_steps == math.inf:
                epoches = trainer.config.training_parameters.max_epochs
                assert epoches != math.inf
                num_training_steps = (
                    trainer.config.training_parameters.max_epochs
                    * trainer.epoch_iterations
                )

            def linear_with_wram_up(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step)
                    / float(max(1, num_training_steps - num_warmup_steps)),
                )

            def cos_with_wram_up(current_step):
                num_cycles = 0.5
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
                return max(
                    0.0,
                    0.5
                    * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
                )

            lr_lambda = (
                cos_with_wram_up
                if trainer.config.training_parameters.cos_lr
                else linear_with_wram_up
            )

        else:

            def lr_lambda(current_step):
                return 0.0  # noqa

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)

    def forward(self, sample_list, *args, **kwargs):
        self.profiler.reset()
        inputs = {
            "input_ids": sample_list.input_ids,  # batch_size, 512
            "bbox": sample_list.bbox,
            "attention_mask": sample_list.attention_mask,  # batch_size, 512
            "token_type_ids": sample_list.token_type_ids,
            "image_data": sample_list.image_data,
        }
        if "targets" in sample_list:
            inputs.update({"labels": sample_list.targets})

        self.profile("preprare inputs")
        output_dict = self.model(**inputs)
        self.profile("model inference")

        # losses should be calculated when targets available, may be in
        # training or val phrase
        if "targets" in sample_list:
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            loss_names = [k for k in output_dict.keys() if "_loss" in k]
            if len(loss_names) > 0:  # losses were calculated in model
                output_dict["losses"] = {}
                for loss_name in loss_names:
                    output_dict["losses"][
                        loss_key + "/%s" % loss_name
                    ] = output_dict.pop(loss_name)
        self.profile("loss calculation")
        return output_dict
