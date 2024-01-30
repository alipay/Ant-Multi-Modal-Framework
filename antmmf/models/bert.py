# Copyright (c) 2023 Ant Group and its affiliates.

import requests
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 用来读取注册的keys
from antmmf.common.registry import registry

# 需要继承的模型的基本类
from antmmf.models.base_model import BaseModel
from antmmf.utils.general import get_transformer_model_vocab_path, get_absolute_path
from antmmf.datasets.build import build_tokenizer
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers import BertModel, AutoConfig


@registry.register_model("bert")
class BERT(BaseModel):
    # 所有模型的初始化，需要这个config
    # 这个config应该包含所有模型需要用到的参数设置
    def __init__(self, config):
        super().__init__(config)
        self._abs_model_path = get_transformer_model_vocab_path(self.config.model_path)
        if self.config.get("use_rule"):
            # self.tokenizer = BertTokenizer.from_pretrained(self._abs_model_path)
            self.tokenizer = build_tokenizer(self.config.tokenizer_config)
            self.all_rules = self._process_rules()
            self.rule_num = len(self.all_rules["rule_label"])

    @staticmethod
    def _download_model(model_path):
        model_url = "YourUrl"  # noqa
        r = requests.get(model_url, stream=True)
        file_size = round(int(r.headers["content-length"]) / 1024 / 1024)
        print(f"file_size: {file_size}MB")
        with open(model_path, "wb") as f:
            # f.write(r.content)
            chunk_idx = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):
                if chunk:
                    f.write(chunk)
                chunk_idx += 1

    def build_for_test(self):
        model_config = BertConfig.from_dict(self.config.roberta_config)
        self.roberta = BertModel(model_config)
        num_output = [len(self.config.label_index)]
        if self.config.get("label_index_hier"):
            num_output += [len(labels) for labels in self.config["label_index_hier"]]
        self.classifier = RobertaClassificationHead(
            model_config.hidden_size,
            self.config.hidden_dropout_prob,
            num_output,
        )
        if self.config.get("use_rule"):
            self.linear_input = nn.Linear(model_config.hidden_size, 1)
            self.linear_rules = nn.Linear(sum(num_output), 1)

    # 在这里定义所需要的模块

    def _process_rules(self):
        max_rule_len = 0
        rules_path = get_absolute_path(self.config.rule_data)
        rules = open(rules_path, "r", encoding="utf-8").readlines()
        all_rules_ids = []
        all_rules_type_ids = []
        all_rules_mask = []
        all_rules_label = []
        for rule in rules:
            r, l = rule.split("###")
            input_ids = (
                [self.tokenizer.cls_token_id]
                + self.tokenizer.encode(r, add_special_tokens=False)
                + [self.tokenizer.sep_token_id]
            )
            token_type_ids = [0] * len(input_ids)
            mask = [1] * len(input_ids)
            rule_label = [0] * len(self.config.label_index)
            for label in l.split(","):
                label_name, label_val = label.split(":")
                rule_label[self.config.label_index.index(label_name)] = float(label_val)

            if len(input_ids) > max_rule_len:
                max_rule_len = len(input_ids)
            all_rules_ids.append(input_ids)
            all_rules_type_ids.append(token_type_ids)
            all_rules_mask.append(mask)
            all_rules_label.append(rule_label)

        max_rule_len = min(max_rule_len, self.config.max_len)
        for sample_idx in range(len(rules)):
            if len(all_rules_ids[sample_idx]) > max_rule_len:
                all_rules_ids[sample_idx] = all_rules_ids[sample_idx][:max_rule_len]
                all_rules_type_ids[sample_idx] = all_rules_type_ids[sample_idx][
                    :max_rule_len
                ]
                all_rules_mask[sample_idx] = all_rules_mask[sample_idx][:max_rule_len]
            else:
                padding_length = max_rule_len - len(all_rules_ids[sample_idx])
                all_rules_ids[sample_idx] = all_rules_ids[sample_idx] + (
                    [self.tokenizer.pad_token_id] * padding_length
                )
                all_rules_type_ids[sample_idx] = all_rules_type_ids[sample_idx] + (
                    [self.tokenizer.pad_token_type_id] * padding_length
                )
                all_rules_mask[sample_idx] = all_rules_mask[sample_idx] + (
                    [0] * padding_length
                )

        all_rules = {
            "rule_ids": torch.tensor(all_rules_ids),
            "rule_type_ids": torch.tensor(all_rules_type_ids),
            "rule_mask": torch.tensor(all_rules_mask),
            "rule_label": torch.tensor(all_rules_label),
        }
        return all_rules

    def build(self):
        self.roberta = BertModel.from_pretrained(self._abs_model_path)
        model_config = AutoConfig.from_pretrained(self._abs_model_path)
        num_output = [len(self.config.label_index)]
        if self.config.get("label_index_hier"):
            num_output += [len(labels) for labels in self.config["label_index_hier"]]
        self.classifier = RobertaClassificationHead(
            model_config.hidden_size,
            self.config.hidden_dropout_prob,
            num_output,
        )
        if self.config.get("use_rule"):
            self.linear_input = nn.Linear(model_config.hidden_size, 1)
            self.linear_rules = nn.Linear(sum(num_output), 1)

    # MMF模型的forward函数的输入是一个词典，包含了有关输入特征的信息
    def forward(self, sample_list):
        # 注意：这些key应该和后面讲到的特征提取相对应
        # 文本特征可以依据text作为key来获得
        # orig_text = sample_list["orig_text"]
        input_ids = sample_list["ids"]
        token_type_ids = sample_list["token_type_ids"]
        attention_mask = sample_list["mask"]

        use_hierarchical_classify = self.config.get("hier_classify")
        if use_hierarchical_classify:
            target_hier = sample_list["target_hier"]
            targets_mask = sample_list["targets_mask"]
        batch_size = input_ids.shape[0]

        # 或者文本与图像特征
        roberta_output = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=self.config.output_attention,
        )

        sequence_output = roberta_output[0]
        logits, logits_hier = self.classifier(sequence_output)
        if self.config.get("multi_label"):
            prob = torch.sigmoid(logits)
        else:
            prob = F.softmax(logits, dim=-1)

        # 使用当前的模型编码规则的文本
        if self.config.get("use_rule"):
            batch_num = math.ceil(self.rule_num / batch_size)
            rule_embeddings = []
            for i in range(batch_num):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                rule_ids = self.all_rules["rule_ids"][start_idx:end_idx].to(
                    input_ids.device
                )
                rule_type_ids = self.all_rules["rule_type_ids"][start_idx:end_idx].to(
                    input_ids.device
                )
                rule_mask = self.all_rules["rule_mask"][start_idx:end_idx].to(
                    input_ids.device
                )
                rule_embedding = self.roberta(
                    rule_ids,
                    token_type_ids=rule_type_ids,
                    attention_mask=rule_mask,
                    output_attentions=False,
                )[0][:, 0, :]
                rule_embeddings.append(rule_embedding)
            rule_embeddings = torch.cat(rule_embeddings, dim=0)
            rule_embeddings = rule_embeddings.expand(
                batch_size, -1, -1
            )  # batch_size * rule_num * hidden_size
            rule_labels = (
                self.all_rules["rule_label"]
                .expand(batch_size, -1, -1)
                .to(input_ids.device)
            )
            input_embeddings = sequence_output[:, 0, :]  # batch_size * hidden_size
            rules_attn = F.softmax(
                torch.bmm(rule_embeddings, input_embeddings.unsqueeze(2)), 1
            )  # batch_size * rule_num * 1
            copy_label = torch.bmm(rule_labels.transpose(1, 2), rules_attn).squeeze(
                -1
            )  # batch_size * num_class
            copy_prob = torch.sigmoid(
                self.linear_input(input_embeddings) + self.linear_rules(copy_label)
            )
            prob = copy_prob * copy_label + (1 - copy_prob) * prob
        output = {"logits": logits, "prob": prob}
        if use_hierarchical_classify:  # TODO: use rules support hier_classification
            output["logits_hier"] = [i.clone() for i in logits_hier]
            for hier_i, logits_hier_i in enumerate(logits_hier):
                for i in range(batch_size):
                    logits_hier_i_i = logits_hier_i[i]
                    targets_mask_i = targets_mask[i]
                    targets_mask_hier_i_i = targets_mask_i[hier_i]
                    logits_hier_i_i -= targets_mask_hier_i_i * 1000
        if self.config.get("output_embedding"):
            output["cls_embedding"] = sequence_output[:, 0, :]
        if self.config.get("output_attention"):
            attentions = roberta_output[-1]
            attention_last_layer = attentions[-1]
            attention_last_layer_cls = attention_last_layer[:, :, 0, :]
            output["attention_cls"] = [i for i in attention_last_layer_cls]

        # losses should be calculated when targets available, may be in
        # training or val phrase
        if "targets" in sample_list:
            targets = sample_list["targets"]
            if self.config.get("use_rule"):
                if self.config.get("multi_label"):
                    loss_fct = nn.BCELoss()
                    bert_ce_loss = loss_fct(prob, targets.float())
                else:
                    loss_fct = nn.NLLLoss()
                    bert_ce_loss = loss_fct(
                        torch.log(prob.view(-1, len(self.config.label_index))),
                        targets.view(-1),
                    )
            else:
                if self.config.get("multi_label"):
                    loss_fct = nn.BCEWithLogitsLoss()
                    bert_ce_loss = loss_fct(logits, targets.float())
                else:
                    loss_fct = CrossEntropyLoss()
                    bert_ce_loss = loss_fct(
                        logits.view(-1, len(self.config.label_index)), targets.view(-1)
                    )
            if use_hierarchical_classify:
                loss_fct = CrossEntropyLoss()
                for hier_i in range(target_hier.size(1)):
                    bert_ce_loss += loss_fct(
                        logits_hier[hier_i], target_hier[:, hier_i].view(-1)
                    )
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output.update({"losses": {loss_key + "/bert_ce_loss": bert_ce_loss}})

        return output


class RobertaClassificationHead(nn.Module):
    # code refer to:
    # https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_roberta.py#L674
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_output):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_output[0])
        self.out_proj_others = None
        if len(num_output) > 1:
            self.out_proj_others = nn.ModuleList(
                [
                    nn.Linear(hidden_size, num_output_others)
                    for num_output_others in num_output[1:]
                ]
            )
        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.dense.weight, std=0.02)
        torch.nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, features, **kwargs):  # TODO: 优化模型结构
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        y = self.out_proj(x)
        if self.out_proj_others is None:
            return y, []
        return y, [output_proj_i(x) for output_proj_i in self.out_proj_others]
