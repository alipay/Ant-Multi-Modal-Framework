# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import numpy as np
import torch
from torch import nn
from torch import sigmoid

from antmmf.modules.utils import build_hier_tree
from antmmf.utils.general import get_absolute_path
from antmmf.utils.file_io import PathManager


class HierarchicalClassifier(nn.Module):
    """
    Classifier perform Hierarchical Softmax.
    Hierarchical Softmax is an alternative to softmax that is faster to evaluate:
    it is  time to evaluate compared to normal softmax. It utilises a multi-layer tree,
    where the probability of a node is calculated through the product of probabilities on each edge on the path to that
    node.

    This AntMMF implementation support outputting all intermediate nodes probabilities and
    allowing complex relations among nodes, such as one label may belong to multiple
    labels, compared to the open source one: https://talbaumel.github.io/blog/softmax/
    """

    def __init__(self, hidden_size, hier_label_schema, config=None):
        """
        Args:
            hidden_size(int): hidden size of hier-softmax
            hier_label_schema(list): list of categories,  it should be a dict if a category has sub categories,
            otherwise be a string. If category is a dict, it should contain only one key-value pair, with key indicate
            label name of itself and value be a hier_label_schema. The following is an example of hier_label_schema:
            ['教育',  {'时尚': ['用车技巧', '新技术汽车', {'养生':['运动']}]}, '体育'], its corresponding category tree should be:
                               root
                            /    |    \
                          教育   体育     时尚
                                     /   |     \
                               用车技巧 新技术汽车 养生
                                                 |
                                                运动
            config.classifier_embedding(str): 存储classifier_embedding的文件路径。classifier_embedding即为标签label的embedding，
                    可以来源于word_embedding、model_embedding和grap_embedding等。参考https://arxiv.org/abs/1904.03582,
                    利用item_embedding和label_embedding做点积的方式，将语义信息融入到多标签分类中。
            config.use_multilabel(bool): False单标签分类，hier_probs为softmax。True多标签分类，hier_probs为sigmoid。
            config.maxnum_of_multilabel(int): 当use_multilabel==True时生效，表示多标签targets最大个数
        """
        super(HierarchicalClassifier, self).__init__()
        self.tree = build_hier_tree(hier_label_schema)
        self.classifier_embedding_path = (
            config.get("classifier_embedding", None) if config else None
        )
        self.use_multilabel = config.get("use_multilabel", False) if config else False
        self.maxnum_of_multilabel = (
            config.get("maxnum_of_multilabel", len(self.tree.ALL_LABELS))
            if config
            else None
        )
        self.fc = nn.ModuleList(
            [
                nn.Linear(hidden_size, param_group["num_outputs"])
                for param_group in self.tree.ParamGroup
            ]
        )
        self.softmax = nn.Softmax(-1)
        self.criterion = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        if self.classifier_embedding_path:
            label_embedding_file = get_absolute_path(self.classifier_embedding_path)
            assert PathManager.exists(
                label_embedding_file
            ), f"{label_embedding_file} not exists"
            f = PathManager.open(label_embedding_file, "r", encoding="utf-8")
            label_e = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                arrs = line.split()
                label = arrs[0]
                embedding = [float(x) for x in arrs[1].split(",")]
                label_e[label] = embedding
            f.close()
            group_id = 0
            for name, m in self.named_modules():
                # 层次分类
                if name == "fc." + str(group_id):
                    init_e = []
                    group_node = self.tree.search_node("group_id", group_id)
                    for x in group_node.children:
                        init_e.append(label_e[x.label_name])
                    m.weight.data = torch.tensor(init_e)
                    nn.init.constant_(m.bias.data, 0)
                    group_id += 1
        else:
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, std=0.02)

    def fully_decode(self, hier_probs):
        # inference all nodes prob, slower
        batch_size = hier_probs[0].shape[0]
        result = []
        for node in self.tree.traverse():
            if not node.is_leaf():  # 输出最细粒度分类，不输出中间节点的结果
                continue
            # record probability of nodes along root->leaf path
            prob = torch.zeros([batch_size, self.tree.get_depth()], dtype=torch.float32)
            if torch.cuda.is_available():
                prob = prob.cuda()
            hier_labels, group_ids, label_str = self.tree.get_node_info(node)
            for level, (label, param_id) in enumerate(zip(hier_labels, group_ids)):
                hier_prob = hier_probs[param_id][:, label]  # batch
                prob[:, level] = hier_prob
            result.append({"label": label_str, "prob": prob})
        # post process
        decode_result = []
        label_list = [x["label"] for x in result]
        # [num_lables, batch_size, tree_depth]
        probs = torch.stack([x["prob"] for x in result], dim=0)
        accumulate_probs = probs.clone().detach()
        num_levels = probs.size(2)
        for level in range(1, num_levels):
            accumulate_probs[:, :, level] = probs[:, :, : level + 1].prod(dim=-1)
        for batch_idx in range(batch_size):
            label_prob_infos = []
            for label_idx in range(len(label_list)):
                label_name = label_list[label_idx]
                label_probs = (
                    accumulate_probs[label_idx][batch_idx].cpu().numpy().tolist()
                )
                label_prob_infos.append([tuple(label_probs), label_name])
            # tuple sort
            label_dis = sorted(label_prob_infos, reverse=True, key=lambda x: x[0])

            best_acc_prob, best_label = label_dis[0]
            best_prob = best_acc_prob[np.where(np.array(best_acc_prob) > 0)[0][-1]]

            cur_res = {
                "result": {"prob": best_prob, "label": best_label, "prob_list": []},
                "detail": [],
            }  # TODO  'prob_list':prob_list
            for acc_prob, label in label_dis:
                label_info = {
                    "prob": acc_prob[np.where(np.array(acc_prob) > 0)[0][-1]],
                    "label": label,
                }
                cur_res["detail"].append(label_info)
            decode_result.append(cur_res)
        return decode_result

    def greedy_decode(self, hier_probs):
        # greedy decode, much faster than fully_decode
        batch_size = hier_probs[0].shape[0]

        def decode_step(node, batch_id, prob_list=list(), node_prob=None):
            if node_prob is None:
                node_prob = 1.0
                prob_list = [1.0]
            if not node.is_leaf():
                # get max prob child
                prob, nodes_id = hier_probs[node.group_id][batch_id].max(
                    dim=0
                )  # batch_sie, num_children
                level_prob = node_prob * prob
                prob_list.append(round(level_prob.item(), 4))
                return decode_step(
                    node.children[nodes_id], batch_id, prob_list, node_prob * prob
                )
            else:
                return node, node_prob, prob_list

        def decode_step_multi(node, batch_id, topk):
            prob_nodes = hier_probs[node.group_id][batch_id]
            k_score, k_nodes = torch.topk(prob_nodes, topk)
            return k_nodes, k_score

        decode_result = []
        for batch_idx in range(batch_size):
            node, node_prob, prob_list = decode_step(self.tree, batch_idx)
            _, _, label_str = self.tree.get_node_info(node)
            cur_res = {
                "result": {
                    "prob": node_prob.item(),
                    "label": label_str,
                    "prob_list": prob_list,
                },
                "detail": [],
            }
            if self.use_multilabel:
                multilabel = []
                multiscore = []
                topk = min(
                    self.maxnum_of_multilabel, len(hier_probs[node.group_id][batch_idx])
                )
                k_nodes, k_score = decode_step_multi(self.tree, batch_idx, topk)
                for node, score in zip(k_nodes, k_score):
                    _, _, label_str = self.tree.get_node_info(self.tree.children[node])
                    multilabel.append(label_str)
                    multiscore.append(score.item())
                cur_res["result"]["multilabel"] = multilabel
                cur_res["result"]["multiscore"] = multiscore
            decode_result.append(cur_res)
        return decode_result

    def inference(self, hier_logits):
        if self.use_multilabel:
            hier_probs = [sigmoid(x) for x in hier_logits]
        else:
            hier_probs = [self.softmax(x) for x in hier_logits]
        ret_dict = {"pred_hier_tags": self.greedy_decode(hier_probs)}

        return ret_dict

    def forward(self, features):
        hier_logits = [
            self.fc[module_idx](features) for module_idx in range(len(self.fc))
        ]
        with torch.no_grad():
            ret_dict = self.inference(hier_logits)
        ret_dict.update({"hier_logits": hier_logits})
        return ret_dict
