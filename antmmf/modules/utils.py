# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import math
import torch
import threading
import collections.abc
from typing import List
from antmmf.utils.general import get_absolute_path, get_package_version
from antmmf.utils.file_io import PathManager
from antmmf.common.constants import HIER_CLASS_SEP


def get_mask(nums: torch.Tensor, max_num: int) -> torch.FloatTensor:
    """
    Return a mask that indicate whether the position is exact data or padding data.

    Args:
        nums (torch.Tensor): the length of each sequence data.
        max_num (int): the max length of the sequence data.

    Returns:
        torch.FloatTensor: a float tensor, where `1` indicate exact data and `0` padding data.
    """
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.shape[0]
    assert (
        len(nums.size()) == 1
    ), f"nums should be a tensor with shape of [batch_size], but got {nums.size()}"
    assert max_num > 0, f"max_num should be greater than 0, but got {max_num}"
    non_pad_mask = (
        torch.arange(max_num, device=nums.device)
        .view(1, -1)
        .expand(batch_size, -1)
        .lt(nums.view(-1, 1))
    )
    return non_pad_mask.type(torch.float32)


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    torch_version = get_package_version("torch")
    if torch_version is None:
        raise ModuleNotFoundError("Not Found torch")
    split_version = torch_version.split(".")
    major_version = int(split_version[0])
    minor_version = int(split_version[1])
    if major_version < 1 or (major_version >= 1 and minor_version <= 7):
        return torch.irfft(
            com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)),
            1,
            signal_sizes=(a.shape[-1],),
        )
    else:
        a_rfft = torch.fft.rfft(a)
        b_rfft = torch.fft.rfft(b)
        a_rfft_stack = torch.stack((a_rfft.real, a_rfft.imag), -1)
        b_rfft_stack = torch.stack((b_rfft.real, b_rfft.imag), -1)
        output_com_mult = com_mult(conj(a_rfft_stack), b_rfft_stack)
        output_complex = torch.complex(output_com_mult[..., 0], output_com_mult[..., 1])
        return torch.fft.irfft(output_complex, a.shape[-1])


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, "p must be in range of [0,1]"

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x):
    """Obtain height and width from x.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


class TreeNode:
    ALL_LABELS = []
    ParamGroup = []

    def __init__(self, parent=None, label_name=None, label_idx=-1):
        self.parent = parent  # TreeNone
        self.children = []  # list of TreeNone
        self.parent_child_idx = -1
        self.label_name = label_name
        self.label_idx = label_idx  # -1 indicate root
        # pytorch params group, nodes with same hierarchical param should perform softmax
        # together
        self.group_id = -1

    def add_child(self, node):
        assert isinstance(node, TreeNode)
        node.parent = self
        node.parent_child_idx = len(self.children)
        if node.label_name not in TreeNode.ALL_LABELS:
            TreeNode.ALL_LABELS.append(node.label_name)
        if node.label_idx is None:  # auto generate label_index
            node.label_idx = TreeNode.ALL_LABELS.index(node.label_name)
        self.children.append(node)

    def __str__(self):
        return f"node[idx={self.label_idx}, name={self.label_name}, group_id={self.group_id}]"

    __repr__ = __str__

    def get_node_info(self, node):
        assert isinstance(node, TreeNode)
        hier_label, group_ids, label_str = [], [], []
        parent = node
        while parent is not None:
            hier_label.append(parent.parent_child_idx)
            group_ids.append(parent.group_id)
            label_str.append(parent.label_name)
            parent = parent.parent
        # hier_label[0] is root node id:-1, group_ids[-1] is node self
        # group_id: -1
        return (
            hier_label[::-1][1:],
            group_ids[::-1][:-1],
            HIER_CLASS_SEP.join(label_str[-2::-1]),
        )

    def get_label_idx_info(self, label_idx):
        node = self.search_node("label_idx", label_idx)
        return self.get_node_info(node)

    def encode_label_str(self, label_str, mask_padding=-1):
        """
        hier_label and hier_param may not have consistent shape for all samples in
        the batch, using mask_padding padding to the same shape.
        Args:
            label_str(str): hier-labels with '-' concatenated, eg. '时尚-用车技巧' or '时尚'.

        Returns:
            padded_hier_label: the path from root to node, aka hierarchical label for label_str.
            padded_hier_param: the softmax param group used from root to node.
        """
        node = self.get_node_from_label_str(label_str)
        padded_hier_label = mask_padding * torch.ones(
            [len(TreeNode.ParamGroup)], dtype=torch.long
        )
        padded_hier_param = mask_padding * torch.ones(
            [len(TreeNode.ParamGroup)], dtype=torch.long
        )
        hier_label, hier_param, _ = self.get_node_info(node)
        padded_hier_label[: len(hier_label)] = torch.tensor(
            hier_label, dtype=torch.long
        )
        padded_hier_param[: len(hier_param)] = torch.tensor(
            hier_param, dtype=torch.long
        )
        return padded_hier_label, padded_hier_param

    def encode_multilabel_str(self, label_str, mask_padding=-1):
        """
        label_str has multiple labels.
        Now TreeNode.ParamGroup must be 1, and TreeNode.ParamGroup > 1 will be supported in subsequent versions.
        Args:
            label_str(str): multi-labels with ',' concatenated, eg. '时尚-用车技巧,体育'.

        Returns:
            padded_hier_label: the path from root to node, aka hierarchical label for label_str.
            padded_hier_param: the group used from root to node.
        """

        assert (
            len(TreeNode.ParamGroup) == 1
        ), f"multilabel: TreeNode.ParamGroup must be 1,but now = {len(TreeNode.ParamGroup)}"
        label_list = label_str.strip().split(",")
        padded_hier_label = mask_padding * torch.ones(
            [len(TreeNode.ParamGroup), len(TreeNode.ALL_LABELS)], dtype=torch.long
        )
        for index, x in enumerate(label_list):
            node = self.get_node_from_label_str(x)
            hier_label, _, _ = self.get_node_info(node)
            padded_hier_label[0][index : index + 1] = torch.tensor(
                hier_label, dtype=torch.long
            )
        padded_hier_param = mask_padding * torch.ones(
            [len(TreeNode.ParamGroup)], dtype=torch.long
        )
        padded_hier_param[:1] = torch.tensor([0], dtype=torch.long)
        padded_hier_label_num = mask_padding * torch.ones(
            [len(TreeNode.ParamGroup)], dtype=torch.long
        )
        padded_hier_label_num[:1] = torch.tensor([len(label_list)], dtype=torch.long)
        return padded_hier_label, padded_hier_param, padded_hier_label_num

    def search_node(self, attr, val):
        if getattr(self, attr) == val:
            return self
        for child in self.children:
            ret = child.search_node(attr, val)
            if ret is not None:
                return ret
        return None

    def print_tree(self, prefix=""):
        print(prefix + str(self))
        for child in self.children:
            child.print_tree(prefix=prefix + "\t")

    def build_hier_group(self):
        # for pytorch params softmax computation
        if len(self.children) > 0:
            group_params = {"num_outputs": len(self.children)}
            group_id = len(TreeNode.ParamGroup)
            group_params["group_id"] = group_id
            TreeNode.ParamGroup.append(group_params)
            self.group_id = group_id
            for child in self.children:
                child.build_hier_group()

    def traverse(self):
        result, queue = [], [self]
        while queue:
            cur_node = queue.pop(0)
            if cur_node.label_idx != -1:
                result.append(cur_node)
            if len(cur_node.children) > 0:
                queue += cur_node.children
        return result

    def is_tree_root(self):
        return self.label_idx == -1

    def is_leaf(self):
        return len(self.children) == 0

    def get_depth(self):
        if self.is_leaf():
            return 0
        return 1 + max([child.get_depth() for child in self.children])

    def get_node_from_label_str(self, label_str):
        """
        get node from hier labels
        Note: Using this func to get tree node, especially when label exists more than once,
        the hierarchical structure will help locate the exactly node.
        Args:
            label_str(str): hier-labels with '-' concatenated, eg. '时尚-用车技巧' or '时尚'.

        Returns:
            TreeNode
        """
        hier_labels = label_str.split(HIER_CLASS_SEP)
        root = self
        for label in hier_labels:
            next = root.search_node("label_name", label)
            if next is None:
                return None
            root = next
        return root

    def is_subnode(self, another_node):
        parent = self.parent
        while parent is not None:
            if parent is another_node:
                return True
            else:
                parent = parent.parent
        return False

    def get_label_schema(self):
        if len(self.children) > 0:
            return {
                self.label_name: [child.get_label_schema() for child in self.children]
            }
        else:
            return self.label_name

    def compare_hier_label(self, pred_hier_label, target_hier_label):
        """
        Args:
            pred_hier_label: 输出的最细粒度的层次分类标签
            target_hier_label: 细粒度或粗粒度标签

        Returns:

        """
        pred_node = self.get_node_from_label_str(pred_hier_label)
        target_node = self.get_node_from_label_str(target_hier_label)
        if pred_node is target_node:
            return True
        # target: 时尚 pred: 时尚-新技术汽车
        if pred_node.is_subnode(target_node):
            return True
        # target: 时尚-新技术汽车 pred: 时尚
        if target_node.is_subnode(pred_node):
            return False
        return False

    @staticmethod
    def build_hier_tree(label_schema, root=None):
        if root is None:
            root = TreeNode(label_name="root")  # root node has no parents
        for label in label_schema:
            if isinstance(label, str):
                cur_node = TreeNode(label_name=label, label_idx=None)
                root.add_child(cur_node)
            elif isinstance(label, collections.abc.Mapping):
                assert (
                    len(label) == 1
                ), "label_schema has multiple labels in same label_idx"
                label_name = list(label.keys())[0]
                cur_node = TreeNode(label_name=label_name, label_idx=None)
                sub_schema = list(label.values())[0]
                assert isinstance(sub_schema, list)
                cur_node = TreeNode.build_hier_tree(sub_schema, root=cur_node)
                root.add_child(cur_node)
        if root.label_idx == -1:  # node of tree root
            root.build_hier_group()
        return root

    @staticmethod
    def build_tree_from_file(label_schema_file):
        label_schema_file = get_absolute_path(label_schema_file)
        assert PathManager.exists(label_schema_file), f"{label_schema_file} not exists"
        hier_label_list = []
        for hier_label in PathManager.open(label_schema_file, "r", encoding="utf-8"):
            hier_label = hier_label.strip()
            if not hier_label:
                continue
            hier_label_list.append(hier_label)

        root = TreeNode(label_name="root")
        for hier_str in hier_label_list:
            hier_labels = hier_str.split(HIER_CLASS_SEP)
            cur_node = root
            for hl in hier_labels:
                search_res = cur_node.search_node("label_name", hl)
                if search_res is None:
                    node = TreeNode(label_name=hl, label_idx=None)
                    cur_node.add_child(node)
                    cur_node = node
                else:
                    cur_node = search_res
        root.build_hier_group()
        return root


tree_instance = None
lock = threading.Lock()


def singleton_tree(func):
    """
    Ensure only one tree will be constructed
    """

    def wrapper(*args, **kwargs):
        global tree_instance
        if tree_instance is None:
            with lock:
                # Double Check Locking:
                # see detail:
                # https://en.wikipedia.org/wiki/Double-checked_locking
                if tree_instance is None:
                    tree_instance = func(*args, **kwargs)
        return tree_instance

    return wrapper


@singleton_tree
def build_hier_tree(label_schema):
    if isinstance(label_schema, (str,)):
        # input is a path of label schema file, with each line indicates a hier label,
        # eg. '时尚-用车技巧'
        tree = TreeNode.build_tree_from_file(label_schema)
    elif isinstance(label_schema, (list,)):
        # input is a list of label_schema, eg.
        # ['教育', '科技', {'汽车': ['用车技巧', '新技术汽车', '二手车']}, '体育']
        tree = TreeNode.build_hier_tree(label_schema)
    else:
        raise Exception(f"unknown hier_label_schema input:{label_schema}")
    return tree


class TimeDistributed(torch.nn.Module):
    """
    Taken shamelessly from AllenNLP repo
    A wrapper that unrolls the second (time) dimension of a tensor
    into the first (batch) dimension, applies some other `Module`,
    and then rolls the time dimension back up.

    Given an input shaped like `(batch_size, time_steps, [rest])` and a `Module` that takes
    inputs like `(batch_size, [rest])`, `TimeDistributed` reshapes the input to be
    `(batch_size * time_steps, [rest])`, applies the contained `Module`, then reshapes it back.
    Note that while the above gives shapes with `batch_size` first, this `Module` also works if
    `batch_size` is second - we always just combine the first two dimensions, then split them.
    It also reshapes keyword arguments unless they are not tensors or their name is specified in
    the optional `pass_through` iterable.
    """

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs, pass_through: List[str] = None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [
            self._reshape_tensor(input_tensor) for input_tensor in inputs
        ]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        new_size = some_input.size()[:2] + reshaped_outputs.size()[1:]
        outputs = reshaped_outputs.contiguous().view(new_size)

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)
