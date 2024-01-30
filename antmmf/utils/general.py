# Copyright (c) 2023 Ant Group and its affiliates.
import collections
import gc
import importlib
import os
import sys
import tarfile
import warnings
import zipfile
from bisect import bisect
from contextlib import contextmanager
from typing import Dict, Any

import jsonlines
import numpy as np
import requests
import torch
import tqdm
import yaml
from torch import nn

from antmmf.common import constants
from antmmf.common.constants import DOWNLOAD_CHUNK_SIZE
from antmmf.utils.file_io import PathManager


def lr_lambda_update(i_iter, trainer):
    cfg = trainer.config
    if (
        cfg["training_parameters"]["use_warmup"] is True
        and i_iter <= cfg["training_parameters"]["warmup_iterations"]
    ):
        alpha = float(i_iter) / float(cfg["training_parameters"]["warmup_iterations"])
        return cfg["training_parameters"]["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        decay_steps = np.asarray(cfg["training_parameters"]["lr_steps"])
        if len(cfg["training_parameters"]["lr_epochs"]) > 0:
            epoch_iters = trainer.epoch_iterations
            # if lr_epochs is indicated, then use instead of lr_steps.
            decay_steps = epoch_iters * np.asarray(
                cfg["training_parameters"]["lr_epochs"]
            )
        idx = bisect(decay_steps, i_iter)
        return pow(cfg["training_parameters"]["lr_ratio"], idx)


def clip_gradients(model, i_iter, writer, config):
    # TODO: Fix question model retrieval
    max_grad_l2_norm = config["training_parameters"]["max_grad_l2_norm"]
    clip_norm_mode = config["training_parameters"]["clip_norm_mode"]

    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)

            writer.add_scalars({"grad_norm": norm}, i_iter)

        elif clip_norm_mode == "question":
            question_embedding = model.module.question_embedding_module
            norm = nn.utils.clip_grad_norm(
                question_embedding.parameters(), max_grad_l2_norm
            )

            writer.add_scalars({"question_grad_norm": norm}, i_iter)
        else:
            raise NotImplementedError(
                "Clip norm mode %s not implemented" % clip_norm_mode
            )


def ckpt_name_from_core_args(config):
    seed = config["training_parameters"]["seed"]

    ckpt_name = "{}_{}".format(
        "-".join(config.task_attributes.keys()),
        "-".join(config.model_attributes.keys()),
    )

    if seed is not None:
        ckpt_name += "_{:d}".format(seed)

    return ckpt_name


def foldername_from_config_override(args):
    cfg_override = None
    if hasattr(args, "config_override"):
        cfg_override = args.config_override
    elif "config_override" in args:
        cfg_override = args["config_override"]

    folder_name = ""
    if cfg_override is not None and len(cfg_override) > 0:
        folder_name = yaml.safe_dump(cfg_override, default_flow_style=True)
        folder_name = folder_name.replace(":", ".").replace("\n", " ")
        folder_name = folder_name.replace("/", "_")
        folder_name = " ".join(folder_name.split())
        folder_name = folder_name.replace(". ", ".").replace(" ", "_")
        folder_name = "_" + folder_name
    return folder_name


def get_antmmf_root():
    from antmmf.common.registry import registry

    antmmf_root = registry.get("antmmf_root", no_warning=True)
    if antmmf_root is None:
        antmmf_root = os.path.dirname(os.path.abspath(__file__))
        antmmf_root = os.path.abspath(os.path.join(antmmf_root, ".."))
        registry.register("antmmf_root", antmmf_root)
    return antmmf_root


def download_file(url, output_dir=".", filename=""):
    if len(filename) == 0:
        filename = os.path.join(".", url.split("/")[-1])

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, filename)
    r = requests.get(url, stream=True)

    if r.status_code != requests.codes["ok"]:
        print(
            "The url {} is broken. If this is not your own url, please open up an issue on GitHub.".format(
                url
            )
        )
    file_size = int(r.headers["Content-Length"])
    num_bars = int(file_size / DOWNLOAD_CHUNK_SIZE)

    with open(filename, "wb") as fh:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE),
            total=num_bars,
            unit="MB",
            desc=filename,
            leave=True,
        ):
            fh.write(chunk)


def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    is_parallel = isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    )

    if is_parallel and hasattr(model.module, "get_optimizer_parameters"):
        parameters = model.module.get_optimizer_parameters(config)

    return parameters


def count_parameters(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num * 1.0 / 1e6, "Trainable": trainable_num * 1.0 / 1e6}


def dict_to_string(dictionary):
    logs = []
    if dictionary is None:
        return ""
    for key, val in dictionary.items():
        if hasattr(val, "item"):
            val = val.item()
        # if key.count('_') == 2:
        #     key = key[key.find('_') + 1:]
        logs.append("%s: %.4f" % (key, val))

    return ", ".join(logs)


def get_overlap_score(candidate, target):
    """Takes a candidate word and a target word and returns the overlap
    score between the two.

    Parameters
    ----------
    candidate : str
        Candidate word whose overlap has to be detected.
    target : str
        Target word against which the overlap will be detected

    Returns
    -------
    float
        Overlap score betwen candidate and the target.

    """
    if len(candidate) < len(target):
        temp = candidate
        candidate = target
        target = temp
    overlap = 0.0
    while len(target) >= 2:
        if target in candidate:
            overlap = len(target)
            return overlap * 1.0 / len(candidate)
        else:
            target = target[:-1]
    return 0.0


def updir(d, n):
    """Given path d, go up n dirs from d and return that path"""
    ret_val = d
    for _ in range(n):
        ret_val = os.path.dirname(ret_val)
    return ret_val


def print_cuda_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / (1024 * 1024))
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))


def get_current_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except Exception:
            pass


def extract_file(path, output_dir="."):
    _FILETYPE_TO_OPENER_MODE_MAPPING = {
        ".zip": (zipfile.ZipFile, "r"),
        ".tar.gz": (tarfile.open, "r:gz"),
        ".tgz": (tarfile.open, "r:gz"),
        ".tar": (tarfile.open, "r:"),
        ".tar.bz2": (tarfile.open, "r:bz2"),
        ".tbz": (tarfile.open, "r:bz2"),
    }

    cwd = os.getcwd()
    os.chdir(output_dir)

    extension = "." + ".".join(os.path.abspath(path).split(".")[1:])

    opener, mode = _FILETYPE_TO_OPENER_MODE_MAPPING[extension]
    with opener(path, mode) as f:
        f.extractall()

    os.chdir(cwd)
    return output_dir


def iterative_support(func, query):
    if isinstance(query, (list, tuple, set)):
        return [iterative_support(func, i) for i in query]
    return func(query)


def get_absolute_path(paths):
    # String check should be first as Sequence would pass for string too
    if isinstance(paths, str):
        # If path is absolute return it directly
        if os.path.isabs(paths):
            return paths

        possible_paths = [
            # Direct path
            paths
        ]

        mmf_root = get_antmmf_root()
        # Relative to root folder of mmf install
        possible_paths.append(os.path.join(mmf_root, paths))

        # Test all these paths, if any exists return
        for path in possible_paths:
            if PathManager.exists(path):
                # URIs
                if path.find("://") == -1:
                    return os.path.abspath(path)
                else:
                    return path

        # If nothing works, return original path so that it throws an error
        return paths
    elif isinstance(paths, collections.abc.Iterable):
        return [get_absolute_path(path) for path in paths]
    else:
        raise TypeError("Paths passed to dataset should either be string or list")


"""
convenient way to access dict keys as obj.foo instead of obj['foo']
following the link below
https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
"""


class AttrDict(dict):
    MARKER = object()

    def __init__(self, value=None):
        super().__init__()
        self.init_from(value)

    def init_from(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("expected dict")

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, AttrDict.MARKER)
        if found is AttrDict.MARKER:
            found = AttrDict()
            super(AttrDict, self).__setitem__(key, found)
        return found

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.init_from(state)

    __setattr__, __getattr__ = __setitem__, __getitem__


def get_bert_configured_parameters(module, lr=None):
    param_optimizer = list(module.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if lr is not None:
        for p in optimizer_grouped_parameters:
            p["lr"] = lr

    return optimizer_grouped_parameters


def get_optimizer_parameters_for_bert(module, config):
    # Pretraining has same LR for all of the parts
    if module.config.training_head_type == "pretraining":
        return get_bert_configured_parameters(module)

    # For finetuning setup, we have classifier
    lr = config.optimizer_attributes.params.lr
    model_config = getattr(
        config.model_attributes, list(config.model_attributes.keys())[0], {}
    )
    finetune_lr_multiplier = getattr(model_config, "finetune_lr_multiplier", 1)
    # Finetune the bert pretrained part with finetune_lr_multiplier if it is
    # set
    parameters = get_bert_configured_parameters(
        module.bert, lr * finetune_lr_multiplier
    )
    # Classifier will be trained on the normal lr
    parameters += get_bert_configured_parameters(module.classifier, lr)

    return parameters


def transform_to_batch_sequence(tensor):
    if tensor is not None:
        if len(tensor.size()) == 2:
            return tensor
        else:
            assert len(tensor.size()) == 3
            return tensor.contiguous().view(-1, tensor.size(-1))
    else:
        return None


def transform_to_batch_sequence_dim(tensor):
    if tensor is not None:
        if len(tensor.size()) == 3:
            return tensor
        else:
            assert len(tensor.size()) == 4
            return tensor.contiguous().view(-1, tensor.size(-2), tensor.size(-1))
    else:
        return None


def get_transformer_model_vocab_path(bert_model_name):
    """
    transformer pretrained weights and vocabs dir, weights can be downloaded from https://huggingface.co/models

    **Note**: for online serving, pre-trained weights must be accessed by this func
    Args:
        bert_model_name:

    Returns:

    """
    # keep absolute path as it is
    if os.path.isabs(bert_model_name):
        return bert_model_name
    pretrained_model_dir = os.environ.get(constants.BERT_PRETRAINED_MODELS_ENV_VAR, "")
    # pretrained_model_dir not found
    if not pretrained_model_dir:
        warnings.warn(
            "Environment variable 'PYTORCH_TRANSFORMERS_CACHE' not set, "
            "automatically downloading pretrained BERT models"
        )
    # invalid pretrained_model_dir
    if not os.path.isdir(pretrained_model_dir):
        warnings.warn(
            f"Invalid 'PYTORCH_TRANSFORMERS_CACHE' path:{pretrained_model_dir}"
        )
    pretrained_model_path = os.path.join(pretrained_model_dir, bert_model_name)
    # pretrained weights not found in pretrained_model_dir
    if not os.path.exists(pretrained_model_path):
        warnings.warn(
            f"Pretrained weights:{bert_model_name} were not found in {pretrained_model_dir}, "
            "will automatically downloading pretrained BERT models"
        )
        # return short-cut name to enable auto-download
        return bert_model_name
    else:
        return pretrained_model_path


def get_user_model_resource_path(resource_name):
    """
    function to locate absolute paths of user model related resources

    **Note**: for online serving, user resource files must be accessed by this func
    Args:
        resource_name:

    Returns: absolute path
    """
    # keep absolute path as it is
    if os.path.isabs(resource_name):
        return resource_name

    user_model_dir = os.environ.get(constants.USER_MODEL_HOME, "")

    if not user_model_dir:
        warnings.warn(
            "Environment variable 'USER_MODEL_HOME' not set, resources in config.yml must be absolute paths"
        )
    # invalid user_model_dir
    if not os.path.isdir(user_model_dir):
        warnings.warn(f"Invalid 'USER_MODEL_HOME' path:{user_model_dir}")
    return os.path.join(user_model_dir, resource_name)


# each line is a json format string


def jsonl_dump(info, filepath, append=False):
    """
    Print a list of json dictionary in a file
    The file then contains information such as
    {xxx:yyy}
    {zzz:qqq}

        Parameters:
            info: a list of json dictionary
            filepath: the path to the file to save the information
        Return: None
    """
    assert isinstance(info, list), "first argument needs to be a list"
    mode = "w" if append is False else "a"
    with jsonlines.open(filepath, mode=mode) as f:
        for ln in info:
            f.write(ln)


def batched_index_select(input, dim, index):
    r"""
    index select on batch

    Arguments:
        input: B x * x ... x *
        dim: 0 <= scalar
        index: B x M
    Return:
        size of B x M * x ... x *
    """
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index.to(device=input.device))


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


def flatten_list(k):
    result = list()
    for i in k:
        if isinstance(i, (list, tuple)):
            # The isinstance() function checks if the object (first argument) is an
            # instance or subclass of classinfo class (second argument)
            result.extend(flatten_list(i))  # Recursive call
        else:
            result.append(i)
    return result


def is_module_available(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def check_required_keys(check_dict: Dict[str, Any], required_keys=[]) -> bool:
    return all([key in check_dict for key in required_keys])


def is_method_override(model_class, base_class, method_name):
    assert issubclass(model_class, base_class)
    this_method = getattr(model_class, method_name)
    base_method = getattr(base_class, method_name)
    return this_method is not base_method


def get_package_version(package_name):
    """
    Args:
        package_name(str): The name of package.

    Returns:
        Package's version(str) if package was found, otherwise return None.
    """
    py_version_info = sys.version_info
    major = py_version_info.major
    minor = py_version_info.minor
    package_version = None
    if major >= 3 and minor >= 8:
        from importlib.metadata import version, PackageNotFoundError

        try:
            package_version = version(package_name)
        except PackageNotFoundError:
            warnings.warn("'{}' was not found.".format(package_name))
    else:
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            package_version = get_distribution(package_name).version
        except DistributionNotFound:
            warnings.warn("'{}' was not found.".format(package_name))
    return package_version
