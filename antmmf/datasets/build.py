# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import os
import os.path as osp

from transformers import AutoTokenizer, AutoConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

from antmmf.common import constants
from antmmf.common.registry import registry
from antmmf.datasets.base_dataset import BaseIterableDataset
from antmmf.datasets.processors.processors import Processor
from antmmf.datasets.samplers import DistributedSampler, SequentialSampler
from antmmf.utils.general import get_absolute_path
from antmmf.utils.general import get_transformer_model_vocab_path


def build_dataset_sampler(dataset, sampler_config, default_config: dict = None):
    """
    Args:
        dataset (antmmf.datasets.BaseDataset): instance of BaseDataset
        sampler_config(configNode):
        default_config (dict)
    Returns:
        sampler(antmmf.datasets.samplers.AntmmfSampler):

    """

    # For iterable-style datasets, data loading order is entirely controlled by
    # the user-defined iterable. No sampler is needed.
    if isinstance(dataset, BaseIterableDataset):
        return None

    sampler_cls = registry.get_sampler_class(sampler_config.type)

    if (
        issubclass(sampler_cls, DistributedSampler)
        and not sampler_config.split_eval
        and dataset.dataset_type != "train"
    ):
        # make sure each process owns a complete dataset for val/test metric calculation
        return SequentialSampler(dataset)  # default as sequential sampler
    assert (
        sampler_cls is not None
    ), f"Sampler of type:{sampler_config.type} is not registered"
    if default_config is None:
        default_config = {}
    default_config.update(sampler_config.params)
    return sampler_cls(dataset, **default_config)


def build_processors(config, *args, **kwargs):
    from antmmf.utils.general import AttrDict

    processors = {}
    for processor_key, processor_params in config.items():
        processor_object = Processor(processor_params, *args, **kwargs)
        processors[processor_key] = processor_object
    # encapsulate with AttrDict for easy access
    return AttrDict(processors)


def build_tokenizer(config, *args, **kwargs):
    """
    AutoTokenizer needs transformer config to decide which tokenizer instance to initialize:
    * If we have downloaded the pretrained model, `tokenizer_config.type` should be the pretrained model
    directory's name, which is placed under the path indicated by environment variable `PYTORCH_TRANSFORMERS_CACHE`.
    The pretrained model dir should have two files:`config.json` and `vocab.txt`. AutoTokenizer will use
    `config.json` to deicde which tokenizer to use, and initializing a Tokenizer instance with `vocab.txt`.

    * If the pretrained model is NOT downloaded, `tokenizer_config.params.model_type` is used for indicating which
    tokenzier we want to instantiate while `tokenizer_config.type` indicating the absolute path of `vocab.txt`
    """

    tokenizer = None

    if config.params.get(constants.PRETRAINED_STR, True):
        auto_config = None  # using pretrained model's `config.json` to decide
        pretrained_model_dir = get_transformer_model_vocab_path(config.type)

        if not osp.exists(pretrained_model_dir):
            # roll back to model_type to trigger auto-downloading
            pretrained_model_dir = osp.basename(pretrained_model_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_dir,
            config=auto_config,
            **config.params,
        )
    else:  # using model_type to decide
        auto_config = AutoConfig.for_model(model_type=config.params.get("model_type"))
        vocab_path = config.params.get("vocab_path")
        vocab_path = (
            os.path.abspath(config.type)
            if vocab_path is None
            else get_absolute_path(vocab_path)
        )
        for (
            config_class,
            (
                tokenizer_class_py,
                tokenizer_class_fast,
            ),
        ) in TOKENIZER_MAPPING.items():
            if isinstance(auto_config, config_class):
                # from IPython import embed; embed()
                tokenizer = tokenizer_class_py(vocab_path, **config.params)
                break
    assert tokenizer, f"Couldn't decide tokenizer type by tokenizer_config: {config}"
    return tokenizer
