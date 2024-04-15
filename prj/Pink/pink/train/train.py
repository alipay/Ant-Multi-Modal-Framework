# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import random

import torch

import transformers
from transformers import CLIPImageProcessor
from pink.train.pink_trainer import PinkTrainer
from pink import conversation as conversation_lib
from torch.utils.data import ConcatDataset
from pink.model import *
import pink.datasets
from pink.datasets import MemoryEfficientConcatDataset

from PIL import Image
import torch.nn as nn
import io
import torch.distributed
import time
import cv2
cv2.setNumThreads(0)


# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
# FIXME: seems wrong?
# DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    llama_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v0")
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    tune_mm_mlp_adapter: bool = field(default=True)
    freeze_vit: bool = field(default=True)
    freeze_llm: bool = field(default=True)
    add_mark_tokens: bool = field(default=False)
    tune_input_output_embeddings: bool = field(default=False)


@dataclass
class DataArguments:
    base_path: str = field(default=None, metadata={"help": "base data path"})
    data_path: str = field(default=None, metadata={"help": "data file e.g. json"})
    image_folder: Optional[str] = field(default=None)
    dataset_name: str = field(default="")
    is_multimodal: bool = False
    image_token_len: int = 0
    image_size: int = field(default=224)
    crop_size: int = field(default=224)
    grounding_caption: bool = field(default=False)
    conversation_template: str = field(default="llamav1")
    expand2square: bool = field(default=False)
    task_pool: str = field(default="Relation,CoarseLocation,RegionGroundingCaption,VisualGrounding,Detection,Counting")
    task_pool_365: str = field(default="Relation,CoarseLocation,RegionGroundingCaption,VisualGrounding,Detection,Counting")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class AdapterArguments:
    adapter_hidden_dim: int = field(default=8)
    adapter_scale: float = field(default=1.0)
    adapter_dropout: float = field(default=0.1)
    llm_adapter_enable: bool = field(default=False)
    visual_adapter_enable: bool = field(default=False)
    adapter_attn: bool = field(default=True)
    adapter_mlp: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            assert all(x is not None and x.shape == images[0].shape for x in images)
            batch['images'] = torch.stack(images)

        assert 'has_image' in instances[0].keys()
        has_images = [instance['has_image'] for instance in instances]
        batch['has_images'] = has_images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    assert not data_args.expand2square, "not support expand2square for now"
    # NOTE: use "@" merge dataset without considering the number of samples in the dataset
    datasets = data_args.dataset_name.split("@")
    image_folders = data_args.image_folder.split("@")
    data_paths = data_args.data_path.split("@")
    merge_datasets = []
    repeat_datasets = []
    for data_path, image_folder, dataset in zip(data_paths, image_folders, datasets):
        if ":" in dataset:
            # NOTE: use : for repeat dataset
            dataset, repeat_number = dataset.split(":")
            repeat_datasets.append(int(repeat_number))
        else:
            repeat_datasets.append(1)
        dataset_cls = getattr(pink.datasets, dataset)
        train_dataset = dataset_cls(tokenizer=tokenizer,
                                    data_path=data_path,
                                    multimodal_cfg=dict(
                                        is_multimodal=data_args.is_multimodal,
                                        image_token_len=data_args.image_token_len,
                                        image_folder=image_folder,
                                        base_path=data_args.base_path,
                                        cur_token_len=data_args.cur_token_len,
                                        image_size=data_args.image_size,
                                        crop_size=data_args.crop_size,
                                        grounding_caption=data_args.grounding_caption,
                                        conversation_template=data_args.conversation_template,
                                        vg_task_pool=data_args.task_pool.split(","),
                                        task_pool_365=data_args.task_pool_365.split(","),
                                        expand2square=data_args.expand2square,
                                        ))
        merge_datasets.append(train_dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=MemoryEfficientConcatDataset(merge_datasets, repeats=repeat_datasets),
                eval_dataset=None,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, AdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    if model_args.model_name_or_path != model_args.llama_path:
        # need to merge parameters
        llama_path = model_args.llama_path
        weight_map_index = json.load(open(os.path.join(llama_path, "pytorch_model.bin.index.json"), "r"))
        shard_files = list(set(weight_map_index["weight_map"].values()))
        loaded_keys = weight_map_index["weight_map"].keys()
        state_dict = {}
        for index, shard_file in enumerate(shard_files):
            state_dict.update(torch.load(os.path.join(llama_path, shard_file), map_location="cpu"))
        peft_parameters = torch.load(os.path.join(model_args.model_name_or_path, "saved_parameters.pth"), map_location="cpu")
        for k, v in peft_parameters.items():
            state_dict[k] = v
        model_path = None
    else:
        state_dict = None
        model_path = model_args.model_name_or_path

    model = PinkModel.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=model_args.model_name_or_path,
        state_dict=state_dict,
        cache_dir=training_args.cache_dir,
        llama_path=model_args.llama_path,
        clip_path=model_args.vision_tower,
        clip_select_layer=model_args.mm_vision_select_layer,
        adapter_vision_enable=adapter_args.visual_adapter_enable,
        adapter_vision_dim=adapter_args.adapter_hidden_dim,
        adapter_vision_scale=adapter_args.adapter_scale,
        adapter_vision_dropout=adapter_args.adapter_dropout,
        adapter_attn=adapter_args.adapter_attn,
        adapter_mlp=adapter_args.adapter_mlp,
        adapter_llm_enable=adapter_args.llm_adapter_enable,
        adapter_llm_dim=adapter_args.adapter_hidden_dim,
        adapter_llm_scale=adapter_args.adapter_scale,
        adapter_llm_dropout=adapter_args.adapter_dropout,
        crop_size=data_args.crop_size)

    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token

    model.initialize_vision_tokenizer(tokenizer=tokenizer)

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    model.model.to(dtype)
    model.lm_head.to(dtype)

    for param in model.parameters():
        param.requires_grad_(False)

    if model_args.tune_mm_mlp_adapter:
        for p in model.mm_projector.parameters():
            p.data = p.data.float()
            p.requires_grad = True

    data_args.image_token_len = model.config.num_patches
    data_args.cur_token_len = model.config.num_patches
    data_args.is_multimodal = True

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    if not model_args.freeze_vit:
        if model.config.adapter_vision_enable:
            for name, param in model.vision_model.named_parameters():
                if "adapter_" not in name:
                    param.requires_grad = False
                else:
                    param.data = param.data.float()
                    param.requires_grad = True
        else:
            for name, param in model.vision_model.named_parameters():
                param.data = param.data.float()
                param.requires_grad = True
            for index, layer in enumerate(model.vision_model.vision_model.encoder.layers[model.config.clip_select_layer:]):
                if index == 0:
                    continue
                else:
                    for name, param in layer.named_parameters():
                        param.requires_grad = False
    else:
        model.vision_model.train = disabled_train
        model.vision_model.eval()

    if not model_args.freeze_llm:
        if model.config.adapter_llm_enable:
            for name, param in model.model.named_parameters():
                if "adapter_" not in name:
                    param.requires_grad = False
                else:
                    param.data = param.data.float()
                    param.requires_grad = True
        else:
            for name, param in model.model.named_parameters():
                param.data = param.data.float()
                param.requires_grad = True

    if model_args.add_mark_tokens:
        if model.config.adapter_llm_enable:
            freeze = True
        else:
            freeze = False
        if model_args.freeze_llm:
            freeze = True
        if model_args.tune_input_output_embeddings:
            freeze = False
        model.add_mark_tokens(tokenizer, device=training_args.device, freeze=freeze)

    params_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    print("param_grad: {}".format(params_grad))

    trainer = PinkTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
