#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.utils import logging
from copy import deepcopy
logger = logging.get_logger(__name__)

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForPreTraining, \
                         LlamaConfig, LlamaForCausalLM, LlamaModel, CLIPVisionModel, \
                         CLIPImageProcessor, CLIPModel, PretrainedConfig, PreTrainedModel

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache

from .adapter import visual_adapter, adapter, AdapterLayer
import math


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"
BEGIN_LOC = "<loc>"
END_LOC = "</loc>"
BEGIN_CLS = "<cls>"
END_CLS = "</cls>"
BEGIN_RELATION = "<rel>"
END_RELATION = "</rel>"
BEGIN_QUESTION = "<qes>"
END_QUESTION = "</qes>"
BEGIN_OPTIONS = "<opt>"
END_OPTIONS = "</opt>"


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def convert_weights_to_dtype(model: nn.Module, dtype):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_dtype(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype=dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype=dtype)

    model.apply(_convert_weights_to_dtype)


class CLIPVisionInitModel(CLIPVisionModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, AdapterLayer):
            nn.init.xavier_uniform_(module.tune_adapter_a.weight)
            module.tune_adapter_a.bias.data.zero_()
            module.tune_adapter_b.weight.data.zero_()
            module.tune_adapter_b.bias.data.zero_()


class PinkConfig(LlamaConfig):
    model_type = "pink"
    def __init__(
        self,
        llama_path=None,
        clip_path=None,
        clip_select_layer=-2,
        adapter_vision_enable=False,
        adapter_vision_dim=8,
        adapter_vision_scale=1.0,
        adapter_vision_dropout=0.05,
        adapter_attn=True,
        adapter_mlp=False,
        adapter_llm_enable=False,
        adapter_llm_dim=8,
        adapter_llm_scale=1.0,
        adapter_llm_dropout=0.05,
        adapter_non_linear=False,
        crop_size=224,
        **kwargs,
    ):
        self.llama_path = llama_path
        self.clip_path = clip_path
        self.clip_select_layer = clip_select_layer
        self.adapter_vision_enable = adapter_vision_enable
        self.adapter_vision_dim = adapter_vision_dim
        self.adapter_vision_scale = adapter_vision_scale
        self.adapter_vision_dropout = adapter_vision_dropout
        self.adapter_attn = adapter_attn
        self.adapter_mlp = adapter_mlp
        self.adapter_llm_enable = adapter_llm_enable
        self.adapter_llm_dim = adapter_llm_dim
        self.adapter_llm_scale = adapter_llm_scale
        self.adapter_llm_dropout = adapter_llm_dropout
        self.adapter_non_linear = adapter_non_linear
        self.crop_size = crop_size
        super().__init__(
            **kwargs,
        )


class PinkModel(LlamaForCausalLM):
    config_class = PinkConfig

    def __init__(self, config: PinkConfig):
        with adapter(hidden_dim=config.adapter_llm_dim, scale=config.adapter_llm_scale, dropout=config.adapter_llm_dropout, enabled=config.adapter_llm_enable, non_linear=config.adapter_non_linear, attn=config.adapter_attn, mlp=config.adapter_mlp):
            super().__init__(config)
        # Initialize weights and apply final processing
        with visual_adapter(hidden_dim=config.adapter_vision_dim, scale=config.adapter_vision_scale, dropout=config.adapter_vision_dropout, attn=config.adapter_attn, mlp=config.adapter_mlp, enabled=config.adapter_vision_enable, non_linear=config.adapter_non_linear):
            vision_model = CLIPVisionInitModel.from_pretrained(config.clip_path, image_size=config.crop_size, ignore_mismatched_sizes=True)
            if config.adapter_vision_enable:
                for index, layer in enumerate(vision_model.vision_model.encoder.layers[config.clip_select_layer:]):
                    if index == 0:
                        continue
                    else:
                        if config.adapter_attn:
                            del layer.adapter_attn
                        if config.adapter_mlp:
                            del layer.adapter_mlp
            self.vision_model = vision_model

        num_features = self.vision_model.config.hidden_size
        self.mm_projector = nn.Linear(num_features, config.hidden_size)

        num_patches = (self.vision_model.config.image_size // self.vision_model.config.patch_size) ** 2
        self.config.num_patches = num_patches

    def get_model(self):
        return self.model

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, AdapterLayer):
            nn.init.xavier_uniform_(module.tune_adapter_a.weight)
            module.tune_adapter_a.bias.data.zero_()
            module.tune_adapter_b.weight.data.zero_()
            module.tune_adapter_b.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value
        if isinstance(module, CLIPVisionModel):
            module.gradient_checkpointing = False

    def encode_image(self, images):
        image_forward_out = self.vision_model(images, output_hidden_states=True)
        select_hidden_state_layer = getattr(self.config, "clip_select_layer", -1)
        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
        image_features = select_hidden_state[:, 1:]
        image_features = self.mm_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        has_images: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if orig_embeds_params is not None:
            orig_embeds_params = orig_embeds_params[0]
            with torch.no_grad():
                # NOTE: fix other word embeddings&lm_head
                assert self.original_tokens_length == orig_embeds_params.shape[0]
                self.get_input_embeddings().weight.data[:self.original_tokens_length].copy_(orig_embeds_params.clone().detach())
                assert self.orig_lm_head is not None
                self.get_output_embeddings().weight.data[:self.original_tokens_length].copy_(self.orig_lm_head[0].clone().detach())
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        useful_images = []
        for image, has_image in zip(images, has_images):
            if has_image:
                useful_images.append(image)
        if len(useful_images) > 0:
            if input_ids[0].shape[0] != 1 or self.training:
                useful_images = torch.stack(useful_images, dim=0)
                image_features = self.encode_image(useful_images)
                image_features = image_features.to(inputs_embeds.dtype)

        new_inputs_embeds = []
        cur_image_index = 0
        for input_id, inputs_embed, has_image in zip(input_ids, inputs_embeds, has_images):
            if has_image and (input_id.shape[0] != 1 or self.training):
                image_feature = image_features[cur_image_index]
                cur_image_index += 1
                num_patches = image_feature.shape[0]
                if (input_id == self.config.im_patch_token).sum() != num_patches:
                    raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(input_id == self.config.im_patch_token)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                new_inputs_embed = torch.cat((inputs_embed[:mask_index_start], image_feature, inputs_embed[mask_index_start+num_patches:]), dim=0)
                new_inputs_embeds.append(new_inputs_embed)
            else:
                new_inputs_embeds.append(inputs_embed)

        new_inputs_embeds = torch.stack(new_inputs_embeds, dim=0)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        llama_output = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = llama_output[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + llama_output[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=llama_output.past_key_values,
            hidden_states=llama_output.hidden_states,
            attentions=llama_output.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if past_key_value := getattr(self.model.layers[0].self_attn, "past_key_value", None):
            # generation with static cache
            past_length = past_key_value.get_seq_length()
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = kwargs.get("cache_position", None)
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + position_ids.shape[-1], device=position_ids.device
            )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "has_images": kwargs.get("has_images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, tokenizer):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        for p in self.get_input_embeddings().parameters():
            p.requires_grad = False
        for p in self.get_output_embeddings().parameters():
            p.requires_grad = False

        self.config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

    def add_mark_tokens(self, tokenizer, device, freeze=True):
        if freeze:
            self.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().detach().to(device=device)]
            self.orig_lm_head = [self.get_output_embeddings().weight.data.clone().detach().to(device=device)]
        self.original_tokens_length = len(tokenizer)
        num_new_tokens = tokenizer.add_tokens([BEGIN_DESCRIPTION, END_DESCRIPTION, BEGIN_LOC, END_LOC, BEGIN_RELATION, END_RELATION, BEGIN_CLS, END_CLS, BEGIN_QUESTION, END_QUESTION, BEGIN_OPTIONS, END_OPTIONS], special_tokens=True)
        if num_new_tokens > 0:
            self.resize_token_embeddings(len(tokenizer))

            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                        dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                        dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = True


AutoConfig.register("pink", PinkConfig)
AutoModelForCausalLM.register(PinkConfig, PinkModel)
