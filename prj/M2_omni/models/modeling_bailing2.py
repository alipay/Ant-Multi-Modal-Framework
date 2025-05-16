# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Bailing2Native model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from peft import LoraConfig, get_peft_model

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from transformers.utils import (
    add_start_docstrings,
    logging,
)
from .qwen2_vit import Qwen2ViT
from .modeling_llama_3d import (
    Llama3dForCausalLM,
)

from .sanm_audio import BailingAudioModel, fix_audio_encoder_bf16_
from .configuration_bailing2 import Bailing2Config

from transformers.cache_utils import Cache, StaticCache

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Bailing2Config"

# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

@dataclass
class Bailing2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Bailing2 causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

@dataclass
class Bailing2NativeCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Bailing2Native causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

Bailing2Native_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Bailing2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare Bailing2Native Model outputting raw hidden-states without any specific head on top.",
    Bailing2Native_START_DOCSTRING,
)
class Bailing2NativeForConditionalGeneration(PreTrainedModel):
    config_class = Bailing2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2ViT", "BailingAudioModel", "LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(self, config: Bailing2Config, vision_model=None, audio_model=None, language_model=None):
        super().__init__(config)
        self.config = config
        self.select_layer = config.select_layer
        self.mlp_depths = config.mlp_depths

        self.audio_compress = config.audio_config.audio_compress
        self.num_query_token_image = config.num_query_token_image
        self.num_query_token_video = config.num_query_token_video
        self.num_query_token_audio = config.num_query_token_audio

        # self.llm_dytpe = config.llm_config.torch_dtype
        self.llm_dytpe = torch.float16  # for audio
        self.vision_type = config.vision_type

        # init base model
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = Qwen2ViT(config.vision_config)
        if config.use_vit_lora > 0:
            self.wrap_vit_lora(r=config.use_vit_lora, lora_alpha=2 * config.use_vit_lora)

        if audio_model is not None:
            self.audio_model = audio_model
        else:
            self.audio_model = BailingAudioModel(config.audio_config)

            self.audio_model = self.audio_model.to(dtype=torch.float32)
            fix_audio_encoder_bf16_(self.audio_model)

        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = Llama3dForCausalLM._from_config(
                config.llm_config,
                torch_dtype=self.llm_dytpe,
                attn_implementation=config._attn_implementation
            )

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.vocab_size = config.llm_config.vocab_size

        # init project module
        vit_hidden_size = config.vision_config.hidden_size

        llm_hidden_size = config.llm_config.hidden_size
        audio_hidden_size = config.audio_config.audio_output_size * config.audio_config.audio_compress

        mlp_modules_img = [nn.Linear(vit_hidden_size, llm_hidden_size)]
        for _ in range(1, self.mlp_depths):
            mlp_modules_img.append(nn.GELU())
            mlp_modules_img.append(nn.Linear(llm_hidden_size, llm_hidden_size))

        self.linear_proj = nn.Sequential(*mlp_modules_img)

        mlp_modules_audio = [nn.Linear(audio_hidden_size, llm_hidden_size)]
        for _ in range(1, self.mlp_depths):
            mlp_modules_audio.append(nn.GELU())
            mlp_modules_audio.append(nn.Linear(
                llm_hidden_size, llm_hidden_size
            ))
        self.linear_proj_audio = nn.Sequential(*mlp_modules_audio)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def wrap_vit_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        target_modules = [
            'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'
        ]
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = 2
        image_token_id = self.config.image_patch_token
        video_token_id = self.config.frame_patch_token
        image_start_token_id = self.config.image_start_token
        video_start_token_id = self.config.video_start_token

        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )

            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                if video_grid_thw is None:
                    vision_start_indices = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                else:
                    vision_start_indices = torch.argwhere(input_ids == video_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_start_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_start_token_id, st) + 1
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_start_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_start_token_id, st) + 1
                    else:
                        ed_video = len(input_tokens) + 1

                    if image_grid_thw is not None and ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                len_llm_positions = llm_positions.shape[-1]
                len_temp_position_ids = position_ids[..., i, attention_mask[i] == 1].shape[-1]

                if len_llm_positions > len_temp_position_ids:
                    llm_positions = llm_positions[..., :len_temp_position_ids]
                elif len_llm_positions < len_temp_position_ids:
                    temp_positions = torch.zeros(len_temp_position_ids - len_llm_positions).repeat(3, 1)
                    temp_positions = temp_positions.to(dtype=llm_positions.dtype, device=llm_positions.device)
                    llm_positions = torch.cat([llm_positions, temp_positions], dim=-1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        pixel_values_audios: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask_audio: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Bailing2CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if (
                pixel_values is not None or pixel_values_videos is not None or pixel_values_audios is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos/pixel_values_audios and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if position_ids is None and input_ids is not None:
            try:
                position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
            except:
                position_ids, _ = self.get_rope_index(input_ids, attention_mask=attention_mask)

        image_embeds, video_embeds, audio_embeds = None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)

        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)

        if pixel_values_audios is not None:
            audio_embeds = self.extract_audio_feature(
                pixel_values_audios,
                attention_mask_audio,
                n_query=self.num_query_token_audio,
                audio_compress=self.audio_compress
            )

        if inputs_embeds is None:
            inputs_embeds = self.prompt_wrap(
                input_ids,
                image_embeds=image_embeds,
                video_embeds=video_embeds,
                audio_embeds=audio_embeds,
            )

        # only bailing2native require this
        inputs_embeds = inputs_embeds.to(dtype=self.llm_dytpe, device=input_ids.device)
        # inputs_embeds = inputs_embeds.to(dtype=self.config.torch_dtype)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Bailing2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def encode_audio_post_processing(self, audio_embeds: torch.Tensor, n_query=None, audio_compress=3):
        # audio_embeds(-1, length, dim)
        batch_size = audio_embeds.size(0)
        real_length = audio_embeds.size(1)
        dim = audio_embeds.size(2)
        valid_length = real_length // audio_compress * audio_compress
        audio_embeds = audio_embeds[:, :valid_length, :].reshape(
            batch_size, valid_length // audio_compress, dim * audio_compress
        )
        audio_embeds = audio_embeds[:, :n_query, :]
        return audio_embeds

    def extract_image_feature(self, pixel_values, grid_thw):
        image_embeds = self.vision_model(
            pixel_values,
            grid_thw=grid_thw
        )
        image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def extract_audio_feature(
        self,
        pixel_values_audios,
        attention_mask_audio,
        n_query=None,
        audio_compress=3
    ):
        audio_embeds = self.audio_model(
            pixel_values_audios,
            attention_mask_audio,
        )
        audio_embeds = audio_embeds.to(self.dtype)
        end_of_audio_emb = self.get_input_embeddings()(
            torch.tensor(self.config.end_of_audio_token).long().to(pixel_values_audios.device),
        ).reshape(1, 1, -1).to(self.dtype)

        audio_embeds = self.encode_audio_post_processing(
            audio_embeds=audio_embeds,
            n_query=n_query,
            audio_compress=audio_compress
        )

        audio_embeds = self.linear_proj_audio(audio_embeds)
        len_end_of_audio = n_query - audio_embeds.size(1)
        if len_end_of_audio > 0:
            audio_embeds = torch.cat(
                [audio_embeds, end_of_audio_emb.repeat([audio_embeds.size(0), len_end_of_audio, 1])], 1)

        assert audio_embeds.size(1) == n_query

        audio_embeds = F.normalize(audio_embeds, dim=-1)

        return audio_embeds

    def prompt_wrap_image(self, input_ids, inputs_embeds, vision_embeds, image_token_id=None):
        if vision_embeds is None or input_ids is None:
            return inputs_embeds

        self.config.image_token_id = image_token_id if image_token_id is not None else self.config.image_patch_token

        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = vision_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_mask = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def prompt_wrap_video(self, input_ids, inputs_embeds, vision_embeds, video_token_id=None):
        if vision_embeds is None or input_ids is None:
            return inputs_embeds

        self.config.video_token_id = video_token_id if video_token_id is not None else self.config.frame_patch_token

        n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
        n_video_features = vision_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        video_mask = (
            (input_ids == self.config.video_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        video_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds

    def prompt_wrap_audio(self, input_ids, inputs_embeds, audio_embeds, audio_token_id=None):
        if audio_embeds is None or input_ids is None:
            return inputs_embeds

        self.config.audio_token_id = audio_token_id if audio_token_id is not None else self.config.audio_patch_token
        n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
        n_audio_features = audio_embeds.shape[0]
        if n_audio_tokens != n_audio_features:
            raise ValueError(
                f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
            )
        audio_mask = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        return inputs_embeds

    def prompt_wrap(self, input_ids, image_embeds=None, video_embeds=None, audio_embeds=None, target_embeds=None):
        inputs_embeds = self.get_input_embeddings()(input_ids.clip(0, self.get_input_embeddings().weight.shape[0] - 1))
        if image_embeds is None and video_embeds is None and audio_embeds is None and target_embeds is None:
            return inputs_embeds

        if image_embeds is not None:
            inputs_embeds = self.prompt_wrap_image(input_ids, inputs_embeds, image_embeds)
        if video_embeds is not None:
            inputs_embeds = self.prompt_wrap_video(input_ids, inputs_embeds, video_embeds)
        if audio_embeds is not None:
            inputs_embeds = self.prompt_wrap_audio(input_ids, inputs_embeds, audio_embeds)
        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        pixel_values_audios=None,
        attention_mask_audio=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "pixel_values_audios": pixel_values_audios,
                "attention_mask_audio": attention_mask_audio,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs
