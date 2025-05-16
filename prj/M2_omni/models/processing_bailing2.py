# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Processor class for Bailing2."""

import math
import sys
from typing import Iterable, List, Union, Dict, Optional, Tuple

import torch
from PIL import Image

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput, get_image_size, to_numpy_array
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from .image_processing_bailing2 import Bailing2ImageProcessor
from .feature_extraction_sanm import SANMFeatureExtractor
from .bailing2_utils import process_vision_info

DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_VIDEO_PATCH_TOKEN = "<videoPatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>"
DEFAULT_GEN_IM_START_TOKEN = "<gen_image>"
DEFAULT_GEN_IM_END_TOKEN = "</gen_image>"
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<imageHere>"
DEFAULT_END_OF_CHUNK_TOKEN = "<end_of_chunk>"

DEFAULT_END_OF_AUDIO_TOKEN = "<end_of_audio>"
DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
DEFAULT_AU_START_TOKEN = "<audio>"
DEFAULT_AU_END_TOKEN = "</audio>"
DEFAULT_GEN_AUDIO_PATCH_TOKEN = "<gen_audioPatch>"
DEFAULT_GEN_AU_START_TOKEN = "<gen_audio>"
DEFAULT_GEN_AU_END_TOKEN = "</gen_audio>"
PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<audioHere>"

class Bailing2ProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {"padding": False, "padding_side": "right"},
        "image_kwargs": {},
        "video_kwargs": {},
        "audio_kwargs": {"padding": "max_length", "return_tensors": True},
    }

# @register_processor("Bailing2Processor")
class Bailing2Processor(ProcessorMixin):
    r"""
    Constructs a Bailing2 processor which wraps a bailing2 image processor, bailing audio processor and a LLaMa tokenizer into a single processor.
    Args:
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        num_audio_tokens (`int`, *optional*):
            Number of audio tokens for one video that will be returned by audio model.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        audio_token (`str`, *optional*, defaults to `"<audio>"`):
            Special token used to denote audio location.
    """

    attributes = ["tokenizer"]
    optional_attributes = ["chat_template"]

    tokenizer_class = "AutoTokenizer"

    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "image_token",
        "video_token",
        "audio_tokens"
    ]

    def __init__(
        self,
        tokenizer=None,
        chat_template=None,
        num_audio_tokens=256,
        image_token="<image>",
        video_token="<video>",
        audio_token="<audio>",
        **kwargs: Unpack[Bailing2ProcessorKwargs],
    ):
        self.image_token = image_token
        self.video_token = video_token
        self.audio_token = audio_token

        if chat_template is None:
            chat_template = tokenizer.chat_template

        self.audio_text = (DEFAULT_AU_START_TOKEN +
                           num_audio_tokens * DEFAULT_AUDIO_PATCH_TOKEN + DEFAULT_AU_END_TOKEN)

        self.gen_terminator = [
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.tokenizer = tokenizer
        self.image_processor = Bailing2ImageProcessor(min_pixels=78400, max_pixels=2007040)
        self.video_processor = Bailing2ImageProcessor(min_pixels=100352, max_pixels=602112)
        self.feature_extractor = SANMFeatureExtractor()
        super().__init__(tokenizer, chat_template=chat_template)
        

    def __call__(
        self,
        images: ImageInput = None,
        videos: VideoInput = None,
        audios: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        min_pixels: int = 78400,
        max_pixels: int = 2007040,
        min_pixels_video: int = 100352,
        max_pixels_video: int = 602112,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or torch Tensor.
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or torch Tensor.
            audios (`torch.Tensor`, `List[torch.Tensor]`, `List[List[torch.Tensor]]`):
                The sequence or batch of audios to be prepared. Each audio can be a 3D torch Tensor.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as a list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_num_patches** -- Patch number to be fed to a model. Returned when `images` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **pixel_values_audios** -- Pixel values of an audio input to be fed to a model. Returned when `audios` is not `None`.

        """
        output_kwargs = self._merge_kwargs(
            Bailing2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        video_inputs = {}
        audio_inputs = {}

        if images is not None:
            self.image_processor.min_pixels = min_pixels
            self.image_processor.max_pixels = max_pixels
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
            text = self._expand_image_tokens(text, image_grid_thw)

        if videos is not None:
            self.video_processor.min_pixels = min_pixels_video
            self.video_processor.max_pixels = max_pixels_video
            video_inputs = self.video_processor(images=None, videos=videos, do_resize=False,
                **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs["video_grid_thw"]
            text = self._expand_video_tokens(text, video_grid_thw)

        if audios is not None:
            audio_inputs = self.feature_extractor(audios, **output_kwargs["audio_kwargs"])
            if "attention_mask" in audio_inputs and audio_inputs["attention_mask"] is not None:
                audio_inputs["attention_mask_audio"] = audio_inputs.pop("attention_mask")

            text = self._expand_audio_tokens(text)
        # Padding side can be in TextKwargs but is not accepted by the tokenizer
        _ = output_kwargs["text_kwargs"].pop("padding_side", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs, **audio_inputs})

    def apply_system_template(self, text):
        def if_all_en(raw_text):
            for c_char in raw_text:
                unicode_val = ord(c_char)
                if 0x4E00 <= unicode_val <= 0x9FFF:
                    return False
            return True

        if if_all_en(text):
            sys_template = "You are a helpful language, vision and audio assistant. You are able to understand the visual and audio content that the user provides, and assist the user with a variety of tasks using natural language."
        else:
            sys_template = "你是一个有帮助的语言,视觉和音频助手。你能理解用户提供的视觉与音频内容，并用自然语言帮助用户完成各种任务。"
        sys_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_template} <|eot_id|>"
        return sys_prompt

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]]],
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        use_system=False,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
            tokenize (`bool`, *optional*, defaults to `False`):
                Whether to tokenize the output or not.
            use_system (`bool`, *optional*, defaults to `False`):
                Whether to use system template or not.
            **kwargs:
                Additional keyword arguments
        """

        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "No chat template is set for this processor. Please either set the `chat_template` attribute, "
                    "or provide a chat template as an argument. See "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating for more information."
                )
        if use_system:
            text = ""
            for message in conversation:
                text += "<|start_header_id|>" + message["role"] + "<|end_header_id|>\n\n"
                image_counts = str(message["content"]).count("<image>")
                video_counts = str(message["content"]).count("<video>")
                audio_counts = str(message["content"]).count("<audio>")

                for cid, content in enumerate(message["content"]):
                    if cid > 0:
                        text += "\n"
                    if content["type"] == "image":
                        num_images = 1 if isinstance(content["image"], (str, Image.Image)) else len(content["image"])
                        if image_counts < num_images:
                            image_placeholder = "<image>\n" * (num_images - image_counts)
                            text += image_placeholder.rstrip("\n")
                    # only one video supported now
                    elif content["type"] == "video":
                        assert video_counts <= 1, "Video count must be at most 1!"
                        if video_counts == 0:
                            text += "<video>"
                    elif content["type"] == "audio":
                        num_audios = 1 if isinstance(content["audio"], str) else len(content["audio"])
                        if audio_counts < num_audios:
                            audio_placeholder = "<audio>\n" * (num_audios - audio_counts)
                            text += audio_placeholder.rstrip("\n")
                    elif content["type"] == "text":
                        text += content['text']
                text += "<|eot_id|>"
            if kwargs.get("add_generation_prompt", False):
                text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            sys_prompt = self.apply_system_template(text)
            text = self.tokenizer.bos_token + sys_prompt + text
            return text
        else:
            return self.tokenizer.apply_chat_template(
                conversation, chat_template=chat_template, tokenize=tokenize, **kwargs
            )

    def process_vision_info(
        self,
        conversations,
    ):
        return process_vision_info(conversations)

    def _expand_image_tokens(
        self,
        text: List[TextInput],
        image_grid_thw: Union[List[int], int],
        special_token: str = "<image>",
    ):
        prompt_strings = []
        image_index = 0
        num_query_token = torch.prod(image_grid_thw, dim=1) // 4
        for sample in text:
            num_images = sample.count(special_token)
            if num_images > 0:
                for i in range(image_index, num_images + image_index):
                    img_text = DEFAULT_IM_START_TOKEN + num_query_token[
                        i] * DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN
                    sample = sample.replace(special_token, img_text, 1)
            image_index += num_images
            prompt_strings.append(sample)
        text = [sample for sample in prompt_strings]
        return text

    def _expand_video_tokens(
        self,
        text: List[TextInput],
        video_grid_thw: Union[List[int], int],
        special_token: str = "<video>",
    ):
        prompt_strings = []
        video_index = 0
        num_query_token = torch.prod(video_grid_thw, dim=1) // 4
        for sample in text:
            num_videos = sample.count(special_token)
            if num_videos > 0:
                for i in range(video_index, num_videos + video_index):
                    video_text = num_query_token[i] * DEFAULT_VIDEO_PATCH_TOKEN
                    video_text = DEFAULT_VID_START_TOKEN + video_text + DEFAULT_VID_END_TOKEN
                    sample = sample.replace(special_token, video_text, 1)
            video_index += num_videos
            prompt_strings.append(sample)
        text = [sample for sample in prompt_strings]
        return text

    def _expand_audio_tokens(
        self,
        text: List[TextInput],
        special_token: str = "<audio>",
    ):
        prompt_strings = []
        for sample in text:
            if special_token in sample:
                sample = sample.replace(special_token, self.audio_text)
            else:
                sample = sample + self.audio_text + "\n"
            prompt_strings.append(sample)
        text = [sample for sample in prompt_strings]
        return text

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names

        return list(
            dict.fromkeys(
                tokenizer_input_names + image_processor_input_names + audio_processor_input_names))
