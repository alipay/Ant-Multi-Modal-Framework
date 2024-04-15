import transformers
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import torch.nn as nn
import io
from copy import deepcopy
import logging
from torchvision.transforms.functional import InterpolationMode
import random
from typing import Dict, Optional, Sequence, List
import os
from pink.conversation import conv_templates
import json


DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"


class BaseDataset(Dataset):
    """The Base Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict,
                 is_train=True,
                 ):
        super(BaseDataset, self).__init__()
        logging.warning("Loading data...")
        self.multimodal_cfg = multimodal_cfg
        self.add_marks = self.multimodal_cfg.get("add_marks", True)
        self.list_data_dict = self._construct_data_list(data_path)

        logging.warning("The number of training samples is {}".format(len(self.list_data_dict)))
        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer

        self.transforms = self._construct_transforms()
        
        self.conversation_template = self.multimodal_cfg.get("conversation_template")

        self.conv = conv_templates[self.conversation_template].copy()
        self.system = self.conv.system
        self.is_train = is_train
        self.expand2square = False

    def _construct_data_list(self, data_path) -> List:
        list_data_dict = json.load(open(data_path, "r"))
        return list_data_dict

    def _construct_transforms(self):
        if 'crop_size' in self.multimodal_cfg.keys():
            image_size = self.multimodal_cfg['image_size']
            crop_size = self.multimodal_cfg['crop_size']
        else:
            image_size = 224
            crop_size = 224
        norm_mean = (0.48145466, 0.4578275, 0.40821073)
        norm_std = (0.26862954, 0.26130258, 0.27577711)
        data_transforms = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        self.crop_size = crop_size
        self.image_mean = norm_mean
        return data_transforms

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        if self.is_train:
            while True:
                use_item, data_dict = self._get_data_item_train(i)
                if use_item:
                    break
                else:
                    i = random.randint(0, self.__len__() - 1)
            return data_dict
        else:
            return self._get_data_item_val(i)

    def _expand2square(self, pil_img):
        width, height = pil_img.size
        background_color = tuple(int(x*255) for x in self.norm_mean)
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def _expand2square_offset(self, orig_width, orig_height):
        if orig_width == orig_height:
            return 0.0, 0.0, 1 / orig_width
        if orig_width > orig_height:
            return 0.0, (orig_width - orig_height) // 2, 1 / orig_width
        assert orig_width < orig_height
        return (orig_height - orig_width) // 2, 0.0, 1 / orig_height

    def _read_image(self, image_path):
        r"""
        Returns:
            use_item (bool): whether successfully get image
            image (Tensor): image tensor
        ```"""
        base_path = self.multimodal_cfg['base_path']
        image_folder = self.multimodal_cfg['image_folder']
        try:
            with open(os.path.join(base_path, image_folder, image_path), "rb") as f:
                image = Image.open(io.BytesIO(f.read())).convert('RGB')
            orig_width, orig_height = image.size
            self.orig_width = orig_width
            self.orig_height = orig_height
        except:
            print("file {} does not exist".format(image_path))
            return False, {}
        if self.expand2square:
            image = self._expand2square(image, tuple(int(x*255) for x in self.image_mean))
            if self.orig_width > self.orig_height:
                self.orig_height = self.orig_width
            else:
                self.orig_width = self.orig_height

        image = self.transforms(image)
        return True, image

    def _parse_image(self, i):
        r"""
        modify this method to parse image
        Returns:
            Dict:
                use_item (bool): whether successfully get image
                has_image (bool): whether has image, model should deal with pure text input 
                image (Tensor): image tensor
        ```"""
        item = self.list_data_dict[i]
        image_path = item['clip_path']
        use_item, image = self._read_image(image_path)
        # image = torch.zeros(3, self.crop_size, self.crop_size)
        return use_item, True, image

    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        self.conv.messages = []
        item = self.list_data_dict[i]
        prompt_sentence = self.conv.get_prompt()
        return prompt_sentence

    def _construct_target(self, prompt_sentence):
        inputs = self.tokenizer(prompt_sentence,
                    return_tensors="pt",
                    padding="longest",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True).input_ids[0]
        target = inputs.clone()
        sep = self.conv.sep_template
        rounds = prompt_sentence.split(self.conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer(rou).input_ids) + self.conv.offset
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2   # <s> <space>
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        return inputs, target
  
    def _get_data_item_train(self, i) -> Dict[str, torch.Tensor]:
        data_dict = {}
        use_item, has_image, image = self._parse_image(i)
        if not use_item:
            return False, {}

        cur_token_len = self.multimodal_cfg['cur_token_len'] # FIXME: 14 is hardcoded patch size
        if has_image:
            self.conv.set_system(PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
        else:
            self.conv.set_system(PREFIX_NO_IMAGE)
        prompt_sentence = self._construct_template(i)

        data_dict["image"] = image
        data_dict['has_image'] = has_image
        # print(prompt_sentence)
        inputs, targets = self._construct_target(prompt_sentence)

        if len(inputs) >= (self.tokenizer.model_max_length - 20):
            return False, {}
        data_dict.update(
            dict(input_ids=inputs,
                labels=targets)
        )

        return True, data_dict

    def _get_data_item_val(self, i) -> Dict[str, torch.Tensor]:
        return self._get_data_item_train(i)
