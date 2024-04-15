import io
from copy import deepcopy

import random
from typing import List
import os
import json
from .Templates import ShortImageCaption
from .BaseDataset import BaseDataset


DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
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
IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"
BEGIN_OPTIONS = "<opt>"
END_OPTIONS = "</opt>"


class COCOCaptionDataset(BaseDataset):
    """Dataset for COCO Caption supervised fine-tuning."""
    def _construct_data_list(self, data_path) -> List:
        merge_data_dict = {}
        data_dict = json.load(open(data_path, "r"))
        annotation_lists = data_dict['annotations']
        for anno in annotation_lists:
            if anno['image_id'] not in merge_data_dict.keys():
                merge_data_dict[anno['image_id']] = {'image_id': anno['image_id'], 'caption': []}
            merge_data_dict[anno['image_id']]['caption'].append(anno['caption'])
        for image_info in data_dict['images']:
            merge_data_dict[image_info['id']]['file_name'] = image_info['file_name']
        list_data_dict = list(merge_data_dict.values())
        return list_data_dict

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
        image_path = item['file_name']
        use_item, image = self._read_image(image_path)
        return use_item, True, image 

    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        item = self.list_data_dict[i]
        self.conv.messages = []
        question = random.choice(ShortImageCaption)
        question = question.replace(" <image>", "")
        
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], random.choice(item['caption']))
        return self.conv.get_prompt()
