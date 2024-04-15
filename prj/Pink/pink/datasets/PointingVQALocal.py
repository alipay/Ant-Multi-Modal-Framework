import io
from copy import deepcopy

import random
from typing import List
import os
import json
from .Templates import QuestionAnswer
from .BaseDataset import BaseDataset
from collections import Counter


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


class PointingVQALocalDataset(BaseDataset):
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
        image_path = item['id']
        use_item, image = self._read_image("{}.jpg".format(image_path))
        return use_item, True, image

    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        item = self.list_data_dict[i]
        self.conv.messages = []
        scaled_width = 1 / self.orig_width
        scaled_height = 1 / self.orig_height
        if self.expand2square:
            offset_x, offset_y, scaled_ratio = self._expand2square_offset(self.orig_width, self.orig_height)
            scaled_bbox = [(item['all_objs']['x'] + offset_x) * scaled_ratio, (item['all_objs']['y'] + offset_y) * scaled_ratio, (item['all_objs']['x'] + offset_x + item['all_objs']['w']) * scaled_ratio, (item['all_objs']['y'] + offset_y + item['all_objs']['h']) * scaled_ratio]
        else:
            scaled_bbox = [item['all_objs']['x'] * scaled_width, item['all_objs']['y'] * scaled_height, (item['all_objs']['x'] + item['all_objs']['w']) * scaled_width, (item['all_objs']['y'] + item['all_objs']['h']) * scaled_height]
        location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
        if self.add_marks:
            question = "{} {}".format(item["question"], BEGIN_LOC + location_tokens + END_LOC)
        else:
            question = "{} {}".format(item["question"], location_tokens)
        caption = item['points']['ans']
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], caption)
        return self.conv.get_prompt()
