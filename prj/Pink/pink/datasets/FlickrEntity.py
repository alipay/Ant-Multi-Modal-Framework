import io
from copy import deepcopy

import random
from typing import List
import os
import json
from .Templates import CaptionGrounding
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


class FlickrEntityDataset(BaseDataset):
    """Dataset for Flickr Entity supervised fine-tuning."""
    def _construct_data_list(self, data_path) -> List:
        list_data_dict = [json.loads(line) for line in open(data_path, "r")]
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
        image_path = item['image_id']
        self.multimodal_cfg['pcache_path'] = ""
        use_item, image = self._read_image("{}.jpg".format(image_path))
        return use_item, True, image

    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        item = self.list_data_dict[i]
        ENTITY_START = "<ph_st>"
        ENTITY_END = "<ph_ed>"
        self.conv.messages = []
        question = random.choice(CaptionGrounding)
        question = question.replace(" <image>", "")

        answer = item['sentence'].replace(ENTITY_START, "")
        location_strs = []
        for bbox_ids in item['boxes_seq']:
            location_str = []
            for bbox_id in bbox_ids:
                bbox = item['boxes'][bbox_id]
                if self.expand2square:
                    offset_x, offset_y, scaled_ratio = self._expand2square_offset(self.orig_width, self.orig_height)
                    scaled_bbox = [(bbox[0] + offset_x) * scaled_ratio, (bbox[1] + offset_y) * scaled_ratio, (bbox[2] + offset_x) * scaled_ratio, (bbox[3] + offset_y) * scaled_ratio]
                else:
                    scaled_width = 1 / self.orig_width
                    scaled_height = 1 / self.orig_height
                    scaled_bbox = [bbox[0] * scaled_width, bbox[1] * scaled_height, bbox[2] * scaled_width, bbox[3] * scaled_height]
                location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                location_str.append((location_tokens, area))
            location_str = sorted(location_str, key=lambda a:a[1])
            location_strs.append(" ".join([l[0] for l in location_str]))
        answer = answer.split(ENTITY_END)
        merge_answer = ""
        final_index = len(answer)
        for index, a in enumerate(answer):
            if index == (final_index - 1):
                merge_answer += a
            else:
                merge_answer += a + " " + location_strs[index]

        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], merge_answer)
        return self.conv.get_prompt()
