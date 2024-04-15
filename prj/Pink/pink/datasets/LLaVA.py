import io
from copy import deepcopy

import random
from typing import List
import os
import json
from .Templates import QuestionAnswer
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


class LLaVADataset(BaseDataset):
    """Dataset for LLaVA instruct supervised fine-tuning."""
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
        image_path = item['image']
        use_item, image = self._read_image(image_path)
        return use_item, True, image

    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        sources = self.list_data_dict[i]
        self.conv.messages = []
        for index, item in enumerate(sources['conversations']):
            if index == 0:
                assert item['from'] == 'human'
                question = item['value'].replace("<image>\n", "")
                question = item['value'].replace("<image>", "")
                self.conv.append_message(self.conv.roles[0], question)
            else:
                if item['from'] == 'human':
                    role = self.conv.roles[0]
                else:
                    role = self.conv.roles[1]
                self.conv.append_message(role, item['value'])
        return self.conv.get_prompt()
