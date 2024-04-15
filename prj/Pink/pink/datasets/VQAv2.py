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


class VQAv2Dataset(BaseDataset):
    """Dataset for GQA supervised fine-tuning."""
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
        image_path = item['image_path']
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
        if self.add_marks:
            question = random.choice(QuestionAnswer)
            question = question.replace(" <image>", "")
            question = question.replace("<question>", "{}{}{}".format(BEGIN_QUESTION, item["question"], END_QUESTION))
        else:
            question = item['question'] + "\nAnswer the question using a single word or phrase."
        caption = item['annotation']['multiple_choice_answer']

        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], caption)
        return self.conv.get_prompt()
