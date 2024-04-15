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


class OKVQADataset(BaseDataset):
    """Dataset for GQA supervised fine-tuning."""
    def _construct_data_list(self, data_path) -> List:
        annotations_list = json.load(open(data_path, "r"))
        questions_list = json.load(open(os.path.join(os.path.dirname(data_path), 'OpenEnded_mscoco_train2014_questions.json'), "r"))
        list_data_dict = []
        for anno, qs in zip(annotations_list['annotations'], questions_list['questions']):
            merge_answers = []
            for ans in anno['answers']:
                merge_answers.append(ans['raw_answer'])
            data_dict = {}
            data_dict.update(anno)
            data_dict['question'] = qs['question']
            data_dict['merge_answers'] = merge_answers
            assert qs['question_id'] == anno['question_id']
            list_data_dict.append(data_dict)
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
        use_item, image = self._read_image("{:0>12d}.jpg".format(image_path))
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
        counter = Counter(item['merge_answers'])
        caption = counter.most_common(1)[0][0]
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], caption)
        return self.conv.get_prompt()
