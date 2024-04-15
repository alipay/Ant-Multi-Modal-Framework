import io
from copy import deepcopy
import random
import os
import json
from .Templates import ChoiceQuestionAnswer
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


class V7WGroundingDataset(BaseDataset):
    """Dataset for AOKVQA supervised fine-tuning."""
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
        scaled_width = 1 / self.orig_width
        scaled_height = 1 / self.orig_height
        answer_bbox = [item['answer_box']['x'] * scaled_width, item['answer_box']['y'] * scaled_height, (item['answer_box']['x'] + item['answer_box']['width']) * scaled_width, (item['answer_box']['y'] + item['answer_box']['height']) * scaled_height]

        multi_choice_bboxes = []
        for bbox in item['multiple_choices_box']:
            multi_choice_bboxes.append([bbox['x'] * scaled_width, bbox['y'] * scaled_height, (bbox['x'] + bbox['width']) * scaled_width, (bbox['y'] + bbox['height']) * scaled_height])
        multi_choice_bboxes.append(answer_bbox)

        random.shuffle(multi_choice_bboxes)
        multi_choice_tokens = []
        for bbox in multi_choice_bboxes:
            multi_choice_tokens.append("[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*bbox))

        index_to_options = {0: "A", 1: "B", 2: "C", 3: "D"}
        question = random.choice(ChoiceQuestionAnswer)
        question = question.replace(" <image>", "")
        if self.add_marks:
            question = question.replace("<question>", "{}{}{}".format(BEGIN_QUESTION, item["question"], END_QUESTION))
        else:
            question = question.replace("<question>", "{}".format(item["question"]))
        options = ""
        for index, opt in enumerate(multi_choice_tokens):
            if self.add_marks:
                options += index_to_options[index] + ". {}{}{}\n".format(BEGIN_LOC, opt, END_LOC)
            else:
                options += index_to_options[index] + ". {}\n".format(opt)
        options = options.rstrip("\n")
        if self.add_marks:
            question = question.replace("<option>", "{}{}{}".format(BEGIN_OPTIONS, options, END_OPTIONS))
        else:
            question = question.replace("<option>", "{}".format(options))
        
        caption = "The answer is {}.".format(index_to_options[multi_choice_bboxes.index(answer_bbox)])

        self.conv.messages = []
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], caption)
        return self.conv.get_prompt()
