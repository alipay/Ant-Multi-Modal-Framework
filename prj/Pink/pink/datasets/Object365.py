import io
from copy import deepcopy

import random
from typing import List
import os
import json
from .Templates import VisualGrounding, GroundingCaption, Detection_Type1, Detection_Type2, CoarseLocation_Type1, CoarseLocation_Type2, CoarseLocation_Type3, Counting_Type1, Counting_Type2, ShortImageCaption
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
TASK_POOL = ["CoarseLocation", "RegionGroundingCaption", "VisualGrounding", "Detection", "Counting"]


class Object365Dataset(BaseDataset):
    """Dataset for GQA supervised fine-tuning."""
    def _construct_data_list(self, data_path) -> List:
        list_data_dict = [json.loads(line) for line in open(data_path, "r")]
        if self.add_marks:
            self.object_template = BEGIN_LOC + '{}' + END_LOC
            self.area_template = BEGIN_DESCRIPTION + "{}" + END_DESCRIPTION
        else:
            self.object_template = '{}'
            self.area_template = '{}'

        self.TASK_POOL = self.multimodal_cfg.get("task_pool_365", TASK_POOL)
        if "Relation" in self.TASK_POOL:
            self.TASK_POOL.remove("Relation")
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
        if self.auth is not None:
            image_path = os.path.join("object365/train", image_path)
            use_item, image = self._read_image_from_oss(image_path)
        else:
            use_item, image = self._read_image(image_path)
        return use_item, True, image

    def get_caption_prompt(self, i):
        item = self.list_data_dict[i]
        question = random.choice(ShortImageCaption)
        question = question.replace(" <image>", "")
        caption = item['generate_caption']
        return question, caption

    def get_visual_grounding_prompt(self, i):
        item = self.list_data_dict[i]
        orig_width = self.orig_width
        orig_height = self.orig_height
        sources = random.choice(item['pred'])
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        if self.expand2square:
            offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
            scaled_bbox = [(sources['bbox'][0] + offset_x) * scaled_ratio, (sources['bbox'][1] + offset_y) * scaled_ratio, (sources['bbox'][2] + offset_x) * scaled_ratio, (sources['bbox'][3] + offset_y) * scaled_ratio]
        else:
            scaled_bbox = [sources['bbox'][0] * scaled_width, sources['bbox'][1] * scaled_height, sources['bbox'][2] * scaled_width, sources['bbox'][3] * scaled_height]
        location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
        question = random.choice(VisualGrounding)
        question = question.replace(" <image>", "")
        question = question.replace("<expr>", self.area_template.format(sources["caption"]))
        caption = location_tokens
        return question, caption

    def get_grounding_caption_prompt(self, i):
        item = self.list_data_dict[i]
        orig_width = self.orig_width
        orig_height = self.orig_height
        sources = random.choice(item['pred'])
        question = random.choice(GroundingCaption)
        question = question.replace(" <image>", "")
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        if self.expand2square:
            offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
            scaled_bbox = [(sources['bbox'][0] + offset_x) * scaled_ratio, (sources['bbox'][1] + offset_y) * scaled_ratio, (sources['bbox'][2] + offset_x) * scaled_ratio, (sources['bbox'][3] + offset_y) * scaled_ratio]
        else:
            scaled_bbox = [sources['bbox'][0] * scaled_width, sources['bbox'][1] * scaled_height, sources['bbox'][2] * scaled_width, sources['bbox'][3] * scaled_height]
        location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
        question = question.replace("<objs>", self.object_template.format(location_tokens))
        caption = sources["caption"]
        return question, caption

    def get_coarse_location_prompt(self, i):
        orig_width = self.orig_width
        orig_height = self.orig_height
        select_task_type = random.randint(0, 2)
        coarse_location = ['top left', 'top right', 'bottom left', 'bottom right']
        item = self.list_data_dict[i]
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        if random.random() > 0.5 or len(item['pred']) == 0:
            while True:
                select_objects = random.choice(item['anno'])
                if select_objects['ignore'] == 0:
                    break
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox = [(select_objects['bbox'][0] + offset_x) * scaled_ratio, (select_objects['bbox'][1] + offset_y) * scaled_ratio, (select_objects['bbox'][2] + offset_x) * scaled_ratio, (select_objects['bbox'][3] + offset_y) * scaled_ratio]
            else:
                scaled_bbox = [select_objects['bbox'][0] * scaled_width, select_objects['bbox'][1] * scaled_height, select_objects['bbox'][2] * scaled_width, select_objects['bbox'][3] * scaled_height]
            location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
            object_tokens = self.object_template.format(location_tokens)
            object_wh = (select_objects['bbox'][2] - select_objects['bbox'][0], select_objects['bbox'][3] - select_objects['bbox'][1])
        else:
            # NOTE: use description
            select_objects = random.choice(item['pred'])
            object_tokens = self.area_template.format(select_objects['caption'])
            object_wh = (select_objects['bbox'][2] - select_objects['bbox'][0], select_objects['bbox'][3] - select_objects['bbox'][1])

        select_location = random.choice(coarse_location)
        select_object_center = ((select_objects['bbox'][0] + select_objects['bbox'][2]) / 2, (select_objects['bbox'][1] + select_objects['bbox'][3]) / 2)
        object_in_answer = []
        for obj in item['anno']:
            if obj['ignore'] != 0:
                continue
            obj_center = ((obj['bbox'][0] + obj['bbox'][2]) / 2, (obj['bbox'][1] + obj['bbox'][3]) / 2)
            if select_location == 'top left':
                if obj_center[0] < select_object_center[0] and obj_center[1] < select_object_center[1]:
                    obj['area'] = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
                    object_in_answer.append(obj)
            elif select_location == 'top right':
                if obj_center[0] > select_object_center[0] and obj_center[1] < select_object_center[1]:
                    obj['area'] = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
                    object_in_answer.append(obj)
            elif select_location == 'bottom left':
                if obj_center[0] < select_object_center[0] and obj_center[1] > select_object_center[1]:
                    obj['area'] = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
                    object_in_answer.append(obj)
            elif select_location == 'bottom right':
                if obj_center[0] > select_object_center[0] and obj_center[1] > select_object_center[1]:
                    obj['area'] = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
                    object_in_answer.append(obj)
        if len(object_in_answer) > 0:
            object_in_answer = sorted(object_in_answer, key=lambda a: a['area'])
        if select_task_type == 0:
            question = random.choice(CoarseLocation_Type1)
            question = question.replace('<loc>', select_location)
            question = question.replace("<object>", object_tokens)
            if len(object_in_answer) > 0:
                answer_tokens = []
                for obj in object_in_answer:
                    if self.expand2square:
                        offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                        scaled_bbox = [(obj['bbox'][0] + offset_x) * scaled_ratio, (obj['bbox'][1] + offset_y) * scaled_ratio, (obj['bbox'][2] + offset_x) * scaled_ratio, (obj['bbox'][3] + offset_y) * scaled_ratio]
                    else:
                        scaled_bbox = [obj['bbox'][0] * scaled_width, obj['bbox'][1] * scaled_height, obj['bbox'][2] * scaled_width, obj['bbox'][3] * scaled_height]
                    location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                    answer_tokens.append("{} {}".format(location_token, obj['category_name']))
                caption = " ".join(answer_tokens)
            else:
                caption = 'N/A'
        elif select_task_type == 1:
            question = random.choice(CoarseLocation_Type2)
            question = question.replace('<loc>', select_location)
            question = question.replace("<object>", object_tokens)
            if len(object_in_answer) > 0:
                answer_tokens = []
                for obj in object_in_answer:
                    answer_tokens.append("{}".format(obj['category_name']))
                caption = " ".join(answer_tokens)
            else:
                caption = "N/A"
        elif select_task_type == 2:
            question = random.choice(CoarseLocation_Type3)
            question = question.replace('<loc>', select_location)
            question = question.replace("<object>", object_tokens)
            if len(object_in_answer) > 0:
                answer_tokens = []
                for obj in object_in_answer:
                    if self.expand2square:
                        offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                        scaled_bbox = [(obj['bbox'][0] + offset_x) * scaled_ratio, (obj['bbox'][1] + offset_y) * scaled_ratio, (obj['bbox'][2] + offset_x) * scaled_ratio, (obj['bbox'][3] + offset_y) * scaled_ratio]
                    else:
                        scaled_bbox = [obj['bbox'][0] * scaled_width, obj['bbox'][1] * scaled_height, obj['bbox'][2] * scaled_width, obj['bbox'][3] * scaled_height]
                    location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                    answer_tokens.append("{}".format(location_token))
                caption = " ".join(answer_tokens)
            else:
                caption = "N/A"

        return question, caption

    def get_detection_prompt(self, i):
        orig_width = self.orig_width
        orig_height = self.orig_height
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        select_task_type = random.randint(0, 1)
        item = self.list_data_dict[i]

        class_names = []
        class_name2objects = {}
        for obj in item['anno']:
            if obj['ignore'] == 1:
                continue
            if obj['category_name'] not in class_name2objects:
                class_name2objects[obj['category_name']] = []
            class_name2objects[obj['category_name']].append(obj)
            class_names.append(obj['category_name'])
        counter = Counter(class_names)
        select_count = random.choice(list(set(counter.values())))
        select_class = random.choice([k for k, v in counter.items() if v == select_count])
        select_objects = random.choice(class_name2objects[select_class])
        select_obj_id = select_objects['id']

        object_in_answer = []
        for obj in class_name2objects[select_class]:
            assert obj['category_name'] == select_class
            obj['area'] = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
            object_in_answer.append(obj)
        if len(object_in_answer) > 1:
            object_in_answer = sorted(object_in_answer, key=lambda a: a['area'])
        else:
            select_task_type = 0

        if select_task_type == 0:
            question = random.choice(Detection_Type1)
            if self.add_marks:
                question = question.replace('<category>', BEGIN_CLS + select_class + END_CLS)
            else:
                question = question.replace('<category>', select_class)
            answer_tokens = []
            for obj in object_in_answer:
                if self.expand2square:
                    offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                    scaled_bbox = [(obj['bbox'][0] + offset_x) * scaled_ratio, (obj['bbox'][1] + offset_y) * scaled_ratio, (obj['bbox'][2] + offset_x) * scaled_ratio, (obj['bbox'][3] + offset_y) * scaled_ratio]
                else:
                    scaled_bbox = [obj['bbox'][0] * scaled_width, obj['bbox'][1] * scaled_height, obj['bbox'][2] * scaled_width, obj['bbox'][3] * scaled_height]
                location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                answer_tokens.append("{}".format(location_token))
            caption = " ".join(answer_tokens)
        elif select_task_type == 1:
            question = random.choice(Detection_Type2)
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox = [(select_objects['bbox'][0] + offset_x) * scaled_ratio, (select_objects['bbox'][1] + offset_y) * scaled_ratio, (select_objects['bbox'][2] + offset_x) * scaled_ratio, (select_objects['bbox'][3] + offset_y) * scaled_ratio]
            else:
                scaled_bbox = [select_objects['bbox'][0] * scaled_width, select_objects['bbox'][1] * scaled_height, select_objects['bbox'][2] * scaled_width, select_objects['bbox'][3] * scaled_height]
            location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
            object_tokens = self.object_template.format(location_tokens)
            question = question.replace("<object>", object_tokens)
            answer_tokens = []
            for obj in object_in_answer:
                if self.expand2square:
                    scaled_bbox = [(obj['bbox'][0] + offset_x) * scaled_ratio, (obj['bbox'][1] + offset_y) * scaled_ratio, (obj['bbox'][2] + offset_x) * scaled_ratio, (obj['bbox'][3] + offset_y) * scaled_ratio]
                else:
                    scaled_bbox = [obj['bbox'][0] * scaled_width, obj['bbox'][1] * scaled_height, obj['bbox'][2] * scaled_width, obj['bbox'][3] * scaled_height]
                location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                answer_tokens.append("{}".format(location_token))
            caption = " ".join(answer_tokens)

        return question, caption

    def get_counting_prompt(self, i):
        orig_width = self.orig_width
        orig_height = self.orig_height
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        select_task_type = random.randint(0, 1)
        item = self.list_data_dict[i]

        class_names = []
        class_name2objects = {}
        for obj in item['anno']:
            if obj['ignore'] == 1:
                continue
            if obj['category_name'] not in class_name2objects:
                class_name2objects[obj['category_name']] = []
            class_name2objects[obj['category_name']].append(obj)
            class_names.append(obj['category_name'])
        counter = Counter(class_names)
        select_count = random.choice(list(set(counter.values())))
        select_class = random.choice([k for k, v in counter.items() if v == select_count])
        select_objects = random.choice(class_name2objects[select_class])

        object_in_answer = []
        for obj in class_name2objects[select_class]:
            assert obj['category_name'] == select_class
            obj['area'] = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
            object_in_answer.append(obj)
        if len(object_in_answer) > 1:
            object_in_answer = sorted(object_in_answer, key=lambda a: a['area'])

        if select_task_type == 0:
            question = random.choice(Counting_Type1)
            if self.add_marks:
                question = question.replace('<category>', BEGIN_CLS + select_class + END_CLS)
            else:
                question = question.replace('<category>', select_class)
            caption = "{}".format(len(object_in_answer))
        elif select_task_type == 1:
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox = [(select_objects['bbox'][0] + offset_x) * scaled_ratio, (select_objects['bbox'][1] + offset_y) * scaled_ratio, (select_objects['bbox'][2] + offset_x) * scaled_ratio, (select_objects['bbox'][3] + offset_y) * scaled_ratio]
            else:
                scaled_bbox = [select_objects['bbox'][0] * scaled_width, select_objects['bbox'][1] * scaled_height, select_objects['bbox'][2] * scaled_width, select_objects['bbox'][3] * scaled_height]
            location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
            object_tokens = self.object_template.format(location_tokens)
            question = random.choice(Counting_Type2)
            question = question.replace("<object>", object_tokens)
            caption = "{}".format(len(object_in_answer))

        return question, caption

    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        item = self.list_data_dict[i]
        while True:
            select_task = random.choice(self.TASK_POOL)
            if select_task == 'CoarseLocation':
                before, caption = self.get_coarse_location_prompt(i)
            elif select_task == 'RegionGroundingCaption':
                if len(item['pred']) == 0:
                    continue
                before, caption = self.get_grounding_caption_prompt(i)
            elif select_task == 'VisualGrounding':
                if len(item['pred']) == 0:
                    continue
                before, caption = self.get_visual_grounding_prompt(i)
            elif select_task == 'Detection':
                before, caption = self.get_detection_prompt(i)
            elif select_task == 'Counting':
                before, caption = self.get_counting_prompt(i)
            elif select_task == 'Caption':
                before, caption = self.get_caption_prompt(i)
            break

        self.conv.messages = []
        self.conv.append_message(self.conv.roles[0], before)
        self.conv.append_message(self.conv.roles[1], caption)
        return self.conv.get_prompt()
