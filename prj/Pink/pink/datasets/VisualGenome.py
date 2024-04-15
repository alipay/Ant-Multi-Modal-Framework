import io
from copy import deepcopy

import random
from typing import List
import os
import json
from .Templates import VisualGrounding, RegionGroundingCaption, Relation_type1, Relation_type2, Relation_type3, Relation_type4, Detection_Type1, Detection_Type2, CoarseLocation_Type1, CoarseLocation_Type2, CoarseLocation_Type3, Counting_Type1, Counting_Type2
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
TASK_POOL = ["Relation", "CoarseLocation", "RegionGroundingCaption", "VisualGrounding", "Detection", "Counting"]


class VisualGenomeDataset(BaseDataset):
    """Dataset for GQA supervised fine-tuning."""
    def _construct_data_list(self, data_path) -> List:
        list_data_dict = json.load(open(data_path, "r"))
        object_data_dict = json.load(open(os.path.join(os.path.dirname(data_path), "objects.json"), "r"))
        relationship_data_dict = json.load(open(os.path.join(os.path.dirname(data_path), "relationships.json"), "r"))
        assert len(list_data_dict) == len(object_data_dict) and len(list_data_dict) == len(relationship_data_dict)
        self.object_data_dict = object_data_dict
        self.relationship_data_dict = relationship_data_dict
        if self.add_marks:
            self.object_template = BEGIN_LOC + '{}' + END_LOC
            self.area_template = BEGIN_DESCRIPTION + "{}" + END_DESCRIPTION
        else:
            self.object_template = '{}'
            self.area_template = "{}"
        self.TASK_POOL = self.multimodal_cfg.get("vg_task_pool", TASK_POOL)
        if "Caption" in self.TASK_POOL:
            self.TASK_POOL.remove("Caption")
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
        if len(item['regions']) == 0 or len(self.object_data_dict[i]['objects']) == 0 or len(self.relationship_data_dict[i]['relationships']) == 0:
            return False, {}, {}
        sources = random.choice(item['regions'])
        image_path = sources['image_id']
        use_item, image = self._read_image("{}.jpg".format(image_path))
        return use_item, True, image

    def get_visual_grounding_prompt(self, i):
        item = self.list_data_dict[i]
        orig_width = self.orig_width
        orig_height = self.orig_height
        sources = random.choice(item['regions']) 
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        if self.expand2square:
            offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
            scaled_bbox = [(sources['x'] + offset_x) * scaled_ratio, (sources['y'] + offset_y) * scaled_ratio, (sources['x'] + sources['width'] + offset_x) * scaled_ratio, (sources['y'] + sources['height'] + offset_y) * scaled_ratio]
        else:
            scaled_bbox = [sources['x'] * scaled_width, sources['y'] * scaled_height, (sources['x'] + sources['width']) * scaled_width, (sources['y'] + sources['height']) * scaled_height]
        location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
        question = random.choice(VisualGrounding)
        question = question.replace(" <image>", "")
        question = question.replace("<expr>", self.area_template.format(sources["phrase"]))
        
        before = question
        caption = location_tokens
        return before, caption

    def get_region_grounding_caption_prompt(self, i):
        item = self.list_data_dict[i]
        orig_width = self.orig_width
        orig_height = self.orig_height
        sources = random.choice(item['regions']) 
        question = random.choice(RegionGroundingCaption)
        question = question.replace(" <image>", "")
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        if self.expand2square:
            offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
            scaled_bbox = [(sources['x'] + offset_x) * scaled_ratio, (sources['y'] + offset_y) * scaled_ratio, (sources['x'] + sources['width'] + offset_x) * scaled_ratio, (sources['y'] + sources['height'] + offset_y) * scaled_ratio]
        else:
            scaled_bbox = [sources['x'] * scaled_width, sources['y'] * scaled_height, (sources['x'] + sources['width']) * scaled_width, (sources['y'] + sources['height']) * scaled_height]
        location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
        question = question.replace("<objs>", self.object_template.format(location_tokens))
        before = question
        caption = sources["phrase"]
        return before, caption

    def get_relation_prompt(self, i):
        downsample_relations = ['on', 'has', 'wearing', 'of', 'in']
        orig_width = self.orig_width
        orig_height = self.orig_height
        select_task_type = random.randint(0, 3)
        item = self.relationship_data_dict[i]
        for _ in range(100):
            select_relation_ship = random.choice(item['relationships'])
            if select_relation_ship['predicate'] in downsample_relations:
                if random.random() < 0.2:
                    break
            else:
                break

        if select_task_type == 0:
            question = random.choice(Relation_type1)
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox_1 = [(select_relation_ship['subject']['x'] + offset_x) * scaled_ratio, (select_relation_ship['subject']['y'] + offset_y) * scaled_ratio, (select_relation_ship['subject']['x'] + select_relation_ship['subject']['w'] + offset_x) * scaled_ratio, (select_relation_ship['subject']['y'] + select_relation_ship['subject']['h'] + offset_y) * scaled_ratio]
                scaled_bbox_2 = [(select_relation_ship['object']['x'] + offset_x) * scaled_ratio, (select_relation_ship['object']['y'] + offset_y) * scaled_ratio, (select_relation_ship['object']['x'] + select_relation_ship['object']['w'] + offset_x) * scaled_ratio, (select_relation_ship['object']['y'] + select_relation_ship['object']['h'] + offset_y) * scaled_ratio]
            else:
                scaled_width = 1 / orig_width
                scaled_height = 1 / orig_height
                scaled_bbox_1 = [select_relation_ship['subject']['x'] * scaled_width, select_relation_ship['subject']['y'] * scaled_height, (select_relation_ship['subject']['x'] + select_relation_ship['subject']['w']) * scaled_width, (select_relation_ship['subject']['y'] + select_relation_ship['subject']['h']) * scaled_height]
                scaled_bbox_2 = [select_relation_ship['object']['x'] * scaled_width, select_relation_ship['object']['y'] * scaled_height, (select_relation_ship['object']['x'] + select_relation_ship['object']['w']) * scaled_width, (select_relation_ship['object']['y'] + select_relation_ship['object']['h']) * scaled_height]

            location_tokens_1 = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox_1)
            location_tokens_2 = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox_2)
            question = question.replace("<object1>", "object1: {}".format(self.object_template.format(location_tokens_1)))
            question = question.replace("<object2>", "object2: {}".format(self.object_template.format(location_tokens_2)))
            before = question
            caption = "Object1 is {} object2".format(select_relation_ship['predicate'])
        elif select_task_type == 1:
            question = random.choice(Relation_type2)
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox_1 = [(select_relation_ship['subject']['x'] + offset_x) * scaled_ratio, (select_relation_ship['subject']['y'] + offset_y) * scaled_ratio, (select_relation_ship['subject']['x'] + select_relation_ship['subject']['w'] + offset_x) * scaled_ratio, (select_relation_ship['subject']['y'] + select_relation_ship['subject']['h'] + offset_y) * scaled_ratio]
            else:
                scaled_width = 1 / orig_width
                scaled_height = 1 / orig_height
                scaled_bbox_1 = [select_relation_ship['subject']['x'] * scaled_width, select_relation_ship['subject']['y'] * scaled_height, (select_relation_ship['subject']['x'] + select_relation_ship['subject']['w']) * scaled_width, (select_relation_ship['subject']['y'] + select_relation_ship['subject']['h']) * scaled_height]
            location_tokens_1 = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox_1)
            question = question.replace("<object>", self.object_template.format(location_tokens_1))
            if self.add_marks:
                question = question.replace("<relation>", BEGIN_RELATION + select_relation_ship['predicate'] + END_RELATION)
            else:
                question = question.replace("<relation>", select_relation_ship['predicate'])
            before = question
            object_id = select_relation_ship['subject']['object_id'] 
            relationship = select_relation_ship['predicate']

            has_relation_item = []
            for r in item['relationships']:
                if r['subject']['object_id'] == object_id and r['predicate'] == relationship:
                    area = r['object']['h'] * r['object']['w']
                    r['object'].update({"area": area})
                    has_relation_item.append(r['object'])

            has_relation_item = sorted(has_relation_item, key=lambda a:a['area'])
            has_relation_location_tokens = []
            for relation_item in has_relation_item:
                if self.expand2square:
                    scaled_bbox = [(relation_item['x'] + offset_x) * scaled_ratio, (relation_item['y'] + offset_y) * scaled_ratio, (relation_item['x'] + relation_item['w'] + offset_x) * scaled_ratio, (relation_item['y'] + relation_item['h'] + offset_y) * scaled_ratio]
                else:
                    scaled_bbox = [relation_item['x'] * scaled_width, relation_item['y'] * scaled_height, (relation_item['x'] + relation_item['w']) * scaled_width, (relation_item['y'] + relation_item['h']) * scaled_height]
                has_relation_location_tokens.append("[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox))
            caption = " ".join(has_relation_location_tokens)
        elif select_task_type == 2:
            question = random.choice(Relation_type3)
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox_1 = [(select_relation_ship['subject']['x'] + offset_x) * scaled_ratio, (select_relation_ship['subject']['y'] + offset_y) * scaled_ratio, (select_relation_ship['subject']['x'] + select_relation_ship['subject']['w'] + offset_x) * scaled_ratio, (select_relation_ship['subject']['y'] + select_relation_ship['subject']['h'] + offset_y) * scaled_ratio]
            else:
                scaled_width = 1 / orig_width
                scaled_height = 1 / orig_height
                scaled_bbox_1 = [select_relation_ship['subject']['x'] * scaled_width, select_relation_ship['subject']['y'] * scaled_height, (select_relation_ship['subject']['x'] + select_relation_ship['subject']['w']) * scaled_width, (select_relation_ship['subject']['y'] + select_relation_ship['subject']['h']) * scaled_height]
            location_tokens_1 = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox_1)
            question = question.replace("<object>", self.object_template.format(location_tokens_1))
            if self.add_marks:
                question = question.replace("<relation>", BEGIN_RELATION + select_relation_ship['predicate'] + END_RELATION)
            else:
                question = question.replace("<relation>", select_relation_ship['predicate'])
            before = question
            object_id = select_relation_ship['subject']['object_id'] 
            relationship = select_relation_ship['predicate']

            has_relation_item = []
            for r in item['relationships']:
                if r['subject']['object_id'] == object_id and r['predicate'] == relationship:
                    area = r['object']['h'] * r['object']['w']
                    r['object'].update({"area": area})
                    has_relation_item.append(r['object'])

            has_relation_item = sorted(has_relation_item, key=lambda a:a['area'])
            has_relation_location_tokens = []
            for relation_item in has_relation_item:
                has_relation_location_tokens.append(relation_item['names'][0])
            caption = " ".join(has_relation_location_tokens)
        elif select_task_type == 3:
            question = random.choice(Relation_type4)
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox_1 = [(select_relation_ship['subject']['x'] + offset_x) * scaled_ratio, (select_relation_ship['subject']['y'] + offset_y) * scaled_ratio, (select_relation_ship['subject']['x'] + select_relation_ship['subject']['w'] + offset_x) * scaled_ratio, (select_relation_ship['subject']['y'] + select_relation_ship['subject']['h'] + offset_y) * scaled_ratio]
            else:
                scaled_width = 1 / orig_width
                scaled_height = 1 / orig_height
                scaled_bbox_1 = [select_relation_ship['subject']['x'] * scaled_width, select_relation_ship['subject']['y'] * scaled_height, (select_relation_ship['subject']['x'] + select_relation_ship['subject']['w']) * scaled_width, (select_relation_ship['subject']['y'] + select_relation_ship['subject']['h']) * scaled_height]
            location_tokens_1 = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox_1)
            question = question.replace("<object>", self.object_template.format(location_tokens_1))
            if self.add_marks:
                question = question.replace("<relation>", BEGIN_RELATION + select_relation_ship['predicate'] + END_RELATION)
            else:
                question = question.replace("<relation>", select_relation_ship['predicate'])
            before = question
            object_id = select_relation_ship['subject']['object_id'] 
            relationship = select_relation_ship['predicate']

            has_relation_item = []
            for r in item['relationships']:
                if r['subject']['object_id'] == object_id and r['predicate'] == relationship:
                    area = r['object']['h'] * r['object']['w']
                    r['object'].update({"area": area})
                    has_relation_item.append(r['object'])

            has_relation_item = sorted(has_relation_item, key=lambda a:a['area'])
            has_relation_location_tokens = []
            for relation_item in has_relation_item:
                if self.expand2square:
                    scaled_bbox = [(relation_item['x'] + offset_x) * scaled_ratio, (relation_item['y'] + offset_y) * scaled_ratio, (relation_item['x'] + relation_item['w'] + offset_x) * scaled_ratio, (relation_item['y'] + relation_item['h'] + offset_y) * scaled_ratio]
                else:
                    scaled_bbox = [relation_item['x'] * scaled_width, relation_item['y'] * scaled_height, (relation_item['x'] + relation_item['w']) * scaled_width, (relation_item['y'] + relation_item['h']) * scaled_height]
                location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                has_relation_location_tokens.append("{} {}".format(location_token, relation_item['names'][0]))
            caption = " ".join(has_relation_location_tokens)
        return before, caption

    def get_coarse_location_prompt(self, i):
        select_task_type = random.randint(0, 2)
        orig_width = self.orig_width
        orig_height = self.orig_height
        coarse_location = ['top left', 'top right', 'bottom left', 'bottom right']
        item = self.object_data_dict[i]
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        if random.random() > 0.5:
            # NOTE: use location
            select_objects = random.choice(item['objects'])
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox = [(select_objects['x'] + offset_x) * scaled_ratio, (select_objects['y'] + offset_y) * scaled_ratio, (select_objects['x'] + select_objects['w'] + offset_x) * scaled_ratio, (select_objects['y'] + select_objects['h'] + offset_y) * scaled_ratio]
            else:
                scaled_bbox = [select_objects['x'] * scaled_width, select_objects['y'] * scaled_height, (select_objects['x'] + select_objects['w']) * scaled_width, (select_objects['y'] + select_objects['h']) * scaled_height]
            location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
            object_tokens = self.object_template.format(location_tokens)
            object_wh = (select_objects['w'], select_objects['h'])
        else:
            # NOTE: use description
            region_item = self.list_data_dict[i]
            select_objects = random.choice(region_item['regions'])
            object_tokens = self.area_template.format(select_objects['phrase'])
            object_wh = (select_objects['width'], select_objects['height'])
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox = [(select_objects['x'] + offset_x) * scaled_ratio, (select_objects['y'] + offset_y) * scaled_ratio, (select_objects['x'] + select_objects['width'] + offset_x) * scaled_ratio, (select_objects['y'] + select_objects['height'] + offset_y) * scaled_ratio]
            else:
                scaled_bbox = [select_objects['x'] * scaled_width, select_objects['y'] * scaled_height, (select_objects['x'] + select_objects['width']) * scaled_width, (select_objects['y'] + select_objects['height']) * scaled_height]
            location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)

        select_location = random.choice(coarse_location)
        select_object_center = (select_objects['x'] + object_wh[0] / 2, select_objects['y'] + object_wh[1] / 2)
        object_in_answer = []
        for obj in item['objects']:
            object_center = (obj['x'] + obj['w'] / 2, obj['y'] + obj['h'] / 2)
            if select_location == 'top left':
                if object_center[0] < select_object_center[0] and object_center[1] < select_object_center[1]:
                    obj['area'] = obj['w'] * obj['h']
                    object_in_answer.append(obj)
            elif select_location == 'top right':
                if object_center[0] > select_object_center[0] and object_center[1] < select_object_center[1]:
                    obj['area'] = obj['w'] * obj['h']
                    object_in_answer.append(obj)
            elif select_location == 'bottom left':
                if object_center[0] < select_object_center[0] and object_center[1] > select_object_center[1]:
                    obj['area'] = obj['w'] * obj['h']
                    object_in_answer.append(obj)
            elif select_location == 'bottom right':
                if object_center[0] > select_object_center[0] and object_center[1] > select_object_center[1]:
                    obj['area'] = obj['w'] * obj['h']
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
                        scaled_bbox = [(obj['x'] + offset_x) * scaled_ratio, (obj['y'] + offset_y) * scaled_ratio, (obj['x'] + obj['w'] + offset_x) * scaled_ratio, (obj['y'] + obj['h'] + offset_y) * scaled_ratio]
                    else:
                        scaled_bbox = [obj['x'] * scaled_width, obj['y'] * scaled_height, (obj['x'] + obj['w']) * scaled_width, (obj['y'] + obj['h']) * scaled_height]
                    location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                    answer_tokens.append("{} {}".format(location_token, obj['names'][0]))
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
                    answer_tokens.append("{}".format(obj['names'][0]))
                caption = " ".join(answer_tokens)
            else:
                caption = 'N/A'
        elif select_task_type == 2:
            question = random.choice(CoarseLocation_Type3)
            question = question.replace('<loc>', select_location)
            question = question.replace("<object>", object_tokens)
            if len(object_in_answer) > 0:
                answer_tokens = []
                for obj in object_in_answer:
                    if self.expand2square:
                        offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                        scaled_bbox = [(obj['x'] + offset_x) * scaled_ratio, (obj['y'] + offset_y) * scaled_ratio, (obj['x'] + obj['w'] + offset_x) * scaled_ratio, (obj['y'] + obj['h'] + offset_y) * scaled_ratio]
                    else:
                        scaled_bbox = [obj['x'] * scaled_width, obj['y'] * scaled_height, (obj['x'] + obj['w']) * scaled_width, (obj['y'] + obj['h']) * scaled_height]
                    location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                    answer_tokens.append("{}".format(location_token))
                caption = " ".join(answer_tokens)
            else:
                caption = 'N/A'

        # before = PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n" + "Question: " + "{}\nResponse:".format(question)
        before = question
        return before, caption

    def get_detection_prompt(self, i):
        orig_width = self.orig_width
        orig_height = self.orig_height
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        select_task_type = random.randint(0, 1)
        item = self.object_data_dict[i]

        class_names = []
        class_name2objects = {}
        for obj in item['objects']:
            for name in set(obj['names']):
                if name not in class_name2objects:
                    class_name2objects[name] = []
                class_name2objects[name].append(obj)
                class_names.append(name)
        counter = Counter(class_names)
        select_count = random.choice(list(set(counter.values())))
        select_class = random.choice([k for k, v in counter.items() if v == select_count])
        select_objects = random.choice(class_name2objects[select_class])
        select_obj_id = select_objects['object_id']

        object_in_answer = []
        for obj in class_name2objects[select_class]:
            assert select_class in obj['names']
            obj['area'] = obj['w'] * obj['h']
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
                    scaled_bbox = [(obj['x'] + offset_x) * scaled_ratio, (obj['y'] + offset_y) * scaled_ratio, (obj['x'] + obj['w'] + offset_x) * scaled_ratio, (obj['y'] + obj['h'] + offset_y) * scaled_ratio]
                else:
                    scaled_bbox = [obj['x'] * scaled_width, obj['y'] * scaled_height, (obj['x'] + obj['w']) * scaled_width, (obj['y'] + obj['h']) * scaled_height]
                location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                answer_tokens.append("{}".format(location_token))
            caption = " ".join(answer_tokens)
        elif select_task_type == 1:
            question = random.choice(Detection_Type2)
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox = [(select_objects['x'] + offset_x) * scaled_ratio, (select_objects['y'] + offset_y) * scaled_ratio, (select_objects['x'] + select_objects['w'] + offset_x) * scaled_ratio, (select_objects['y'] + select_objects['h'] + offset_y) * scaled_ratio]
            else:
                scaled_bbox = [select_objects['x'] * scaled_width, select_objects['y'] * scaled_height, (select_objects['x'] + select_objects['w']) * scaled_width, (select_objects['y'] + select_objects['h']) * scaled_height]
            location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
            object_tokens = self.object_template.format(location_tokens)
            question = question.replace("<object>", object_tokens)
            answer_tokens = []
            for obj in object_in_answer:
                if self.expand2square:
                    scaled_bbox = [(obj['x'] + offset_x) * scaled_ratio, (obj['y'] + offset_y) * scaled_ratio, (obj['x'] + obj['w'] + offset_x) * scaled_ratio, (obj['y'] + obj['h'] + offset_y) * scaled_ratio]
                else:
                    scaled_bbox = [obj['x'] * scaled_width, obj['y'] * scaled_height, (obj['x'] + obj['w']) * scaled_width, (obj['y'] + obj['h']) * scaled_height]
                location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                answer_tokens.append("{}".format(location_token))
            caption = " ".join(answer_tokens)

        before = question
        return before, caption

    def get_counting_prompt(self, i):
        orig_width = self.orig_width
        orig_height = self.orig_height
        scaled_width = 1 / orig_width
        scaled_height = 1 / orig_height
        select_task_type = random.randint(0, 1)
        item = self.object_data_dict[i]

        class_names = []
        class_name2objects = {}
        for obj in item['objects']:
            for name in set(obj['names']):
                if name not in class_name2objects:
                    class_name2objects[name] = []
                class_name2objects[name].append(obj)
                class_names.append(name)
        counter = Counter(class_names)
        select_count = random.choice(list(set(counter.values())))
        select_class = random.choice([k for k, v in counter.items() if v == select_count])
        select_objects = random.choice(class_name2objects[select_class])
        select_obj_id = select_objects['object_id']

        object_in_answer = []
        answer_tokens = []
        for obj in class_name2objects[select_class]:
            assert select_class in obj['names']
            obj['area'] = obj['w'] * obj['h']
            object_in_answer.append(obj)
        object_in_answer = sorted(object_in_answer, key=lambda a: a['area'])

        answer_tokens = []
        for obj in object_in_answer:
            if self.expand2square:
                offset_x, offset_y, scaled_ratio = self._expand2square_offset(orig_width, orig_height)
                scaled_bbox = [(obj['x'] + offset_x) * scaled_ratio, (obj['y'] + offset_y) * scaled_ratio, (obj['x'] + obj['w'] + offset_x) * scaled_ratio, (obj['y'] + obj['h'] + offset_y) * scaled_ratio]
            else:
                scaled_bbox = [obj['x'] * scaled_width, obj['y'] * scaled_height, (obj['x'] + obj['w']) * scaled_width, (obj['y'] + obj['h']) * scaled_height]
            location_token = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
            answer_tokens.append("{}".format(location_token))

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
                scaled_bbox = [(select_objects['x'] + offset_x) * scaled_ratio, (select_objects['y'] + offset_y) * scaled_ratio, (select_objects['x'] + select_objects['w'] + offset_x) * scaled_ratio, (select_objects['y'] + select_objects['h'] + offset_y) * scaled_ratio]
            else:
                scaled_bbox = [select_objects['x'] * scaled_width, select_objects['y'] * scaled_height, (select_objects['x'] + select_objects['w']) * scaled_width, (select_objects['y'] + select_objects['h']) * scaled_height]
            location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
            object_tokens = self.object_template.format(location_tokens)
            question = random.choice(Counting_Type2)
            question = question.replace("<object>", object_tokens)
            caption = "{}".format(len(object_in_answer))

        before = question
        return before, caption

    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        item = self.list_data_dict[i]
        select_task = random.choice(self.TASK_POOL)
        if select_task == 'Relation':
            before, caption = self.get_relation_prompt(i)
        elif select_task == 'CoarseLocation':
            before, caption = self.get_coarse_location_prompt(i)
        elif select_task == 'RegionGroundingCaption':
            before, caption = self.get_region_grounding_caption_prompt(i)
        elif select_task == 'VisualGrounding':
            before, caption = self.get_visual_grounding_prompt(i)
        elif select_task == 'Detection':
            before, caption = self.get_detection_prompt(i)
        elif select_task == 'Counting':
            before, caption = self.get_counting_prompt(i)

        self.conv.messages = []
        self.conv.append_message(self.conv.roles[0], before)
        self.conv.append_message(self.conv.roles[1], caption)
        return self.conv.get_prompt()
