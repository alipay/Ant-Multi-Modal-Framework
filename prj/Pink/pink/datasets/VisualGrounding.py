import io
from copy import deepcopy
import random
import os
import json
from .Templates import VisualGrounding, GroundingCaption
from .BaseDataset import BaseDataset
from PIL import Image


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


class VisualGroundingDataset(BaseDataset):
    """Dataset for AOKVQA supervised fine-tuning."""
    def _construct_data_list(self, data_path):
        list_data_dict = json.load(open(data_path, "r"))
        if 'grounding_caption' in self.multimodal_cfg.keys():
            self.grounding_caption = self.multimodal_cfg['grounding_caption']
        else:
            self.grounding_caption = False
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
        use_item, image = self._read_image(image_path)
        return use_item, True, image

    def _get_data_item_val(self, i):
        sources = self.list_data_dict[i]
        result_dict = {}
        idx = sources["exp_id"]
        ref_id = sources['ref_id']
        image_id = sources['image_id']
        referring = sources['sentence']
        gt_ans = sources["bbox"]
        image_folder = self.multimodal_cfg['image_folder']
        with open(os.path.join(image_folder, image_id), "rb") as f:
            image = Image.open(io.BytesIO(f.read())).convert('RGB')
        orig_width, orig_height = image.size
        if self.expand2square:
            image = self._expand2square(image)
            offset_x, offset_y, _ = self._expand2square_offset(orig_width, orig_height)
            if orig_width > orig_height:
                orig_height = orig_width
            else:
                orig_width = orig_height
            gt_ans = [gt_ans[0] + offset_x, gt_ans[1] + offset_y, gt_ans[2] + offset_x, gt_ans[3] + offset_y]
        image_tensor = self.transforms(image)
        current_height, current_width = image_tensor.shape[-2:]
        images = image_tensor
        result_dict['idx'] = idx
        result_dict['sentence'] = referring
        result_dict['exp_id'] = idx
        result_dict['ref_id'] = ref_id
        result_dict['bbox'] = gt_ans
        result_dict['orig_wh'] = [orig_width, orig_height]
        result_dict['cur_wh'] = [current_width, current_height]
        result_dict['images'] = images
        return result_dict

    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        item = self.list_data_dict[i]
        if self.expand2square:
            offset_x, offset_y, scaled_ratio = self._expand2square_offset(self.orig_width, self.orig_height)
            scaled_bbox = [(item['bbox'][0] + offset_x) * scaled_ratio, (item['bbox'][1] + offset_y) * scaled_ratio, (item['bbox'][2] + offset_x) * scaled_ratio, (item['bbox'][3] + offset_y) * scaled_ratio]
        else: 
            scaled_width = 1 / self.orig_width
            scaled_height = 1 / self.orig_height
            scaled_bbox = [item['bbox'][0] * scaled_width, item['bbox'][1] * scaled_height, item['bbox'][2] * scaled_width, item['bbox'][3] * scaled_height]
        location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
        if self.grounding_caption:
            # TODO: fix the probability to 0.5
            if random.random() >= 0.5:
                question = random.choice(GroundingCaption)
                question = question.replace(" <image>", "")
                if self.add_marks:
                    question = question.replace("<objs>", (BEGIN_LOC + '{}' + END_LOC).format(location_tokens))
                else:
                    question = question.replace("<objs>", '{}'.format(location_tokens))
                caption = item["sentence"]
            else:
                question = random.choice(VisualGrounding)
                question = question.replace(" <image>", "")
                if self.add_marks:
                    question = question.replace("<expr>", (BEGIN_DESCRIPTION + '{}' + END_DESCRIPTION).format(item["sentence"]))
                else:
                    question = question.replace("<expr>", '{}'.format(item["sentence"]))
                caption = location_tokens
        else:
            question = random.choice(VisualGrounding)
            question = question.replace(" <image>", "")
            if self.add_marks:
                question = question.replace("<expr>", (BEGIN_DESCRIPTION + '{}' + END_DESCRIPTION).format(item["sentence"]))
            else:
                question = question.replace("<expr>", '{}'.format(item["sentence"]))
            caption = location_tokens
        self.conv.messages = []
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], caption)
        return self.conv.get_prompt()
