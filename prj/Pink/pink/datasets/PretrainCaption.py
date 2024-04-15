import io
from typing import List
import json
from .BaseDataset import BaseDataset


DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"


class PretrainCaptionDataset(BaseDataset):
    """Dataset for multi-modal align pretrain."""
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
        image_path = item['clip_name']
        self.multimodal_cfg['image_folder'] = ""
        use_item, image = self._read_image(image_path)
        return use_item, True, image

    #NOTE: The pretrain is different from other dataset, we do not use instruct template
    def _construct_template(self, i):
        r"""
        modify this method to parse item
        Returns:
            prompt_sentence: str
        ```"""
        item = self.list_data_dict[i]
        cur_token_len = self.multimodal_cfg['cur_token_len'] # FIXME: 14 is hardcoded patch size
        before = PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n"
        after = item['caption']
        return (before, after)

    def _construct_target(self, prompt_sentence):
        before, after = prompt_sentence
        before_ids = self.tokenizer(
                before,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=False,
            ).input_ids[0]
        merge_ids = self.tokenizer(
            before + after + DEFAULT_EOS_TOKEN,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length + len(before_ids),
            truncation=True,
        ).input_ids[0]
        targets = merge_ids.clone()
        targets.masked_fill(targets == self.tokenizer.pad_token_id, IGNORE_INDEX)
        targets[:len(before_ids)] = IGNORE_INDEX

        return merge_ids, targets
