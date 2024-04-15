import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from pink.model import adapter, mark_only_adapter_as_trainable
from pink.conversation import conv_templates
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria, CLIPModel
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from pink import *
import re

import io
from PIL import Image
import random
import math
from pink.datasets.Templates import VisualGrounding, GroundingCaption, CaptionGrounding, ShortImageCaption
from torch.utils.data import Dataset, DataLoader


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
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


class SimpleCollator(object):
    def __call__(self, instances):
        batch_images = []
        batch_items = []
        batch_orig_height = []
        batch_orig_width = []
        for i, line in enumerate(instances):
            image, item, orig_width, orig_height = line
            batch_images.append(image)
            batch_items.append(item)
            batch_orig_width.append(orig_width)
            batch_orig_height.append(orig_height)
        return batch_images, batch_items, batch_orig_width, batch_orig_height
            

class SimpleDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_list: str,
                 transforms: dict,
                 ):
        super(SimpleDataset, self).__init__()
        self.data_list = data_list
        print("The number of training samples is {}".format(len(data_list)))
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        while True:
            use_item, image, item, orig_width, orig_height = self.parse_item(i)
            if use_item:
                break
            else:
                i = random.randint(0, self.__len__() - 1)
        return image, item, orig_width, orig_height

    def parse_item(self, i):
        item = self.data_list[i]
        image_id = item['image_id']
        try:
            with open(os.path.join(args.image_folder, image_id), "rb") as f:
                image = Image.open(io.BytesIO(f.read())).convert('RGB')
        except:
            print("file {} does not exist in pcache".format(image_id))
            return False, 0, 0, 0, 0
        orig_width, orig_height = image.size
        image = self.transforms(image)
        return True, image, item, orig_width, orig_height


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def expand2square_offset(orig_width, orig_height):
    if orig_width == orig_height:
        return 0.0, 0.0, 1 / orig_width
    if orig_width > orig_height:
        return 0.0, (orig_width - orig_height) // 2, 1 / orig_width
    assert orig_width < orig_height
    return (orig_height - orig_width) // 2, 0.0, 1 / orig_height


def patch_config(config):
    cfg = AutoConfig.from_pretrained(config)
    print(cfg)
    # if not hasattr(cfg, "mm_vision_tower"):
    #     print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
    #     for k, v in patch_dict.items():
    #         setattr(cfg, k, v)
    #     cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

@torch.no_grad()
def eval_model(args):
    # Model
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')
    patch_config(model_name)
    config = AutoConfig.from_pretrained(model_name, use_cache=True)
    if config.llama_path != model_name:
        # need to merge parameters
        llama_path = config.llama_path
        weight_map_index = json.load(open(os.path.join(llama_path, "pytorch_model.bin.index.json"), "r"))
        shard_files = list(set(weight_map_index["weight_map"].values()))
        loaded_keys = weight_map_index["weight_map"].keys()
        state_dict = {}
        for index, shard_file in enumerate(shard_files):
            state_dict.update(torch.load(os.path.join(llama_path, shard_file), map_location="cpu"))
        peft_parameters = torch.load(os.path.join(model_name, "saved_parameters.pth"), map_location="cpu")
        for k, v in peft_parameters.items():
            state_dict[k] = v
    else:
        state_dict = None

    model = AutoModelForCausalLM.from_pretrained(None, config=config, state_dict=state_dict)
    for name, param in model.model.named_parameters():
        if not ("adapter_" in name or "lora_" in name):
            param.data = param.data.half()
    model.lm_head.to(torch.float16)
    model = model.cuda()
    print(model)
    crop_size = model.config.crop_size
    norm_mean = (0.48145466, 0.4578275, 0.40821073)
    norm_std = (0.26862954, 0.26130258, 0.27577711)
    image_processor = transforms.Compose(
            [
                transforms.Resize(
                    (crop_size, crop_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
    image_token_len = model.config.num_patches

    model.eval()
    conv = conv_templates[args.conv_mode].copy()

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if os.path.exists(answers_file):
        ans_file = open(answers_file, "r")
        solved_file = [json.loads(line) for line in ans_file]
        ans_file.close()
        ans_file = open(answers_file, "a")
        solved_file_path = {}
        for f in solved_file:
            solved_file_path[f['image_id']] = 0
        del solved_file
    else:
        ans_file = open(answers_file, "w")
        solved_file_path = {}

    ori_questions = json.load(open(os.path.join(args.question_file), "r"))
    ori_questions = get_chunk(ori_questions, args.num_chunks, args.chunk_idx)

    questions = []
    for q in ori_questions:
        if q['image_id'] not in solved_file_path.keys():
            questions.append(q)
    del ori_questions

    dataset = SimpleDataset(questions, image_processor)
    # If you find the output is None, set batch_size = 1 
    batch_size = 4
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=SimpleCollator())
    
    pattern = re.compile(r'[0-9]\.[0-9][0-9][0-9]')

    for batch_images, items, orig_widths, orig_heights in tqdm(data_loader):
        for images, line, orig_width, orig_height in zip(batch_images, items, orig_widths, orig_heights):
            result_dict = {}
            idx = line["id"]
            image_id = line['image_id']
            gt_ans = line["anno"]
            total_items = len(gt_ans)
            if total_items > 15:
                # NOTE: filter out with multi objects
                continue
            assert image_id not in solved_file_path.keys()
            batch_anno = gt_ans
            if len(batch_anno) == 0:
                continue
            tmp_anno = []
            for object_id, anno in enumerate(batch_anno):
                if anno['ignore'] != 1:
                    bbox = anno['bbox']
                else:
                    continue
                if anno['area'] < 2000:
                    # NOTE: filter out small objects
                    continue
                tmp_anno.append(anno)
            if len(tmp_anno) == 0:
                continue

            images = images.unsqueeze(0).cuda()
            has_images = True

            result_dict['id'] = idx
            result_dict['image_id'] = image_id
            result_dict['anno'] = gt_ans
            result_dict['pred'] = []

            batch_prompts = []
            batch_images = []
            batch_has_images = []
            batch_object_id = []

            caption_question = random.choice(CaptionGrounding)
            caption_question = caption_question.replace(" <image>", "")
            conv.messages = []
            conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            conv.append_message(conv.roles[0], caption_question)
            conv.append_message(conv.roles[1], None)
            caption_grounding_cur_prompt = conv.get_prompt()

            tokenized_output = tokenizer(
                [caption_grounding_cur_prompt],
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

            input_ids = torch.as_tensor(tokenized_output.input_ids).cuda()
            attention_mask = torch.as_tensor(tokenized_output.attention_mask).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    has_images=[has_images],
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    temperature=0.7,
                    max_new_tokens=1024,
                    )

            input_token_len = input_ids[0].shape[0]
            n_diff_input_output = (input_ids[0] != output_ids[0][:input_token_len]).sum().item()
            output = tokenizer.batch_decode(output_ids[0][input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
            output = output.strip()
            result_dict['grounding_caption'] = output
            result_dict['orig_width'] = orig_width
            result_dict['orig_height'] = orig_height

            for object_id, anno in enumerate(batch_anno):
                if anno['ignore'] != 1:
                    bbox = anno['bbox']
                else:
                    continue
                if anno['area'] < 2000:
                    # NOTE: filter out small objects
                    continue
                real_object_id = object_id
                scaled_width = 1 / orig_width
                scaled_height = 1 / orig_height
                scaled_bbox = [bbox[0] * scaled_width, bbox[1] * scaled_height, bbox[2] * scaled_width, bbox[3] * scaled_height]
                location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
                question = random.choice(GroundingCaption)
                question = question.replace(" <image>", "")
                question = question.replace("<objs>", BEGIN_LOC + location_tokens + END_LOC)
                conv.messages = []
                conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                grounding_caption_cur_prompt = conv.get_prompt()
                batch_prompts.append(grounding_caption_cur_prompt)
                batch_images.append(images)
                batch_object_id.append(real_object_id)
                batch_has_images.append(has_images)
            if len(batch_prompts) == 0:
                continue

            tokenized_output = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

            input_ids = torch.as_tensor(tokenized_output.input_ids).cuda()
            attention_mask = torch.as_tensor(tokenized_output.attention_mask).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=torch.cat(batch_images, dim=0),
                    has_images=batch_has_images,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    temperature=0.7,
                    max_new_tokens=1024,
                    )
            outputs = []
            pred_result_dict = []
            visual_grounding_prompts = []
            for index, (input_id, output_id) in enumerate(zip(input_ids, output_ids)):
                pred_dict = {}
                input_token_len = input_id.shape[0]
                n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
                output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
                output = output.strip()
                pred_dict['object_id'] = batch_object_id[index]
                pred_dict['caption'] = output
                pred_result_dict.append(pred_dict)
                outputs.append(output)

                # back for grounding
                question = random.choice(VisualGrounding)
                question = question.replace(" <image>", "")
                question = question.replace("<expr>", (BEGIN_DESCRIPTION + '{}' + END_DESCRIPTION).format(output))
                conv.messages = []
                conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                cur_prompt = conv.get_prompt()

                visual_grounding_prompts.append(cur_prompt)

            tokenized_output = tokenizer(
                visual_grounding_prompts,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

            input_ids = torch.as_tensor(tokenized_output.input_ids).cuda()
            attention_mask = torch.as_tensor(tokenized_output.attention_mask).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=torch.cat(batch_images, dim=0),
                    has_images=batch_has_images,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    temperature=0.7,
                    max_new_tokens=1024,
                    )

            for index, (input_id, output_id) in enumerate(zip(input_ids, output_ids)):
                input_token_len = input_id.shape[0]
                n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
                output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
                output = output.strip()
                res = pattern.findall(output)
                pred_bbox = []
                if len(res) == 4:
                    for r in res:
                        pred_bbox.append(float(r))
                    pred_bbox = [pred_bbox[0] * orig_width, pred_bbox[1] * orig_height, pred_bbox[2] * orig_width, pred_bbox[3] * orig_height]
                    pred_result_dict[index]['bbox'] = pred_bbox
                    pred_result_dict[index]['format_error'] = 0
                else:
                    pred_bbox = [0.0, 0.0, 1.0, 1.0]
                    pred_result_dict[index]['bbox'] = pred_bbox
                    pred_result_dict[index]['format_error'] = 1
            result_dict['pred'] += pred_result_dict
            ans_file.write(json.dumps(result_dict) + "\n")
            ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llamav1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)