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

import io
from PIL import Image
import random
import math
from pink.datasets.Templates import QuestionAnswer
from pink.conversation import conv_llava_v1, conv_simple_v1, conv_llama2


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
    questions = json.load(open(os.path.join(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # If you find the output is None, set batch_size = 1 
    batch_size = 8

    total_items = len(questions)
    for batch_id in tqdm(range(total_items // batch_size + 1)):
        batch_questions = questions[batch_id * batch_size: (batch_id + 1) * batch_size]
        if len(batch_questions) == 0:
            continue

        batch_pormpts_point = []
        batch_pormpts_bbox = []
        batch_images = []
        batch_has_images = []
        result_dicts = []

        for i, line in enumerate(batch_questions):
            result_dict = {}
            image_id = line['id']
    
            try:
                with open(os.path.join(args.image_folder, "{}.jpg".format(image_id)), "rb") as f:
                    image = Image.open(io.BytesIO(f.read())).convert('RGB')
            except:
                print("file {} does not exist in pcache".format(image_id))
            orig_width, orig_height = image.size
            scaled_width = 1 / orig_width
            scaled_height = 1 / orig_height
            scaled_point = [line['points']['x'] * scaled_width, line['points']['y'] * scaled_height]
            scaled_bbox = [line['all_objs']['x'] * scaled_width, line['all_objs']['y'] * scaled_height, (line['all_objs']['x'] + line['all_objs']['w']) * scaled_width, (line['all_objs']['y'] + line['all_objs']['h']) * scaled_height]
            location_token_point = "[{:.3f},{:.3f}]".format(*scaled_point)
            location_tokens_bbox = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)

            image_tensor = image_processor(image)
            images = image_tensor.unsqueeze(0).cuda()
            question = "{} {}{}{}".format(line["question"], BEGIN_LOC, location_token_point, END_LOC)
            bbox_question = "{} {}{}{}".format(line["question"], BEGIN_LOC, location_tokens_bbox, END_LOC)
        
            conv.messages = []
            conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            cur_prompt_point = conv.get_prompt()

            conv.messages = []
            conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            conv.append_message(conv.roles[0], bbox_question)
            conv.append_message(conv.roles[1], None)
            cur_prompt_bbox = conv.get_prompt()

            gt_ans = line['points']['ans']

            has_images = True

            result_dict['id'] = image_id
            result_dict['gt_ans'] = gt_ans
            batch_pormpts_point.append(cur_prompt_point)
            batch_pormpts_bbox.append(cur_prompt_bbox)
            batch_images.append(images)
            batch_has_images.append(has_images)
            result_dicts.append(result_dict)

        tokenized_output_point = tokenizer(
            batch_pormpts_point,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = torch.as_tensor(tokenized_output_point.input_ids).cuda()
        attention_mask = torch.as_tensor(tokenized_output_point.attention_mask).cuda()

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
        outputs_point = []
        for input_id, output_id in zip(input_ids, output_ids):
            input_token_len = input_id.shape[0]
            n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
            output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
            output = output.strip()
            outputs_point.append(output)

        tokenized_output_bbox = tokenizer(
            batch_pormpts_bbox,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = torch.as_tensor(tokenized_output_bbox.input_ids).cuda()
        attention_mask = torch.as_tensor(tokenized_output_bbox.attention_mask).cuda()

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
        outputs_bbox = []
        for input_id, output_id in zip(input_ids, output_ids):
            input_token_len = input_id.shape[0]
            n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
            output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
            output = output.strip()
            outputs_bbox.append(output)

        for i in range(len(outputs_bbox)):
            result_dicts[i].update({
                                "answer_bbox": outputs_bbox[i],
                                "answer_point": outputs_point[i],
                                })
            ans_file.write(json.dumps(result_dicts[i]) + "\n")
        ans_file.flush()
    ans_file.close()
    os.system("python pink/eval/eval_pointingvqa_local.py --result-file {}".format(args.answers_file))

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