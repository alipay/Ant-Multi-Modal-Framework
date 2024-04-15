import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from pink.model import adapter, mark_only_adapter_as_trainable
from pink import *
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria, CLIPModel
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from torch.utils.data import DataLoader

import io
from PIL import Image
import random
import math
from pink.datasets import VisualGroundingDataset
from dataclasses import dataclass
from pink.datasets.Templates import VisualGrounding, GroundingCaption
from pink.conversation import conv_templates


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

@dataclass
class DataCollatorForSupervisedDataset(object):
    image_token_len = None
    def __call__(self, instances):
        """Collate examples for supervised fine-tuning."""
        conv = conv_templates[args.conv_mode].copy()
        batch_prompts = []
        batch_images = []
        batch_has_images = []
        result_dicts = []
        batch_grounding_caption_pormpts = []
        for i, line in enumerate(instances):
            result_dict = {}
            idx = line["exp_id"]
            ref_id = line['ref_id']
            referring = line['sentence']
            gt_ans = line["bbox"]
            orig_width, orig_height = line['orig_wh']
            current_width, current_height = line['cur_wh']
            images = line['images'].unsqueeze(0)
            question = random.choice(VisualGrounding)
            question = question.replace(" <image>", "")
            question = question.replace("<expr>", (BEGIN_DESCRIPTION + '{}' + END_DESCRIPTION).format(referring))
            conv.messages = []
            conv.set_system(PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            cur_prompt = conv.get_prompt()

            scaled_width = 1 / orig_width
            scaled_height = 1 / orig_height
            scaled_bbox = [gt_ans[0] * scaled_width, gt_ans[1] * scaled_height, gt_ans[2] * scaled_width, gt_ans[3] * scaled_height]
            location_tokens = "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox)
            question = random.choice(GroundingCaption)
            question = question.replace(" <image>", "")
            question = question.replace("<objs>", BEGIN_LOC + location_tokens + END_LOC)

            conv.messages = []
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            conv.set_system(PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            grounding_caption_cur_prompt = conv.get_prompt()

            batch_grounding_caption_pormpts.append(grounding_caption_cur_prompt)

            has_images = True

            result_dict['initial_prompt'] = cur_prompt
            result_dict['exp_id'] = idx
            result_dict['ref_id'] = ref_id
            result_dict['bbox'] = gt_ans
            result_dict['orig_wh'] = [orig_width, orig_height]
            result_dict['cur_wh'] = [current_width, current_height]
            batch_prompts.append(cur_prompt)
            batch_images.append(images)
            batch_has_images.append(has_images)
            result_dicts.append(result_dict)
        return result_dicts, batch_prompts, batch_images, batch_has_images, batch_grounding_caption_pormpts


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
        if "adapter_" not in name or "lora_" not in name:
            param.data = param.data.half()
    model.lm_head.to(torch.float16)
    model = model.cuda()
    print(model)
    crop_size = model.config.crop_size

    image_token_len = model.config.num_patches

    dataset = VisualGroundingDataset(tokenizer=tokenizer,
                        data_path=os.path.join(args.question_file),
                        multimodal_cfg=dict(
                            is_multimodal=True,
                            image_token_len=image_token_len,
                            image_folder=args.image_folder,
                            base_path="",
                            cur_token_len=image_token_len,
                            image_size=model.config.crop_size,
                            crop_size=model.config.crop_size,
                            grounding_caption=False,
                            conversation_template=args.conv_mode,),
                            is_train=False)

    model.eval()

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    batch_size = 8
    DataCollatorForSupervisedDataset.image_token_len = image_token_len
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=DataCollatorForSupervisedDataset())

    for result_dicts, batch_prompts, batch_images, batch_has_images, batch_grounding_caption_pormpts in tqdm(data_loader):
        tokenized_output = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        batch_images = torch.cat(batch_images, dim=0).cuda()

        input_ids = torch.as_tensor(tokenized_output.input_ids).cuda()
        attention_mask = torch.as_tensor(tokenized_output.attention_mask).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=batch_images,
                has_images=batch_has_images,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                temperature=0.7,
                max_new_tokens=1024,
                )
        outputs = []
        for input_id, output_id in zip(input_ids, output_ids):
            input_token_len = input_id.shape[0]
            n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
            output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
            output = output.strip()
            outputs.append(output)

        if args.grounding_caption:
            gc_tokenized_output = tokenizer(
                batch_grounding_caption_pormpts,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

            gc_input_ids = torch.as_tensor(gc_tokenized_output.input_ids).cuda()
            gc_attention_mask = torch.as_tensor(gc_tokenized_output.attention_mask).cuda()

            with torch.inference_mode():
                gc_output_ids = model.generate(
                    gc_input_ids,
                    images=batch_images,
                    has_images=batch_has_images,
                    attention_mask=gc_attention_mask,
                    do_sample=False,
                    num_beams=1,
                    temperature=0.7,
                    max_new_tokens=1024,
                    )
            gc_outputs = []
            for input_id, output_id in zip(gc_input_ids, gc_output_ids):
                input_token_len = input_id.shape[0]
                n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
                output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
                output = output.strip()
                gc_outputs.append(output)
            for i in range(len(gc_outputs)):
                result_dicts[i].update({
                    "groudning_caption": gc_outputs[i],
                })

        for i in range(len(outputs)):
            result_dicts[i].update({
                                "pred_bbox": outputs[i],
                                })
            ans_file.write(json.dumps(result_dicts[i]) + "\n")
        ans_file.flush()
    ans_file.close()
    os.system("python pink/eval/eval_vg.py --result-file {} --crop-size {}".format(answers_file, crop_size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llamav1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--grounding_caption", action="store_true")
    args = parser.parse_args()

    eval_model(args)
