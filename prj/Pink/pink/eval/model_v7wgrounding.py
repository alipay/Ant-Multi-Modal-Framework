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
from pink.conversation import conv_simple_v1, conv_llama2
from pink.datasets.Templates import ChoiceQuestionAnswer
import re


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
BEGIN_OPTIONS = "<opt>"
END_OPTIONS = "</opt>"


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
    pattern = re.compile(r'The answer is ([A-Z]).')

    questions = json.load(open(os.path.join(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    index_to_options = {0: "A", 1: "B", 2: "C", 3: "D"}

    batch_size = 8
    total_items = len(questions)
    num_correct = 0
    num_sample = 0
    for batch_id in tqdm(range(total_items // batch_size + 1)):
        batch_questions = questions[batch_id * batch_size: (batch_id + 1) * batch_size]
        if len(batch_questions) == 0:
            continue

        batch_pormpts = []
        batch_images = []
        batch_has_images = []
        result_dicts = []

        for i, line in enumerate(batch_questions):
            num_sample += 1
            result_dict = {}
            image_id = line['file_name']
    
            try:
                with open(os.path.join(args.image_folder, image_id), "rb") as f:
                    image = Image.open(io.BytesIO(f.read())).convert('RGB')
            except:
                print("file {} does not exist in pcache".format(image_id))
            orig_width, orig_height = image.size
            image_tensor = image_processor(image)
            images = image_tensor.unsqueeze(0).cuda()

            scaled_width = 1 / orig_width
            scaled_height = 1 / orig_height
            answer_bbox = [line['answer_box']['x'] * scaled_width, line['answer_box']['y'] * scaled_height, (line['answer_box']['x'] + line['answer_box']['width']) * scaled_width, (line['answer_box']['y'] + line['answer_box']['height']) * scaled_height]

            multi_choice_bboxes = []
            for bbox in line['multiple_choices_box']:
                multi_choice_bboxes.append([bbox['x'] * scaled_width, bbox['y'] * scaled_height, (bbox['x'] + bbox['width']) * scaled_width, (bbox['y'] + bbox['height']) * scaled_height])
            multi_choice_bboxes = [answer_bbox] + multi_choice_bboxes

            random.shuffle(multi_choice_bboxes)
            multi_choice_tokens = []
            for bbox in multi_choice_bboxes:
                multi_choice_tokens.append("[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*bbox))

            gt_ans = multi_choice_tokens.index("{}".format("[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*answer_bbox)))

            select_question = random.choice(ChoiceQuestionAnswer)
            select_question = select_question.replace(" <image>", "")
            select_question = select_question.replace("<question>", "{}{}{}".format(BEGIN_QUESTION, line["question"], END_QUESTION))
            options = ""
            for index, opt in enumerate(multi_choice_tokens):
                options += index_to_options[index] + ". {}{}{}\n".format(BEGIN_LOC, opt, END_LOC)
            options = options.rstrip("\n")
            select_question = select_question.replace("<option>", "{}{}{}".format(BEGIN_OPTIONS, options, END_OPTIONS))

            conv.messages = []
            conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            conv.append_message(conv.roles[0], select_question)
            conv.append_message(conv.roles[1], None)
            cur_prompt = conv.get_prompt()

            has_images = True

            result_dict['id'] = image_id
            result_dict['gt_ans'] = gt_ans
            result_dict['cur_prompt'] = cur_prompt
            batch_pormpts.append(cur_prompt)
            batch_images.append(images)
            batch_has_images.append(has_images)
            result_dicts.append(result_dict)

        tokenized_output = tokenizer(
            batch_pormpts,
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
        for input_id, output_id in zip(input_ids, output_ids):
            input_token_len = input_id.shape[0]
            n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
            output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
            output = output.strip()
            outputs.append(output)

        for i in range(len(outputs)):
            result_dicts[i].update({
                                "answer": outputs[i],
                                })
            ans_file.write(json.dumps(result_dicts[i]) + "\n")
            res = pattern.findall(outputs[i])
            if len(res) == 1:
                answer = res[0]  # 'A', 'B', ...
                if answer in ["A", "B", "C", "D", "E"]:
                    pred_idx = ["A", "B", "C", "D", "E"].index(answer)
                    if pred_idx == result_dicts[i]['gt_ans']:
                        num_correct += 1
        ans_file.flush()
    ans_file.close()
    print("num correct: {} num samples: {} accuracy: {}".format(num_correct, num_sample, num_correct / num_sample))

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