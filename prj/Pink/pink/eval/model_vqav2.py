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
import re
from collections import Counter


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_word(token):
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token


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
DEFAULT_EOS_TOKEN = "</s>"


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

def construct_trie_tree(word_list):
    tree = {}
    for word in word_list:
        current_tree = tree
        for w in word:
            if w not in current_tree.keys():
                current_tree[w] = {}
                current_tree = current_tree[w]
            else:
                current_tree = current_tree[w]
    return tree

class Prefix_allowed_tokens_fn:
    def __init__(self, trie_tree, prefix_length, tokenizer):
        self.trie_tree = trie_tree
        self.prefix_length = prefix_length
        self.tokenizer = tokenizer

    def __call__(self, batch_id, inputs_id):
        current_batch_inputs = inputs_id
        # print(current_batch_inputs)
        if current_batch_inputs.shape[0] == self.prefix_length:
            return list(self.trie_tree.keys())
        if self.tokenizer.convert_tokens_to_ids([DEFAULT_EOS_TOKEN])[0] in current_batch_inputs:
            return [self.tokenizer.pad_token_id]
        else:
            generate_list = current_batch_inputs[self.prefix_length:]
            current_tree = self.trie_tree
            for g in generate_list:
                current_tree = current_tree[int(g)]
            return list(current_tree.keys())

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
    # NOTE: fix the data augmentation parameters ugly!!!
    image_processor = transforms.Compose(
    [
        transforms.Resize((crop_size, crop_size), interpolation=InterpolationMode.BICUBIC),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    image_token_len = model.config.num_patches

    model.eval()
    conv = conv_templates[args.conv_mode].copy()

    questions = [json.loads(line) for line in open(args.question_file, "r")]
    if args.strict_answer:
        merge_dataset = questions + [json.loads(line) for line in open(os.path.join(os.path.dirname(args.question_file), "v2_OpenEnded_mscoco_train2014_questions.jsonl"), "r")]
        answer_list = []
        for t in tqdm(merge_dataset):
            answer_list.append(normalize_word(t['annotation']['multiple_choice_answer']))
        candidate_answer = [(k + DEFAULT_EOS_TOKEN) for k, v in Counter(answer_list).items() if v >= 9]
        candidate_token_index = tokenizer(candidate_answer, add_special_tokens=False).input_ids
        trie_tree = construct_trie_tree(candidate_token_index)
    else:
        candidate_token_index = None

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    batch_size = 8
    total_items = len(questions)
    for batch_id in tqdm(range(total_items // batch_size + 1)):
        batch_questions = questions[batch_id * batch_size: (batch_id + 1) * batch_size]
        if len(batch_questions) == 0:
            continue

        batch_pormpts = []
        batch_images = []
        batch_has_images = []
        result_dicts = []

        for i, line in enumerate(batch_questions):
            result_dict = {}
            idx = line["question_id"]
            image_id = line['image_path']
            question = line['question']
            gt_ans = [a['answer'] for a in line["annotation"]['answers']]
            try:
                with open(os.path.join(args.image_folder, image_id), "rb") as f:
                    image = Image.open(io.BytesIO(f.read())).convert('RGB')
            except:
                print("file {} does not exist in pcache".format(image_id))
            image_tensor = image_processor(image)
            images = image_tensor.unsqueeze(0).cuda()
            select_question = random.choice(QuestionAnswer)
            select_question = select_question.replace(" <image>", "")
            select_question = select_question.replace("<question>", "{}{}{}".format(BEGIN_QUESTION, question, END_QUESTION))
            has_images = True

            conv.messages = []
            conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            conv.append_message(conv.roles[0], select_question)
            conv.append_message(conv.roles[1], None)
            cur_prompt = conv.get_prompt()
            result_dict['initial_prompt'] = question
            result_dict['question_id'] = idx
            result_dict['gt_answer'] = gt_ans
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

        if args.strict_answer:
            prefix_allowed_tokens_fn = Prefix_allowed_tokens_fn(trie_tree, len(input_ids[0]), tokenizer)
        else:
            prefix_allowed_tokens_fn = None

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
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
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
    parser.add_argument("--strict_answer", action='store_true')
    args = parser.parse_args()

    eval_model(args)