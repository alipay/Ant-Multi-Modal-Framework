# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageDraw

import copy
import gradio as gr
import os
import re
import secrets
import tempfile
import torch
import json
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import os
import sys
sys.path.append("./")
from pink import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pink.conversation import conv_llama2


DEFAULT_CKPT_PATH = 'pink/Pink-Chat'
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-c", "--llama-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="LLaMA Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    model_name = args.checkpoint_path
    config = AutoConfig.from_pretrained(model_name, use_cache=True)
    config.llama_path = args.llama_path
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')
    return model, tokenizer

def _launch_demo(args, model, tokenizer):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )
    image_processor = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    image_token_len = model.config.num_patches

    model.eval()
    conv = conv_llama2.copy()
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

    def predict(_chatbot, task_history, bbox_state):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        history_cp = task_history
        image = copy.deepcopy(bbox_state['ori_image'])
        image_tensor = image_processor(image)
        images = image_tensor.unsqueeze(0).cuda()
        conv.messages = []
        referring_bbox = bbox_state['bbox']
        bbox_pattern = re.compile(r"<bbox>")
        loc_pattern = re.compile(r"(\[[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9]\])")
        format_bbox_pattern = re.compile(r"[0-9].[0-9][0-9][0-9]")
        num_bbox = 0
        conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):
                conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            else:
                if isinstance(a, (tuple, list)):
                    continue
                loc_token = bbox_pattern.findall(q)
                if len(loc_token) > 0:
                    for _ in range(len(loc_token)):
                        scaled_bbox = [b / 336 for b in referring_bbox[num_bbox]]
                        q = q.replace("<bbox>", "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(*scaled_bbox), 1)
                        num_bbox += 1
                _chatbot[i] = (q, a)
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], a)

        cur_prompt = conv.get_prompt()

        tokenized_output = tokenizer(
            [cur_prompt],
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
                has_images=[True],
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                temperature=0.7,
                max_new_tokens=1024,
            )

        for input_id, output_id in zip(input_ids, output_ids):
            input_token_len = input_id.shape[0]
            n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
            output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
            output = output.strip()
            print(output)

        loc_token = loc_pattern.findall(output)
        copy_image1 = copy.deepcopy(bbox_state['ori_image'])
        width, height = copy_image1.size

        for loc_id, loc in enumerate(loc_token):
            bbox = format_bbox_pattern.findall(loc)
            assert len(bbox) == 4
            scaled_bbox = [float(bbox[0])*width,float(bbox[1])*height,float(bbox[2])*width,float(bbox[3])*height]
            print(scaled_bbox)
            print(width)
            print(height)
            draw1 = ImageDraw.Draw(copy_image1)
            draw1.rectangle(scaled_bbox, fill=None, outline=(0,0,0), width=5)

        _chatbot[-1] = (_chatbot[-1][0], output)
        task_history[-1] = (query, output)

        if len(loc_token) > 0:
            temp_dir = secrets.token_hex(20)
            temp_dir = Path(uploaded_file_dir) / temp_dir
            temp_dir.mkdir(exist_ok=True, parents=True)
            name = f"tmp{secrets.token_hex(5)}.jpg"
            filename = temp_dir / name
            copy_image1.save(str(filename))
            _chatbot.append((None, (str(filename),)))
            task_history.append((None, (str(filename),)))
        
        print(_chatbot)
        print(output)
        yield _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        # history = history + [(_parse_text(text), None)]
        history = history + [(text, None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, bbox_state, file):
        # history = history + [((file.name,), None)]
        # task_history = task_history + [((file.name,), None)]
        copy_image = Image.open(file.name).convert("RGB")
        bbox_state['ori_image'] = copy.deepcopy(copy_image)
        copy_image = copy_image.resize((336, 336), Image.Resampling.BILINEAR)
        bbox_state['raw_image'] = copy_image
        bbox_state['bbox_image'] = copy.deepcopy(copy_image)
        return history, task_history, bbox_state, copy_image

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history, bbox_state):
        task_history.clear()
        bbox_state.clear()
        bbox_state["press"] = False
        bbox_state["bbox"] = []
        bbox_state["raw_image"] = None
        return [], None

    def reset_anno(bbox_state):
        bbox_state['bbox'] = []
        raw_image = bbox_state['raw_image']
        bbox_state['bbox_image'] = copy.deepcopy(raw_image)
        bbox_state['press'] = False
        return bbox_state, raw_image

    def bbox_select(bbox_state, evt: gr.SelectData):
        print(evt.index)
        bbox_image = bbox_state['bbox_image']
        if bbox_state['press']:
            bbox_state['bbox'][-1] += evt.index
            bbox_state['press'] = False
            print(bbox_state)
            draw = ImageDraw.Draw(bbox_image)
            draw.rectangle(bbox_state['bbox'][-1], outline='black', width=5)
            return bbox_state, bbox_image
        else:
            bbox_state['press'] = True
            copy_image = copy.deepcopy(bbox_image)
            draw = ImageDraw.Draw(copy_image)
            radis = 5
            draw.ellipse((evt.index[0] - radis, evt.index[1] - radis, evt.index[0] + radis, evt.index[1] + radis), fill='red')
            bbox_state['bbox'].append([])
            bbox_state['bbox'][-1] += evt.index

            print(bbox_state)
            return bbox_state, copy_image

    with gr.Blocks(fill_height=True) as demo:
        gr.Markdown("""<center><font size=6>Pink-Chat</center>""")
        gr.HTML(
            f"""
        <h2>User Manual</h2>
        <ul>
        <li><p><strong>Step 1.</strong> Upload an image</p>
        </li>
        <li><p><strong>Step 2.</strong> Input the questions. Note: We add some special tokens for the visual grounding or grounding caption tasks.</p>
        <ul>
        <li><strong>Visual Grounding</strong>: Where is &lt;des&gt;left man&lt;/des&gt; in the image?.</li>
        <li><strong>Grounding Caption</strong>: Give me a description of the region &lt;loc&gt;[0.111,0.222,0.333,0.444]&lt;/loc&gt; in image</li>
        </ul>
        </li>
        </ul>
        <p>The following step are needed <strong>only</strong> when input has bounding box.</p>
        <ul>
        <li><p><strong>Step 3.</strong> Add referring object in the question if needed. Use &lt;bbox&gt; placeholder if input has bounding box.</p>
        </li>
        <ul>
        <li><strong>Example</strong>: Who is &lt;bbox&gt?.</li>
        </ul>
        <li><p><strong>Step 4.</strong> Click the image to draw Bounding Box </p>
        </li>
        </ul>
        """)

        image_holder = gr.Image(height=336, width=336)
        chatbot = gr.Chatbot(label='Pink-Chat')
        query = gr.Textbox(lines=1, label='输入')
        with gr.Row():
            submit_btn = gr.Button("🚀 Submit (发送)")
            addfile_btn = gr.UploadButton("📁 Upload (上传图像)", file_types=["image"])
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
        task_history = gr.State([])
        bbox_state = gr.State({"press": False, "bbox": [], "raw_image": None})

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history, bbox_state], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history, bbox_state], [chatbot, image_holder], show_progress=True)
        addfile_btn.click(reset_state, [task_history, bbox_state], [chatbot, image_holder], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, bbox_state, addfile_btn], [chatbot, task_history, bbox_state, image_holder], show_progress=True)
        image_holder.select(bbox_select, [bbox_state], [bbox_state, image_holder])


    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)
    _launch_demo(args, model, tokenizer)


if __name__ == '__main__':
    main()
