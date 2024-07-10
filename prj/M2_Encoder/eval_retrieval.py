import torch
import numpy as np
from collections import defaultdict
import json

from nn4k.consts import NN_EXECUTOR_KEY
from nn4k.invoker import LLMInvoker


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", '"').replace("”", '"')
    return text


def get_data(data_file):
    with open(data_file, "r") as f:
        lines = f.readlines()
    img2txt, txt2img = defaultdict(list), defaultdict(list)
    texts, images = [], []
    text_ids, image_ids = {}, {}
    for i, line in enumerate(lines):
        data = json.loads(line.strip())
        img = data["image"]
        cap = data["caption"]
        images.append(img)
        image_ids[img] = i
        for j in range(len(cap)):
            cap[j] = _preprocess_text(cap[j])
            img2txt[img].append(cap[j])
            txt2img[cap[j]].append(img)
            texts.append(cap[j])
            text_ids[cap[j]] = len(texts) - 1
    img2txt_gt = np.zeros((len(images), len(texts)))
    txt2img_gt = np.zeros((len(texts), len(images)))
    for i, img in enumerate(images):
        txts = img2txt[img]
        for txt in txts:
            img2txt_gt[i][text_ids[txt]] = 1
    for i, txt in enumerate(texts):
        imgs = txt2img[txt]
        for img in imgs:
            txt2img_gt[i][image_ids[img]] = 1
    return texts, images, txt2img_gt, img2txt_gt


def extract_feats(model, texts, images, args):
    txt_feats = []
    for i, txt in enumerate(texts):
        input_text = {"texts": [txt]}
        with torch.no_grad():
            txt_feat = model.local_inference(
                input_text, kwargs={"extract_feat": "text"}
            )
        txt_feats.extend(txt_feat.detach().cpu().numpy().tolist())

    img_feats = []
    for i, img in enumerate(images):
        input_image = {"image_path": img}
        with torch.no_grad():
            img_feat = model.local_inference(
                input_image, kwargs={"extract_feat": "image"}
            )
        img_feats.extend(img_feat.detach().cpu().numpy().tolist())

    txt_feats = np.asarray(txt_feats)
    img_feats = np.asarray(img_feats)
    return txt_feats, img_feats


def calu_recall(txt_feats, img_feats, txt2img_gt, img2txt_gt):
    t2i_mat = txt_feats @ img_feats.T

    t2i_idx = np.argsort(-t2i_mat, axis=1)
    i2t_mat = img_feats @ txt_feats.T
    i2t_idx = np.argsort(-i2t_mat, axis=1)

    t2i_pred = np.zeros((len(t2i_idx), 10))
    i2t_pred = np.zeros((len(i2t_idx), 10))
    for i in range(len(t2i_idx)):
        for j in range(10):
            if txt2img_gt[i][t2i_idx[i][j]] == 1:
                t2i_pred[i][j] = 1
    for i in range(len(i2t_idx)):
        for j in range(10):
            if img2txt_gt[i][i2t_idx[i][j]] == 1:
                i2t_pred[i][j] = 1

    t2i_pred = np.cumsum(t2i_pred, axis=1)
    i2t_pred = np.cumsum(i2t_pred, axis=1)

    t2i_topk = [0] * 10
    for i in range(len(t2i_pred)):
        for j in range(10):
            if t2i_pred[i][j] > 0:
                t2i_topk[j] += 1

    i2t_topk = [0] * 10
    for i in range(len(i2t_pred)):
        for j in range(10):
            if i2t_pred[i][j] > 0:
                i2t_topk[j] += 1

    t2i_topk = np.asarray(t2i_topk) / len(t2i_pred) * 100
    i2t_topk = np.asarray(i2t_topk) / len(i2t_pred) * 100

    print(
        "t2i_topk", round(t2i_topk[0], 1), round(t2i_topk[4], 1), round(t2i_topk[9], 1)
    )
    print(
        "i2t_topk", round(i2t_topk[0], 1), round(i2t_topk[4], 1), round(i2t_topk[9], 1)
    )
    print(
        "MR",
        round(
            (
                t2i_topk[0]
                + t2i_topk[4]
                + t2i_topk[9]
                + i2t_topk[0]
                + i2t_topk[4]
                + i2t_topk[9]
            )
            / 6,
            1,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="命令行输入模型, 数据")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/Encoder_0.4B.json",
        help="模型设置地址",
        required=False,
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/coco-cn_test.jsonl",
        help="数据地址",
        required=False,
    )

    args = parser.parse_args()

    args.data_file = "data/coco-cn_test.jsonl"
    # args.data_file = 'data/coco_caption_karpathy_test.jsonl'
    # args.data_file = 'data/f30k-cn_test.jsonl'
    # args.data_file = 'data/f30k_caption_karpathy_test.jsonl'
    print("data_file: ", args.data_file)

    cfg = {
        "model_config": args.config_path,
        NN_EXECUTOR_KEY: "m2_encoder.M2EncoderExecutor",
    }

    encoder = LLMInvoker.from_config(cfg)
    encoder.warmup_local_model()

    texts, images, txt2img_gt, img2txt_gt = get_data(args.data_file)
    print("texts", len(texts), texts[0], "image", len(images), images[0])

    txt_feats, img_feats = extract_feats(encoder, texts, images, args)
    calu_recall(txt_feats, img_feats, txt2img_gt, img2txt_gt)
