# Copyright (c) 2023 Ant Group and its affiliates.
import copy
from itertools import chain

import torch
from torch import nn
from tqdm import tqdm

from antmmf.common.meter import Meter
from antmmf.common.registry import registry
from antmmf.common.report import Report
from antmmf.structures import SampleList
from antmmf.trainers import BaseTrainer
from antmmf.utils.distributed_utils import (
    synchronize,
    is_main_process,
    get_world_size,
    all_gather,
    get_rank,
)


def split_batch(input_batch):
    visual_batch = SampleList()
    text_batch = SampleList()
    for b_key in input_batch.keys():
        # for visual info
        if "image" in b_key:
            visual_batch[b_key] = input_batch[b_key]
        # for text info
        if "caption" in b_key:
            text_batch[b_key] = input_batch[b_key]

    return visual_batch, text_batch


def nested_cpu(model_output):
    if isinstance(model_output, torch.Tensor):
        return model_output.cpu()
    elif isinstance(model_output, dict):
        for k, v in model_output.items():
            model_output[k] = nested_cpu(v)
    elif isinstance(model_output, (list, tuple)):
        for i, v in enumerate(model_output):
            model_output[i] = nested_cpu(v)
    return model_output


@registry.register_trainer("retrieval_trainer")
class RetrievalTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def evaluate_set(self, loader_list):
        dataset_name, meter = self._evaluate_set(loader_list)
        synchronize()
        return dataset_name, meter

    def _forward_single(self, batch, infer_model, device):
        prepared_batch = self.task_loader.prepare_batch(batch)
        if prepared_batch is None:
            return None, None, None

        prepared_batch["text_stage1_output"] = tuple(
            [
                x.to(device) if isinstance(x, torch.Tensor) else x
                for x in prepared_batch["text_stage1_output"]
            ]
        )
        prepared_batch["visual_stage1_output"] = tuple(
            [
                x.to(device) if isinstance(x, torch.Tensor) else x
                for x in prepared_batch["visual_stage1_output"]
            ]
        )
        model_output = nested_cpu(infer_model(prepared_batch))
        report = Report()
        report.losses = model_output.pop("losses")
        report.metrics = model_output.pop("metrics")
        report.dataset_type = prepared_batch["dataset_type"]
        report.dataset_name = prepared_batch["dataset_name"]
        # logger is not pickle-able
        report.writer = None
        return report, model_output, None

    def _evaluate_set(self, loader_list):
        other_info_list = []
        meter = Meter()

        self.model.eval()
        is_parallel = isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, nn.parallel.DistributedDataParallel
        )
        with torch.no_grad():
            # step1: 存储所有任务的输入pair， 1k val/test v-t pairs, retrieval需要构造 1k x 1k 个pair组合
            parameters_tuple_list = []
            exist_vids = set()
            all_text_batchs, all_visual_batchs = [], []
            for batch in tqdm(
                chain(*loader_list),
                total=self._len_of_loader_list(loader_list),
                disable=not is_main_process(),
            ):
                batch = self.task_loader.prepare_batch(batch)
                visual_batch, text_batch = split_batch(batch)
                other_info_list.append((batch.dataset_type, batch.dataset_name))
                all_text_batchs.append(
                    (
                        text_batch["caption_raw_input_ids"],
                        text_batch["caption_input_mask"],
                        text_batch["caption_tid"],
                        text_batch["caption_vid_list"],
                        text_batch.get("caption_twm_input_mask"),
                    )
                )
                batch_data = visual_batch["image_data"]
                batch_mask = visual_batch["image_pad_mask"]
                batch_n_clips = visual_batch["image_n_clips"]
                batch_num_frames = visual_batch["image_num_frames"]
                batch_vids = visual_batch["image_vid"]
                batch_tid_list = visual_batch["image_tid_list"]
                # 对visual batch进行去重
                mask = torch.ones_like(batch_vids, dtype=torch.bool)
                for i, v in enumerate(batch_vids):
                    if v.item() not in exist_vids:
                        exist_vids.add(v.item())
                    else:
                        mask[i] = False
                # mask select
                if mask.long().sum() == 0:
                    continue
                batch_n_clips = [v for m, v in zip(mask.tolist(), batch_n_clips) if m]
                batch_num_frames = [
                    v for m, v in zip(mask.tolist(), batch_num_frames) if m
                ]
                batch_tid_list = [v for m, v in zip(mask.tolist(), batch_tid_list) if m]

                all_visual_batchs.append(
                    (
                        batch_data[mask],
                        batch_mask[mask],
                        batch_n_clips,
                        batch_num_frames,
                        batch_vids[mask],
                        batch_tid_list,
                    )
                )

            self.writer.write("Extract visual/text features", "info")

            # visual/text encoder
            v_encoder = (
                self.model.module.model.module.forward_img_encoder
                if is_parallel
                else self.model.model.module.forward_img_encoder
            )
            t_encoder = (
                self.model.module.model.module.forward_text_encoder
                if is_parallel
                else self.model.model.module.forward_text_encoder
            )
            prepare_cross_text = (
                self.model.module.model.module.prepare_cross_text
                if is_parallel
                else self.model.model.module.prepare_cross_text
            )
            prepare_cross_visual = (
                self.model.module.model.module.prepare_cross_visual
                if is_parallel
                else self.model.model.module.prepare_cross_visual
            )

            # 提取文本特征
            text_features = []
            for (
                txt_input_ids,
                input_mask,
                caption_tid,
                caption_vid_list,
                twm_input_mask,
            ) in all_text_batchs:
                text_embed_l1 = t_encoder(txt_input_ids, input_mask)["pooled_output"]
                cap_embed, cap_mask, batch_size = prepare_cross_text(
                    txt_input_ids, input_mask
                )
                l2_txt_input = (cap_embed, cap_mask, text_embed_l1, batch_size)
                text_features.append((l2_txt_input, caption_tid, caption_vid_list))
            text_features = sorted(text_features, key=lambda x: x[1].min().item())

            batch_text_list = [x[0] for x in text_features]
            text2video = [x[-1] for x in text_features]

            # 提取视觉特征
            visual_features = []
            for (
                vis_data,
                vis_mask,
                vis_n_clips,
                vis_num_frames,
                vis_ids,
                v2t_id_list,
            ) in all_visual_batchs:
                visual_embed_dict = v_encoder(
                    vis_data, vis_mask, vis_n_clips, vis_num_frames
                )
                visual_embed, visual_mask, num_clip = prepare_cross_visual(
                    visual_embed_dict["visual_embed"], visual_embed_dict["visual_mask"]
                )
                video_embed_l1 = visual_embed_dict["clip_feature"]
                l2_vis_input = (visual_embed, visual_mask, video_embed_l1, num_clip)
                visual_features.append((l2_vis_input, vis_ids, v2t_id_list))
            visual_features = sorted(visual_features, key=lambda x: x[1].min().item())
            batch_visual_list = [x[0] for x in visual_features]
            video2text = [x[-1] for x in visual_features]

            for idx_t, batch_t in enumerate(batch_text_list):
                current_t_batch = SampleList()
                current_t_batch["dataset_type"] = other_info_list[idx_t][0]
                current_t_batch["dataset_name"] = other_info_list[idx_t][1]
                for idx_v, batch_v in enumerate(batch_visual_list):
                    current_batch = copy.deepcopy(current_t_batch)
                    current_batch["text_stage1_output"] = batch_t
                    current_batch["visual_stage1_output"] = batch_v
                    parameters_tuple_list.append([idx_t, idx_v, current_batch])

            # step2: 把样本分配在不同gpu_device上
            n_gpu = get_world_size()
            split_len = (len(parameters_tuple_list) + n_gpu - 1) // n_gpu
            dev_id = get_rank()
            s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
            batch_splits = parameters_tuple_list[s_:e_]

            self.writer.write("Evaluate on val/test retrieval items", "info")

            # 并行apply
            def _run_on_single_gpu(model, list_splits):
                results = []
                if torch.cuda.is_available():
                    device = torch.device(torch.cuda.current_device())
                else:
                    device = torch.device("cpu")
                enable = device.type == "cpu" or device.index == 0
                for idx_t, idx_v, current_batch in tqdm(
                    list_splits, total=len(list_splits), disable=not enable
                ):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # 模型支持:
                    # 1. forward中检测到txt/visual 特征，则跳过text/visual特征提取
                    # 2. 模型负责计算cross-similarity
                    with torch.no_grad():
                        report, model_output, prepare_batch = self._forward_single(
                            current_batch, model, device
                        )
                    results.append((report, model_output, prepare_batch, idx_t, idx_v))

                return results

            model = self.model.module if is_parallel else self.model

            # step3: 每个gpu对分到的任务进行inference
            result = _run_on_single_gpu(model, batch_splits)
            del parameters_tuple_list, batch_splits, model

            # step4: 合并所有gpu的推理结果
            parallel_outputs = all_gather(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            num_batch_t, num_batch_v = len(batch_text_list), len(batch_visual_list)
            outputs = [[] for _ in range(num_batch_t)]
            for gpu_idx in range(len(parallel_outputs)):
                for r, o, p, idx_t, v in parallel_outputs[gpu_idx]:
                    outputs[idx_t].append([r, o, p, idx_t, v])

            self.overall_metric_evaluator.reset()
            for idx_t in range(num_batch_t):
                for output in sorted(outputs[idx_t], key=lambda x: x[-1]):
                    report, model_output, prepare_batch, idx_t, idx_v = output
                    self._update_meter(report, meter, sync=False)
                    self.overall_metric_evaluator.collect(
                        report,
                        model_output,
                        idx_t,
                        idx_v,
                        t2v=text2video[idx_t],
                        v2t=video2text[idx_v],
                    )

            dataset_name = report.dataset_name

        self.model.train()
        return dataset_name, meter
