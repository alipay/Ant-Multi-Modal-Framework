# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import copy

import torch
import torch.nn.functional as F
from torch import nn

from antmmf.utils.distributed_utils import gather_tensor


class MocoUtils(nn.Module):
    def __init__(self, config, img_encoder=None, txt_encoder=None):
        assert img_encoder is not None or txt_encoder is not None
        super().__init__()
        self.config = config
        self.dim = self.config.hidden_size
        self.txt_K = getattr(
            self.config, "K", 16384
        )  # queue size; number of negative keys
        self.img_K = 16384
        self.m = getattr(
            self.config, "M", 0.9999
        )  # moco momentum of updating key encoder (default: 0.9999)
        self.T = getattr(self.config, "T", 0.05)  # softmax temperature (default: 0.05)
        self.img_encoder_q = None
        if img_encoder is not None:
            # copy img encoder
            self.img_encoder_q = img_encoder
            self.img_encoder_k = copy.deepcopy(self.img_encoder_q)
            for param_q, param_k in zip(
                self.img_encoder_q.parameters(), self.img_encoder_k.parameters()
            ):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            # create the queue
            self.register_buffer("img_queue", torch.randn(self.dim, self.img_K))
            self.img_queue = F.normalize(self.img_queue, dim=0)
            self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.txt_encoder_q = None
        if txt_encoder is not None:
            # copy text encoder
            self.txt_encoder_q = txt_encoder
            self.txt_encoder_k = copy.deepcopy(self.txt_encoder_q)
            for param_q, param_k in zip(
                self.txt_encoder_q.parameters(), self.txt_encoder_k.parameters()
            ):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            self.register_buffer("txt_queue", torch.randn(self.dim, self.txt_K))
            self.txt_queue = F.normalize(self.txt_queue, dim=0)
            self.register_buffer("txt_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        if self.img_encoder_q is not None:
            for param_q, param_k in zip(
                self.img_encoder_q.parameters(), self.img_encoder_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        if self.txt_encoder_q is not None:
            for param_q, param_k in zip(
                self.txt_encoder_q.parameters(), self.txt_encoder_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def moco_loss(self, pos, neg, mining_top_K=None):
        # pos: bsz, N_pos
        # neg: bsz, K
        if mining_top_K is not None:
            neg = neg.topk(mining_top_K, dim=1, largest=True).values

        pos_neg = torch.cat([pos, neg], 1)
        # apply temperature
        v_nominator = torch.logsumexp(pos / self.T, dim=1)
        v_denominator = torch.logsumexp(pos_neg / self.T, dim=1)
        return torch.mean(v_denominator - v_nominator)

    @torch.no_grad()
    def dequeue_and_enqueue(self, vis_keys, txt_keys):
        def _dequeue_and_enqueue(keys, queue_ptr, queue, K):
            # gather keys before updating queue
            _keys = gather_tensor(
                keys, method="cat", back_gradient=False, pad_tensors=True
            )
            if torch.isnan(_keys).any().item():
                return
            ptr = int(queue_ptr)
            # replace the keys at ptr (dequeue and enqueue)
            end_ptr = min(ptr + _keys.shape[0], K)
            start_ptr = end_ptr - _keys.shape[0]
            queue[:, start_ptr:end_ptr] = _keys.T
            ptr = end_ptr % K  # move pointer
            queue_ptr[0] = ptr

        if self.img_encoder_q is not None:
            # keys: bsz*num_clips, hidden
            _dequeue_and_enqueue(
                vis_keys, self.img_queue_ptr, self.img_queue, self.img_K
            )
        if self.txt_encoder_q is not None:
            _dequeue_and_enqueue(
                txt_keys, self.txt_queue_ptr, self.txt_queue, self.txt_K
            )
