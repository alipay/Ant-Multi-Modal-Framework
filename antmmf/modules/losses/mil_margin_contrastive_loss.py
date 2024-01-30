# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
import torch.nn.functional as F
from antmmf.common.registry import registry
from .mil_nce_loss import MILNCELoss


@registry.register_loss("mil-margin-contrastive")
class MILMarginContrastiveLoss(MILNCELoss):
    r"""
    Loss is defined as
    loss = max(0, (logsumpex(video_i * text_k^T) + margin - logsumpexp(video_i * text_i^T)))

    The above definition fits the definition of contrastive learning loss, which aims at maximizing
    the similarity between the positive pairs and minimizeing the similarity between the negative pairs

    Differences from MIL-NCE
    * 分母没有positive pair的similarities
    * 应该比较适合batch-size小的情况
    * 增加了一个margin项与hinge loss；margin的缺省为1；hinge的帮助不大（在MSR-VTT数据上），但是采用这样的方式，有灵活性
    * 实验表明，比MIL-NCE效果（median rank作为metric）更好
    """

    def __init__(self, modalities, margin, **kwargs):
        super(MILMarginContrastiveLoss, self).__init__(modalities)
        self.margin = margin
        self.weight = kwargs.get("weight", 1.0)

    def _forward(self, video_embd, text_embd):
        # e.g. v = randn((b,d)), t = randn((b,d))
        x = torch.matmul(video_embd, text_embd.t())
        # x.shape = [b, b]
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        # x.shape = [b, b, 1]
        y = torch.eye(x.shape[0])[:, :, None].to(device=x.device)
        # y is diagonal matrix but with shape of [b, b, 1]
        nominator = x * y
        # obtain the diagonal element of x, e.g., norinator = [[[3.38], [0.0]], [[0.0], [2.01]]]
        # where x = [[[3.38], [3.12]], [[2.15], [2.02]]]
        nominator = nominator.sum(dim=1)
        # get the diagonal element in a vector
        nominator = torch.logsumexp(nominator, dim=1)
        # get the diagonal element, e.g., tensor([3.3888, 2.0121])

        # TODO: permutation here increases number of negative samples,
        # but it is not described in the paper
        denominator = x.permute(1, 0, 2).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(
            F.relu(
                denominator
                - nominator
                + torch.tensor(self.margin).float().to(device=x.device)
            )
        )

    def forward(self, sample_list, model_output, *args, **kwargs):
        for m in self._modalities:
            if m not in model_output:
                return None

        video = model_output[self._modalities[0]]
        text = model_output[self._modalities[1]]

        num_clip = video.shape[0] // text.shape[0]
        if num_clip > 1:
            # replicate text output to cover multiple time clips in video
            text = text.view(text.shape[0], 1, -1)
            text = text.repeat_interleave(num_clip, dim=1)
            text = text.view(video.shape[0], -1)

        return torch.tensor(self.weight).float().to(
            device=video.device
        ) * self._forward(video, text)
