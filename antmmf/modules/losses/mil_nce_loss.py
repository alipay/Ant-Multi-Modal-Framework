# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch
from torch import nn
from antmmf.common.constants import TEXT_MODALITY
from antmmf.common.registry import registry


@registry.register_loss("mil-nce")
class MILNCELoss(nn.Module):
    r"""
    Multi-Instance-Learning Noise Contrastive Learning
    Compared to the above NCE loss, this doesn't work for huge number of classes such as millions of class numbers

    We wish to learn a video representation based on the previously described probabilistic model p(P).
    However, this is challenging as one cannot directly apply standard generative techniques such as maximum likelihood
    due to the intractability of computing the normalization constant over all possible pairs of videos
    and narrations. Instead, we rely on a discriminative technique, namely the noise-contrastive estimation (NCE)
    approach, that has recently been shown to be effective in the context of feature learning.
    The core idea is to directly optimize the unnormalized probabilistic model to discriminate between data
    obtained from the true joint distribution P(X × Y) and some artificially generated noise data,
    also called “negatives”.
    Here we use the softmax version of NCE
        Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu.
        Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016. 5
    and replacing the probability of a single positive match with the MIL like extension.
    Given this, we can simply estimate the parameters of our model by maximizing the ratio of likelihood
    from positive samples versus negative samples, where negative samples are other samples in the same minibatch.

    Reference:
    End-to-End Learning of Visual Representations from Uncurated Instructional Videos, CVPR 2020
    """

    def __init__(self, modalities):
        super(MILNCELoss, self).__init__()
        self._modalities = modalities
        assert len(self._modalities) == 2, "current support two modalities only"
        assert (
            self._modalities[1] == TEXT_MODALITY
        ), "text has to be the second in the modalities"

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
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)

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

        return self._forward(video, text)
