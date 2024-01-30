# Copyright (c) 2023 Ant Group and its affiliates.

from collections import defaultdict

import torch
from antmmf.common.registry import registry
from torch.optim.optimizer import Optimizer, required


@registry.register_optimizer("freelb")
class FreeLB(Optimizer):
    # adversarial parameter optimizer
    # paper reference:
    # https://arxiv.org/pdf/1909.11764.pdf
    # sign : away from target if sign == True, else False
    def __init__(
        self,
        params,
        lr=required,
        alpha=required,
        epsilon=required,
        away_from_target=required,
    ):
        defaults = dict(lr=lr, lr_old=lr)
        super(FreeLB, self).__init__(params, defaults)

        self.sign = 1 if away_from_target is True else -1

        self.alpha = alpha
        self.epsilon = epsilon
        self.lr = lr

    @property
    def supports_memory_efficient_fp16(self):
        return True

    def reset(self, info):
        self.state = defaultdict(dict)
        self.sample_info = info

    def step(self, closure=None, sign=1):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Return:
            loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            alpha = group.get("alpha", self.alpha)
            epsilon = group.get("epsilon", self.epsilon)

            modality = group["modality"]
            # this is the norm of the original/clean sample
            orig_norm = self.sample_info.get("orig_norm", {}).get(modality, None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    # state initialization
                    state["acc_delta"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                acc_delta = state["acc_delta"]

                data = p.data
                grad = p.grad

                # use Frobenius norm by default (2-norm)
                d_p_norm = grad.norm()
                grad.div_(d_p_norm)
                acc_delta.add_(grad, alpha=alpha)
                # constrain to within a norm ball
                acc_delta_norm = acc_delta.norm()

                # data norm
                # this can be zero, if data is for deltas
                p_norm = data.norm() if orig_norm is None else orig_norm
                if acc_delta_norm > epsilon * p_norm:
                    acc_delta.mul_(epsilon * p_norm).div_(acc_delta_norm)
                # Eq. (11) in https://arxiv.org/pdf/1909.11764.pdf

                # increase loss
                p_data_fp32 = p.data
                p_data_fp32.add_(acc_delta, alpha=self.sign * lr)

                p.data.copy_(p_data_fp32)

        return loss
