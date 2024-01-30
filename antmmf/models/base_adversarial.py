# Copyright (c) 2023 Ant Group and its affiliates.

import sys
from antmmf.common.registry import registry
from antmmf.utils.timer import Timer
import torch.nn as nn


class BaseAdversarial(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.adversarial_parameters = self.config.adversarial_parameters
        self.profiler = Timer()

        # learning rate for the adversarial parameters
        self.lr = self.adversarial_parameters.get("lr", 1.0)

        # the model to be attacked
        self.model = model

        self.device = registry.get("current_device", "cpu")

        # the batch size used in attacking
        self.bsz = self.adversarial_parameters.get("backward_batch_size", 16)
        # the maximum number of internal iterations to obtain adversarial samples
        self.max_iter = self.adversarial_parameters.get("max_iter", 1)

        # the interval to flush generated adversarial samples
        # not used, unless asked for adversarial sample generation
        self.flush_updated_data_interval = self.adversarial_parameters.get(
            "flush_updated_data_interval", 100
        )

        # True if attacking aims at pushing predictions away from the label
        # False if attacking aims at producing particular predictions
        self.away_from_target = self.config.adversarial_optimizer_parameters.params.get(
            "away_from_target", True
        )

        self.lnorm = self.adversarial_parameters.get("lnorm", 0)

    # interface for attacking
    def attack(self, external_optimizer, sample_list):

        raise Exception("not implemented")

    def profile(self, text):
        sys.stdout.write(text + ": " + self.profiler.get_time_since_start() + "\n")
        self.profiler.reset()

    def _extract_loss(self, report):
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        return loss
