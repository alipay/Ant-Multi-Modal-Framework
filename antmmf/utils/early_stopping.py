# Copyright (c) 2023 Ant Group and its affiliates.
import numpy as np
import torch

from antmmf.utils.distributed_utils import is_main_process


class EarlyStopping:
    """
    Provides early stopping functionality. Keeps track of model metrics,
    and if it doesn't improve over time restores last best performing
    parameters.
    """

    def __init__(
        self,
        model,
        checkpoint_instance,
        monitored_metric="total_loss",
        patience=1000,
        minimize=False,
        should_stop=True,
    ):
        self.minimize = minimize
        self.patience = patience
        self.model = model
        self.checkpoint = checkpoint_instance
        self.monitored_metric = monitored_metric

        if "val/" not in self.monitored_metric:
            self.monitored_metric = "val/{}".format(self.monitored_metric)

        self.best_monitored_value = -np.inf if not minimize else np.inf
        self.best_monitored_iteration = 0
        self.should_stop = should_stop
        self.activated = False
        self.metric = self.monitored_metric

    def __call__(self, iteration, overall_metric, meter):
        """
        Method to be called everytime you need to check whether to
        early stop or not
        Arguments:
            iteration {number}: Current iteration number
            overall_metric {dict}: overall metric summarized by Metric
            meter {SmoothedValue}: metric generated in training process
        Returns:
            int -- Tells whether early stopping occurred or not. nccl not support boolean type
        """
        if not is_main_process():
            return 0

        # global_avg of total_loss is from meter
        if self.monitored_metric.endswith("total_loss"):
            value = meter.meters.get(self.monitored_metric, None)
            if value is not None:
                value = value.global_avg
        else:
            value = overall_metric.get(self.monitored_metric, None)
        if value is None:
            raise ValueError(
                f"Metric used for early stopping ({self.monitored_metric}) is not present in meter."
            )

        if isinstance(value, torch.Tensor):
            value = value.item()

        if (self.minimize and value < self.best_monitored_value) or (
            not self.minimize and value > self.best_monitored_value
        ):
            self.best_monitored_value = value
            self.best_monitored_iteration = iteration
            self.checkpoint.save(iteration, update_best=True)

        elif self.best_monitored_iteration + self.patience < iteration:
            self.activated = True
            if self.should_stop is True:
                # since only master process will get here,
                # synchronize before restoring will causing block.
                self.checkpoint.restore(with_sync=False)
                self.checkpoint.finalize()
                return 1  # nccl not support boolean type
            else:
                return 0
        else:
            self.checkpoint.save(iteration, update_best=False)

        return 0

    def is_activated(self):
        return self.activated

    def init_from_checkpoint(self, load):
        if "best_iteration" in load:
            self.best_monitored_iteration = load["best_iteration"]

        if "best_metric_value" in load:
            self.best_monitored_value = load["best_metric_value"]

    def get_info(self):
        return {
            "best iteration": self.best_monitored_iteration,
            "best {}".format(self.metric): "{:.4f}".format(self.best_monitored_value),
        }
