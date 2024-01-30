# Copyright (c) 2023 Ant Group and its affiliates.

# Adversarial Training

import math
import torch
from antmmf.common import constants
from antmmf.common.meter import Meter
from antmmf.common.registry import registry
from antmmf.common.task_loader import TaskLoader
from antmmf.trainers.build import build_adversarial_training
from antmmf.utils.logger import Logger
from antmmf.utils.timer import Timer
from tqdm import tqdm
from antmmf.trainers.base_trainer import BaseTrainer, check_configuration


@registry.register_trainer("adv_trainer")
class AdvTrainer(BaseTrainer):
    ALLOWED_RUNTYPE = BaseTrainer.ALLOWED_RUNTYPE + [
        # obtain adversarial inputs for adversarial training
        "adversarial_train_generate",
        "adversarial_val_generate",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.dump_data = self.config.get("dump_data", False)
        self.adv_max_iter = self.config.adversarial_parameters.get("max_iter", 1)
        self.rel_adv_weight = self.config.adversarial_parameters.get(
            "rel_adv_weight", 0.5
        )

    def load(self, has_check_point=True):
        self._init_process_group()
        self.writer = Logger(self.config)

        self.run_type = self.config.training_parameters.get("run_type", "train")
        assert (
            self.run_type in AdvTrainer.ALLOWED_RUNTYPE
        ), "unrecognized run_type:{}".format(self.run_type)

        self.task_loader = TaskLoader(self.config)

        # flag indicates train/val/test state, not online serving
        registry.register(constants.STATE, constants.STATE_LOCAL)

        registry.register("writer", self.writer)

        self.configuration = check_configuration(registry.get("configuration"))
        self.writer.write(self.configuration)

        self.config_based_setup()

        self.load_task()
        self.load_model()
        self.load_optimizer()
        self.load_extras(has_check_point)

    def load_extras(self, has_check_point=True):
        super().load_extras(has_check_point)

        self.adversarial_obj, self.adv_optimizer = build_adversarial_training(
            self.config, self.model
        )

        self.meter_adv = Meter()

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)

        if "generate" in self.run_type:
            self.generation()
            return

        should_break = False

        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_iterations = min(
                self.max_iterations, self.max_epochs * self.epoch_iterations
            )

        self.model.eval()
        self.adversarial_obj.train()

        data_updater = (
            self.task_loader.get_data_updater(self.run_type) if self.dump_data else None
        )

        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        self.profile("Setup Time")

        self.writer.write("Starting training...")
        while self.current_iteration < self.max_iterations and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            self.task_loader.seed_sampler("train", self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break
            for batch in tqdm(self.train_loader):
                self.profile("Batch load time")
                self.current_iteration += 1
                self.writer.write(self.current_iteration, "debug")
                registry.register("current_iteration", self.current_iteration)
                if self.current_iteration > self.max_iterations:
                    break

                # model to be attacked under eval mode, so behaviours such as drop-out can be fixed
                self.model.eval()

                def _acc_on_samples(obs):
                    report, mdl_output, _ = self._forward_pass(obs)
                    if report is None:
                        return None, None, None

                    loss = self._extract_loss(report)

                    return report, loss, mdl_output

                # first doing training on clean samples first
                # obtain natural loss
                report_nat, loss_nat, org_model_output = _acc_on_samples(batch)
                if report_nat is None:
                    continue

                orig_arg_max = torch.argmax(org_model_output["logits"], dim=-1)

                # secondly, doing adversarial training
                batch_adv, data_report, output = self.adversarial_obj.attack(
                    self.adv_optimizer,
                    batch,
                    self.config.adversarial_parameters.max_iter,
                    orig_arg_max,
                )

                report_adv, loss_adv, adv_model_output = _acc_on_samples(batch_adv)
                if report_adv is None:
                    continue

                self._backward(
                    loss_nat * (1.0 - self.rel_adv_weight)
                    + loss_adv * self.rel_adv_weight
                )

                self._update_meter(report_nat, self.meter)
                self._update_meter(report_adv, self.meter_adv)

                if data_updater is not None:
                    data_updater.add_to_report(data_report)
                    data_updater.flush_intermediate_report()

                self._run_scheduler()

                should_break = self._logistics(report_nat)
                if should_break:
                    break

            if data_updater is not None:
                data_updater.flush_report()

        self.finalize()

    def generation(self):
        if "val" in self.run_type:
            self._adversarial_data_generation("val")

        if "train" in self.run_type:
            self._adversarial_data_generation("train")

    def _adversarial_data_generation(self, dataset_type):
        self.writer.write("===== Adversarial Data Generation based on Model =====")
        self.writer.write(self.model)

        reporter = self.task_loader.get_data_updater(dataset_type)

        self.model.eval()
        self.adversarial_obj.train()
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        self.profile("Setup Time")

        torch.autograd.set_detect_anomaly(True)

        self.writer.write(
            "Starting adversarial attacks to generate adversarial samples ..."
        )

        # Seed the sampler in case if it is distributed
        self.task_loader.seed_sampler("train", self.current_epoch)

        while reporter.next_dataset():
            dataloader = reporter.get_dataloader()
            b = 1
            for batch in tqdm(dataloader):
                prepared_batch = reporter.prepare_batch(batch)
                if prepared_batch is None:
                    continue

                # model to be attacked under eval mode, so behaviours such as drop-out can be fixed
                self.model.eval()
                _, report, _ = self.adversarial_obj.attack(
                    self.adv_optimizer, prepared_batch, self.adv_max_iter, None
                )

                reporter.add_to_report(report)

                if b % self.adversarial_obj.flush_updated_data_interval == 0:
                    reporter.flush_intermediate_report()
                b += 1

            reporter.flush_report()
