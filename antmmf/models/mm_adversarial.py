# Copyright (c) 2023 Ant Group and its affiliates.
import torch
import numpy as np
from antmmf.common.registry import registry
from antmmf.models.base_adversarial import BaseAdversarial
from antmmf.common.report import Report
import copy
from antmmf.common import constants


def _get_input_ids(text, mask):
    length = text.size(0)
    # exclude [SEP] at the end
    input_ids = [int(text[l]) for l in range(length) if mask[l] == 1]
    last_id = input_ids.pop()
    return input_ids, last_id


def _unconstrained_adversarial_distance(param, base_param, grad, sign=1):
    """
    compute the inner product between the difference of parameter and a base parameter and
    a gradient

    to compute Eq. (3)
    in https://arxiv.org/pdf/1903.06620.pdf
    """
    bsz = grad.size(0)
    score = []
    # too large, so do explicit loop with batches
    for b in range(bsz):
        dist = param - base_param[b].squeeze().unsqueeze(0).expand(param.size(0), -1)
        dist = sign * dist
        score.append(
            torch.mean(
                dist * grad[b].squeeze().unsqueeze(0).expand(param.size(0), -1), dim=-1
            ).unsqueeze(0)
        )

    score = torch.cat(score, dim=0)
    return score


@registry.register_adversarial("MMFreeLB")
class MMFreeLB(BaseAdversarial):
    # maximum buffer size informaton
    MAX_BATCH = 32
    CHN_NUMBER = 3
    MAX_WIDTH = 1024
    MAX_HEIGHT = 1024

    def __init__(self, config, model):
        super().__init__(config, model)

        self._datasets = []
        for _, attr in registry.get("config").task_attributes.items():
            for dataset in attr.dataset_attributes:
                self._datasets.append(dataset)
        self._tokenizer = registry.get("{}_text_tokenizer".format(self._datasets[0]))

        self.device = self.config.training_parameters.device

        self._image_delta_buffer = torch.zeros(
            (
                MMFreeLB.MAX_BATCH,
                MMFreeLB.CHN_NUMBER,
                MMFreeLB.MAX_WIDTH * MMFreeLB.MAX_HEIGHT,
            ),
            requires_grad=True,
            dtype=torch.float,
            device=self.config.training_parameters.device,
        )

        self._all_diff = self.config.adversarial_parameters.get("all_diff", True)

    def _init_attack(self, sample_list, optimizer):
        self.position_changed = {}
        sample = self._adv_build_clean_sample(sample_list)
        self._image_delta_buffer.data.fill_(0.0)

        image_norm = sample.image.norm()
        optimizer.reset({"orig_norm": {constants.IMAGE_MODALITY: image_norm}})

        self.model.init_adv_train()

        return sample

    def attack(self, external_optimizer, sample, max_iter=None, orig_max=None):

        sample = self._init_attack(sample, external_optimizer)

        iter = 0
        orig_arg_max = (
            copy.deepcopy(orig_max.detach()) if orig_max is not None else None
        )
        report = None

        iter_nbr = self.max_iter if max_iter is None else max_iter
        model_output = None
        while iter < iter_nbr:
            # need to change different positions
            model_output, noisy_sample = self.forward(sample)

            report = Report(noisy_sample, model_output)
            if report is None:
                continue

            arg_max = torch.argmax(model_output["logits"], dim=-1)
            if orig_arg_max is not None:
                diff = np.array(
                    [
                        1
                        for l in range(len(orig_arg_max))
                        if orig_arg_max[l] != arg_max[l]
                    ]
                )
                diff = diff.sum()
                if diff == len(orig_arg_max) and self._all_diff:
                    # stop further if adversarial has changed all predictions from natural training
                    break
                elif diff > 0 and self._all_diff is False:
                    # stop further if adversarial has changed any predictions from natural training
                    break
            else:
                orig_arg_max = arg_max

            # here it is to incur loss from adversarial
            loss = self._extract_loss(report)

            # update noisy sample
            sample = self.backward(noisy_sample, external_optimizer, loss)

            iter += 1

        sample = self._adv_build_noisy_sample(sample)

        report = self._adv_update_report(report, sample)

        return sample, report, model_output

    def forward(self, sample_list):

        noisy_sample = self._adv_build_noisy_sample(sample_list)

        noisy_sample = noisy_sample.to(self.device)
        noisy_sample.dataset_type = sample_list.dataset_type
        noisy_sample.dataset_name = sample_list.dataset_name

        pred = self.model(noisy_sample)

        return pred, noisy_sample

    def backward(self, samples, optimizer, loss):

        # obtain gradients and updates adversarials
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        return samples

    def _adv_build_clean_sample(self, sample_list):
        # for adversarial training of this model

        self.clean_sample = sample_list

        if constants.TEXT_MODALITY in self.clean_sample:
            self.clean_sample[constants.TEXT_MODALITY] = (
                self.clean_sample[constants.TEXT_MODALITY]
                .detach()
                .to(device=self.config.training_parameters.device)
            )
        if constants.IMAGE_MODALITY in self.clean_sample:
            self.clean_sample[constants.IMAGE_MODALITY] = (
                self.clean_sample[constants.IMAGE_MODALITY]
                .detach()
                .to(device=self.config.training_parameters.device)
            )

            self.acc_image_delta = torch.zeros_like(
                self.clean_sample[constants.IMAGE_MODALITY],
                device=self.config.training_parameters.device,
            )

        return copy.deepcopy(self.clean_sample)

    def get_optimizer_parameters(self, config):

        adv_params = {}
        for k in list(config["adversarial_optimizer_parameters"]["params"].keys()):
            v = config["adversarial_optimizer_parameters"]["params"][k]
            adv_params.update({k: v})

        # include image modality
        adv_params.update(
            {
                "params": self._image_delta_buffer,
                "modality": constants.IMAGE_MODALITY,
                "name": "image_distortion",
            }
        ),
        self.adv_params = [adv_params]

        # include text modality
        model_adv_paramers = self.model.get_adv_parameters()
        for d in model_adv_paramers:
            d.update({"modality": constants.TEXT_MODALITY})
        self.adv_params.extend(model_adv_paramers)

        return self.adv_params

    def _adv_build_noisy_sample(self, current_sample_list):

        noisy_sample = current_sample_list

        # free-lb method learns improved word-embeddings from adversarials
        # this improved word-embeddings are persistent across batchs
        # updating word embeddings is carried out in adv_free_lb optimizer
        if constants.TEXT_MODALITY in current_sample_list:
            text = current_sample_list[constants.TEXT_MODALITY].detach()
            noisy_sample[constants.TEXT_MODALITY] = text

        # image pertrubation is added here
        if constants.IMAGE_MODALITY in current_sample_list:
            image = self.clean_sample[constants.IMAGE_MODALITY].detach()
            image_shape = image.shape
            bsz, chn = image_shape[0], image_shape[1]
            fea_dim = np.prod(image_shape[2:])
            noisy_sample[constants.IMAGE_MODALITY] = image + self._image_delta_buffer[
                0:bsz, :chn, :fea_dim
            ].reshape(image_shape)

        return noisy_sample

    def _adv_update_report(self, report, noisy_sample):

        if constants.TEXT_MODALITY in noisy_sample:
            text = noisy_sample[constants.TEXT_MODALITY]
            mask = noisy_sample["mask"]
            bsz, length = text.shape

            report.org_tokens = []
            report.adv_tokens = []
            report.clean_text = self.clean_sample[constants.TEXT_MODALITY]

            report.noisy_text = text
            for b in range(bsz):

                input_ids, _ = _get_input_ids(
                    self.clean_sample[constants.TEXT_MODALITY][b], mask[b]
                )
                report.org_tokens.append(
                    self._tokenizer.convert_ids_to_tokens(ids=input_ids)
                )

                input_ids, _ = _get_input_ids(text[b], mask[b])

                report.adv_tokens.append(
                    self._tokenizer.convert_ids_to_tokens(ids=input_ids)
                )

        if constants.IMAGE_MODALITY in noisy_sample:
            imag = noisy_sample[constants.IMAGE_MODALITY].detach()
            image_shape = imag.shape

            bsz, chn = image_shape[0], image_shape[1]
            feat_dim = np.prod(image_shape[2:])

            report.clean_image = self.clean_sample[constants.IMAGE_MODALITY]
            report.noisy_image = imag
            report.image = imag

            report.image_delta = (
                self._image_delta_buffer[0:bsz, :chn, :feat_dim]
                .reshape(image_shape)
                .clone()
                .detach()
            )

        return report


@registry.register_adversarial("MMHotFlip")
class MMHotFlip(MMFreeLB):
    def __init__(self, config, model):
        super().__init__(config, model)

        self.sign = 1 if self.away_from_target else -1

        self._attack_modalities = self.adversarial_parameters.get(
            "attack_modalities", [constants.IMAGE_MODALITY, constants.TEXT_MODALITY]
        )

    def backward(self, samples, optimizer, loss):

        # obtain gradients and updates adversarials
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # before updating text adversarials, obtain base word embeddings
        text_params_info = self._adv_text_param(samples)
        self.adv_info = self._backward(samples, text_params_info)

        updated_sample = self._adv_update_noisy_sample(samples, self.adv_info)

        return updated_sample

    def get_optimizer_parameters(self, config):

        adv_params = {}
        for k in list(config["adversarial_optimizer_parameters"]["params"].keys()):
            v = config["adversarial_optimizer_parameters"]["params"][k]
            adv_params.update({k: v})

        # include image modality
        adv_params.update(
            {
                "params": self._image_delta_buffer,
                "modality": constants.IMAGE_MODALITY,
                "name": "image_distortion",
            }
        ),
        self.adv_params = [adv_params]

        # include text modality
        model_adv_paramers = self.model.get_adv_parameters()
        self.adv_params.extend(model_adv_paramers)
        return self.adv_params

    def _backward(self, samples, base_params_info):
        """
        compute the inner product between the difference of parameter and a base parameter and
        a gradient,
        different from directly solving equation (3) here https://arxiv.org/abs/1903.06620,
        which is very slow,

        instead, follow hotflip that locates indices with the maximum gradient norm
        https://github.com/allenai/allennlp/blob/master/allennlp/interpret/attackers/hotflip.py

        for positions that have been changed before, mask them and find other positions to change
        """

        param = base_params_info["param"]
        embedding = base_params_info["base_param"]
        grads = self._adv_grad(samples, param)["grad"]

        bsz = grads.size(0)

        # compute L2 norm of all grads
        grads_magnitude = torch.cat(
            [
                torch.tensor([g.dot(g) for g in grads[b]])
                .unsqueeze(0)
                .to(device=grads.device)
                for b in range(bsz)
            ],
            dim=0,
        )
        assert grads_magnitude[0][0] > 0

        # only flip a token once
        for bidx in range(bsz):
            if bidx not in self.position_changed:
                self.position_changed[bidx] = []
            grads_magnitude[bidx][
                torch.tensor(self.position_changed[bidx]).long()
            ] = torch.tensor(-np.inf).to(device=grads.device)

        # We flip the token with highest gradient norm.
        index_of_token_to_flip = torch.argmax(grads_magnitude, dim=-1)
        row_mask = []
        base_param = []
        base_grads = []
        for b in range(bsz):
            if grads_magnitude[b][index_of_token_to_flip[b]] <= 0:
                # If we've already flipped all of the tokens, we give up.
                row_mask.append(0)
                index_of_token_to_flip[b] = -1
                continue
            self.position_changed[b].append(index_of_token_to_flip[b])

            base_param.append(
                embedding[b, index_of_token_to_flip[b], :].squeeze().unsqueeze(0)
            )
            base_grads.append(
                grads[b, index_of_token_to_flip[b], :].squeeze().unsqueeze(0)
            )
            row_mask.append(1)

        if len(base_param) > 0:
            base_param = torch.cat(base_param, dim=0)
            base_grads = torch.cat(base_grads, dim=0)

            score = _unconstrained_adversarial_distance(
                param, base_param, base_grads, self.sign
            )

            worst_idx = torch.argmax(score, dim=-1)
        else:
            worst_idx = []

        adv_info = {"replaced_pos": index_of_token_to_flip, "replacing_id": worst_idx}

        return adv_info

    def _adv_text_param(self, noisy_samples):
        text = noisy_samples[constants.TEXT_MODALITY]

        idx = -1
        for i, param in enumerate(self.adv_params):
            if (
                "word_embeddings.weight" in param["name"]
                or "text_embedding.weight" in param["name"]
            ):
                idx = i
                break

        assert idx >= 0, "cannot find word_embeddings.weight to attack"

        backward_param = self.adv_params[idx]["params"]
        if isinstance(backward_param, list):
            assert len(backward_param) == 1
            backward_param = backward_param[0]

        embedding = torch.cat(
            [backward_param[text[b]].unsqueeze(0) for b in range(text.size(0))], dim=0
        )
        return {"param": backward_param, "base_param": embedding}

    def _adv_grad(self, noisy_samples, backward_param):
        text = noisy_samples[constants.TEXT_MODALITY]
        mask = noisy_samples["mask"]
        bsz, length = text.shape

        grads = torch.cat(
            [backward_param.grad[text[b]].unsqueeze(0) for b in range(text.size(0))],
            dim=0,
        )
        grads = grads * mask.unsqueeze(-1).expand(-1, -1, grads.size(-1))
        return {"grad": grads}

    def _attack_text(self, noisy_samples, adv_info):

        pos = adv_info["replaced_pos"]
        new_id = adv_info["replacing_id"]

        text = noisy_samples[constants.TEXT_MODALITY].detach()

        def _update_text(txt, pos, new_id):
            # the new_id only saves for valid positions,
            # therefore it has smaller number of rows than position ids
            bsz = pos.size(0)
            i_iter = 0
            for b in range(bsz):
                p = pos[b]
                if p >= 0:
                    a = new_id[i_iter]
                    txt[b][p] = a.clone().detach().long()
                    i_iter += 1
            return txt

        text = _update_text(text, pos, new_id)
        return text

    def _adv_update_noisy_sample(self, noisy_samples, adv_info):
        samples = noisy_samples
        for modality in self._attack_modalities:
            if modality == constants.TEXT_MODALITY:
                samples[constants.TEXT_MODALITY] = self._attack_text(
                    noisy_samples, adv_info
                )

        return samples
