# Copyright (c) 2023 Ant Group and its affiliates.
import os
from typing import List

import numpy
import torch

from antmmf.common.constants import SS_GRAD_INPUT, SS_SALIENCE, EPSILON
from antmmf.common.registry import registry
from antmmf.utils.general import jsonl_dump, get_absolute_path


class Interpreter(object):
    """
    An `Interpreter` interprets a predictor's outputs by assigning a saliency
    score to each input token or model parameters
    """

    def __init__(self, predictor, config) -> None:
        self._predictor = predictor
        self._config = config
        datasets = config.get("datasets", [])
        assert len(datasets) > 0
        self._datasets = datasets[0] if isinstance(datasets, list) else datasets
        self._tokenizer = registry.get("{}_text_tokenizer".format(self._datasets))
        self._serialization = self._config.get("serialization", [])
        for s in self._serialization:
            if s != "plot":
                fn = get_absolute_path(s)
                if os.path.exists(fn):
                    os.remove(fn)

    def interpret(self, inputs):
        """
        This function finds saliency values for each input token.
        # Parameters
        inputs : `JsonDict`
            The input you want to interpret (the same as the argument to a predictor, e.g., forward).
        # Returns
        interpretation : `JsonDict`
            Contains the normalized saliency values for each input token. The dict has entries for
            each instance in the inputs JsonDict, e.g., `{instance_1: ..., instance_2:, ... }`.
            Each one of those entries has entries for the saliency of the inputs, e.g.,
            `{grad_input_1: ..., grad_input_2: ... }`.
        """
        raise NotImplementedError("Implement this for saliency interpretations")

    def get_gradient(self, instance, embedding_layer):
        embedding_gradients: List[torch.Tensor] = []
        hooks = self._register_embedding_gradient_hooks(
            embedding_gradients, embedding_layer
        )

        outputs = self._predictor(instance)  # type: ignore
        assert (
            "losses" in outputs
        ), 'loss should be computed in function "get_gradients"'
        loss = outputs["losses"]
        # Zero gradients.
        self._predictor.zero_grad()

        assert (
            len(loss) == 1
        ), "support only one loss to avoid confusion from multiple losses for interpretation"
        for ln, lv in loss.items():
            lv.backward()

        for hook in hooks:
            hook.remove()

        return embedding_gradients, outputs

    def _compute_saliency(self, grad_dict, embeddings_list):
        embedding = embeddings_list[0]
        grad = grad_dict[SS_GRAD_INPUT]
        if len(grad.shape) == len(embedding.shape) + 1:
            embedding = numpy.expand_dims(embedding, axis=0)
        assert (
            grad.shape == embedding.shape
        ), f"grad shape ${grad.shape} is not equal to embedding shape ${embedding.shape}"
        embedding = numpy.array(grad * embedding)
        emb_grad = numpy.sum(embedding, axis=-1)
        norm = numpy.linalg.norm(emb_grad, axis=-1, ord=1)
        normalized_grad = numpy.matmul(
            numpy.diag(1.0 / (EPSILON + norm)), numpy.fabs(emb_grad)
        )
        assert len(normalized_grad.shape) == 2
        grad_dict[SS_SALIENCE] = normalized_grad
        return grad_dict

    @staticmethod
    def _aggregate_token_embeddings(
        embeddings_list: List[torch.Tensor],
    ) -> List[numpy.ndarray]:
        return [embeddings.cpu().numpy() for embeddings in embeddings_list]

    def export(self, attr_info, instance):
        from antmmf.utils.visualize import (
            visualize_text_importance,
            visualize_image_importance,
        )

        vis_info = {}
        for nm, info in attr_info.items():
            modality = info["modality"]

            title = info["title"]
            vis_info[modality] = {"title": title}

            if modality == "text":
                vis_info[modality]["attributions"] = self._prep_text_attribution(
                    info["attributions"], instance
                )
            if modality == "image":
                vis_info[modality]["attributions"] = self._prep_image_attribution(
                    info["attributions"], instance
                )

        for s in self._serialization:
            if s == "plot":
                for m, vis in vis_info.items():
                    title = vis["title"]
                    att = vis["attributions"]
                    if m == "text":
                        visualize_text_importance(att, self._tokenizer.pad_token, title)
                    if m == "image":
                        visualize_image_importance(att, title)
            else:
                info = []
                for m, vis in vis_info.items():
                    att = vis["attributions"]
                    if m == "text":
                        lcl_info = _formta_text_importance_str(
                            att, self._tokenizer.pad_token
                        )
                    if m == "image":
                        lcl_info = _format_image_importance_str(att)

                    if len(info) == 0:
                        info = lcl_info
                    else:
                        for idx, ifo in enumerate(lcl_info):
                            info[idx].update(ifo)
                jsonl_dump(info, get_absolute_path(s), True)

    def _prep_text_attribution(self, attributions, instance):
        from antmmf.utils.visualize import VisualizationDataRecord

        bsz = attributions[SS_SALIENCE].shape[0]
        ln = attributions[SS_SALIENCE].shape[1]

        assert bsz == len(instance.text)
        assert ln == len(instance.text[0])

        attr, text_attributions = _extract_normalized_salience_across_time(attributions)
        pred, prob, answer_idx = _extract_pred_answer(attributions)

        vis_data_records = []
        for idx in range(bsz):
            text_ids = instance["text"][idx]
            text_str = self._tokenizer.convert_ids_to_tokens(ids=text_ids)

            target = instance["targets"][idx]
            target = target.detach().cpu().numpy()

            vis_info = VisualizationDataRecord(
                attr[idx],
                prob[idx],
                answer_idx[idx],
                answer_idx[idx],
                target,
                text_attributions[idx],
                text_str,
                0.0,
            )
            vis_data_records.append(vis_info)
        return vis_data_records

    def _prep_image_attribution(self, attributions, instance):
        from antmmf.utils.visualize import VisualizationDataRecord

        bsz = attributions[SS_SALIENCE].shape[0]
        ln = attributions[SS_SALIENCE].shape[1]
        # Visualize text attributions
        assert bsz == len(instance.image)

        attr, image_attributions = _extract_normalized_salience_across_time(
            attributions
        )
        pred, prob, answer_idx = _extract_pred_answer(attributions)

        vis_data_records = []
        for idx in range(bsz):
            target = instance["targets"][idx]
            target = target.detach().cpu().numpy()

            vis_info = VisualizationDataRecord(
                attr[idx],
                prob[idx],
                answer_idx[idx],
                answer_idx[idx],
                target,
                image_attributions[idx],
                [],
                0.0,
            )
            vis_data_records.append(vis_info)
        return vis_data_records


def _extract_normalized_salience_across_time(attr):
    attributions = attr[SS_SALIENCE]
    salience = numpy.fabs(attributions)
    attributions = numpy.sum(salience, axis=-1)

    return salience, attributions


def _extract_pred_answer(attr):
    pred = attr["logits"]
    prob, answer_idx = torch.nn.functional.softmax(pred, dim=-1).data.cpu().max(dim=-1)
    prob = prob.detach().cpu().numpy()
    answer_idx = answer_idx.detach().cpu().numpy()

    return pred, prob, answer_idx


def _formta_text_importance_str(visual_record, pad_token):
    assert isinstance(visual_record, list)

    info = []
    for vr in visual_record:
        max_pad_pos = -1
        for idx, st in enumerate(vr.raw_input):
            if st == pad_token:
                max_pad_pos = max(max_pad_pos, idx)
                break
        token_score = []
        for wrd, scr in zip(
            vr.raw_input[:max_pad_pos], vr.word_attributions[:max_pad_pos]
        ):
            token_score.append((wrd, float(scr)))

        prd = int(vr.pred_class)
        prob = float(vr.pred_prob)
        tgt = int(vr.true_class)
        info.append(
            {
                "target": tgt,
                "pred": prd,
                "probability": prob,
                "token_attributions": token_score,
            }
        )
    return info


def _format_image_importance_str(visual_record):
    assert isinstance(visual_record, list)

    info = []
    for vr in visual_record:
        score = [float(f) for f in numpy.asarray(vr.word_attributions).flatten()]
        prd = int(vr.pred_class)
        prob = float(vr.pred_prob)
        tgt = int(vr.true_class)
        info.append(
            {
                "target": tgt,
                "pred": prd,
                "probability": prob,
                "image_attributions": score,
            }
        )

    return info
