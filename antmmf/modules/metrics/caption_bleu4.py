# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import torch

from antmmf.modules.metrics import BaseMetric
from antmmf.common.registry import registry


@registry.register_metric("caption_bleu4")
class CaptionBleu4Metric(BaseMetric):
    """Metric for calculating caption accuracy using BLEU4 Score.

    **Key:** ``caption_bleu4``
    """

    import nltk.translate.bleu_score as bleu_score

    def __init__(self, name: str = "caption_bleu4"):
        super().__init__(name)
        self.caption_processor = registry.get("coco_caption_processor")
        assert (
            self.caption_processor is not None
        ), "missing coco_caption_processor in registry"

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: bleu4 score.

        """
        # Create reference and hypotheses captions.
        references = []
        hypotheses = []

        # References
        targets = sample_list.answers
        for j, p in enumerate(targets):
            img_captions = [
                self.caption_processor(c)["tokens"] for c in targets[j].tolist()
            ]
            references.append(img_captions)

        # Hypotheses
        scores = torch.max(model_output["logits"], dim=-1)[1]
        scores = scores.tolist()
        predictions = []
        for j, p in enumerate(scores):
            caption = self.caption_processor(scores[j])["tokens"]
            predictions.append(caption)
        hypotheses.extend(predictions)

        assert len(references) == len(hypotheses)

        bleu4 = self.bleu_score.corpus_bleu(references, hypotheses)

        return targets.new_tensor(bleu4, dtype=torch.float)
