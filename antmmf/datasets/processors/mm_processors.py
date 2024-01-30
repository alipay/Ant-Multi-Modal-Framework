# Copyright (c) 2023 Ant Group and its affiliates.
"""
The processors exist in antmmf to make data processing pipelines in various
datasets as similar as possible while allowing code reuse.

The processors also help maintain proper abstractions to keep only what matters
inside the dataset's code. This allows us to keep the dataset ``get_item``
logic really clean and no need about maintaining opinions about data type.
Processors can work on both images and text due to their generic structure.

To create a new processor, follow these steps:

1. Inherit the ``BaseProcessor`` class.
2. Implement ``_call`` function which takes in a dict and returns a dict with
   same keys preprocessed as well as any extra keys that need to be returned.
3. Register the processor using ``@registry.register_processor('name')`` to
   registry where 'name' will be used to refer to your processor later.

In processor's config you can specify ``preprocessor`` option to specify
different kind of preprocessors you want in your dataset.

Let's break down processor's config inside a dataset (VQA2.0) a bit to understand
different moving parts.

Config::

    task_attributes:
        vqa:
            dataset_attributes:
                vqa2:
                    processors:
                      text_processor:
                        type: vocab
                        params:
                          max_length: 14
                          vocab:
                            type: intersected
                            embedding_name: glove.6B.300d
                            vocab_file: vocabs/vocabulary_100k.txt
                          answer_processor:
                            type: vqa_answer
                            params:
                              num_answers: 10
                              vocab_file: vocabs/answers_vqa.txt
                              preprocessor:
                                type: simple_word
                                params: {}

``BaseDataset`` will init the processors and they will be available inside your
dataset with same attribute name as the key name, e.g. `text_processor` will
be available as `self.text_processor` inside your dataset. As is with every module
in antmmf, processor also accept a ``Configuration`` with a `type` and `params`
attributes. `params` defined the custom parameters for each of the processors.
By default, processor initialization process will also init `preprocessor` attribute
which can be a processor config in itself. `preprocessor` can be then be accessed
inside the processor's functions.

Example::

    from antmmf.common.registry import registry
    from antmmf.tasks.processors import BaseProcessor


    class MyProcessor(BaseProcessor):
        def __init__(self, config, *args, **kwargs):
            return

        def __call__(self, item, *args, **kwargs):
            text = item['text']
            text = [t.strip() for t in text.split(" ")]
            return {"text": text}
"""

import warnings
import torch

from antmmf.common.registry import registry
from antmmf.utils.text_utils import VocabDict
from antmmf.datasets.processors.processors import BaseProcessor, Processor


@registry.register_processor("vqa_answer")
class VQAAnswerProcessor(BaseProcessor):
    """Processor for generating answer scores for answers passed using VQA
    accuracy formula. Using VocabDict class to represent answer vocabulary,
    so parameters must specify "vocab_file". "num_answers" in parameter config
    specify the max number of answers possible. Takes in dict containing
    "answers" or "answers_tokens". "answers" are preprocessed to generate
    "answers_tokens" if passed.

    Args:
        config (Configuration): Configuration for the processor

    Attributes:
        answer_vocab (VocabDict): Class representing answer vocabulary
    """

    DEFAULT_NUM_ANSWERS = 10

    def __init__(self, config, *args, **kwargs):
        self.writer = registry.get("writer")
        if not hasattr(config, "vocab_file"):
            raise AttributeError(
                "'vocab_file' argument required, but not present in AnswerProcessor's config"
            )

        self.answer_vocab = VocabDict(config.vocab_file, *args, **kwargs)

        self.preprocessor = None

        if hasattr(config, "preprocessor"):
            self.preprocessor = Processor(config.preprocessor)

            if self.preprocessor is None:
                raise ValueError(
                    "No processor named {} is defined.".format(config.preprocessor)
                )

        if hasattr(config, "num_answers"):
            self.num_answers = config.num_answers
        else:
            self.num_answers = self.DEFAULT_NUM_ANSWERS
            warnings.warn(
                "'num_answers' not defined in the config. Setting to default of {}".format(
                    self.DEFAULT_NUM_ANSWERS
                )
            )

    def __call__(self, item):
        """Takes in dict with answers or answers_tokens, and returns back
        a dict with answers (processed), "answers_indices" which point to
        indices of the answers if present and "answers_scores" which represent
        VQA style scores for the answers.

        Args:
            item (Dict): Dict containing answers or answers_tokens

        Returns:
            Dict: Processed answers, indices and scores.

        """
        tokens = None

        if not isinstance(item, dict):
            raise TypeError("'item' passed to processor must be a dict")

        if "answer_tokens" in item:
            # have done the tokenization
            tokens = item["answer_tokens"]
        elif "answers" in item:
            # need tokenization of the answer words
            if self.preprocessor is None:
                raise AssertionError(
                    "'preprocessor' must be defined if you don't pass 'answer_tokens'"
                )

            tokens = [
                self.preprocessor({"text": answer})["text"]
                for answer in item["answers"]
            ]
        else:
            raise AssertionError(
                "'answers' or 'answer_tokens' must be passed to answer processor in a dict"
            )

        answers_indices = torch.zeros(self.num_answers, dtype=torch.long)
        answers_indices.fill_(self.answer_vocab.get_unk_index())

        for idx, token in enumerate(tokens):
            answers_indices[idx] = self.answer_vocab.word2idx(token)

        answers_scores = self.compute_answers_scores(answers_indices)

        return {
            "answers": tokens,
            "answers_indices": answers_indices,
            "answers_scores": answers_scores,
        }

    def get_vocab_size(self):
        """Get vocab size of the answer vocabulary. Can also include
        soft copy dynamic answer space size.

        Returns:
            int: size of the answer vocabulary

        """
        return self.answer_vocab.num_vocab

    def get_true_vocab_size(self):
        """True vocab size can be different from normal vocab size in some cases
        such as soft copy where dynamic answer space is added.

        Returns:
            int: True vocab size.

        """
        return self.answer_vocab.num_vocab

    def word2idx(self, word):
        """Convert a word to its index according to vocabulary

        Args:
            word (str): Word to be converted to index.

        Returns:
            int: Index of the word.

        """
        return self.answer_vocab.word2idx(word)

    def idx2word(self, idx):
        """Index to word according to the vocabulary.

        Args:
            idx (int): Index to be converted to the word.

        Returns:
            str: Word corresponding to the index.

        """
        return self.answer_vocab.idx2word(idx)

    def compute_answers_scores(self, answers_indices):
        """Generate VQA based answer scores for answers_indices.
        Args:
            answers_indices (torch.LongTensor): tensor containing indices of the answers

        Returns:
            torch.FloatTensor: tensor containing scores.

        """
        scores = torch.zeros(self.get_vocab_size(), dtype=torch.float)
        gt_answers = list(enumerate(answers_indices))
        unique_answers = set(answers_indices.tolist())

        for answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]

                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            avg_acc = sum(accs) / len(accs)

            if answer != self.answer_vocab.UNK_INDEX:
                scores[answer] = avg_acc

        return scores
