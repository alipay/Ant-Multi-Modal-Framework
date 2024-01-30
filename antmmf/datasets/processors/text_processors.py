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
import multiprocessing
import os
import random
import warnings
from collections import Counter

import torch
import json

from antmmf.common import Configuration
from antmmf.common.constants import (
    CLS_TOKEN_STR,
    SEP_TOKEN_STR,
    TEXT_MODALITY,
    CLS_ID_STR,
    SEP_ID_STR,
    LM_LABEL_IDS_STR,
)
from antmmf.common.registry import registry
from antmmf.datasets.processors.mm_processors import VQAAnswerProcessor
from antmmf.datasets.processors.processors import BaseProcessor, Processor
from antmmf.utils.distributed_utils import synchronize
from antmmf.utils.general import get_antmmf_root
from antmmf.utils.phoc import build_phoc
from antmmf.utils.text_utils import START_TOKEN, END_TOKEN, keep_till_eos
from antmmf.utils.vocab import Vocab, WordToVectorDict


@registry.register_processor("vocab")
class VocabProcessor(BaseProcessor):
    """Use VocabProcessor when you have vocab file and you want to process
    words to indices. Expects UNK token as "<unk>" and pads sentences using
    "<pad>" token. Config parameters can have ``preprocessor`` property which
    is used to preprocess the item passed and ``max_length`` property which
    points to maximum length of the sentence/tokens which can be convert to
    indices. If the length is smaller, the sentence will be padded. Parameters
    for "vocab" are necessary to be passed.

    **Key**: vocab

    Example Config::

        task_attributes:
            vqa:
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

    Args:
        config (Configuration): node containing configuration parameters of
                             the processor

    Attributes:
        vocab (Vocab): Vocab class object which is abstraction over the vocab
                       file passed.
    """

    MAX_LENGTH_DEFAULT = 50
    PAD_TOKEN = "<pad>"
    PAD_INDEX = 0

    def __init__(self, config, *args, **kwargs):
        if not hasattr(config, "vocab"):
            raise AttributeError(
                "config passed to the processor has no attribute vocab"
            )

        self.vocab = Vocab(*args, **config.vocab, **kwargs)
        self._init_extras(config)

    def _init_extras(self, config, *args, **kwargs):
        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            warnings.warn(
                "No 'max_length' parameter in Processor's "
                "configuration. Setting to {}.".format(self.MAX_LENGTH_DEFAULT)
            )
            self.max_length = self.MAX_LENGTH_DEFAULT

        self.prepend_bos_append_eos = config.get("prepend_bos_append_eos", False)

        self.remove_unk = config.get("remove_unk", False)

        super()._init_extras(config)

    def __call__(self, item, *args, **kwargs):
        """Call requires item to have either "tokens" attribute or either
        "text" attribute. If "text" is present, it will be tokenized using
        the preprocessor.

        Args:
            item (Dict): Dict containing the "text" or "tokens".

        Returns:
            Dict: Dict containing indices in "text" key, "tokens" in "tokens"
                  key and "length" of the string in "length" key.

        """
        indices = None
        if not isinstance(item, dict):
            raise TypeError(
                "Argument passed to the processor must be "
                "a dict with either 'text' or 'tokens' as "
                "keys"
            )
        if "tokens" in item:
            # has done the tokenization
            tokens = item["tokens"]
            indices, tokens = self._map_strings_to_indices(tokens)
        elif "text" in item:
            # map raw text to tokens, via tokenization in preprocessor
            if self.preprocessor is None:
                raise AssertionError(
                    "If tokens are not provided, a text processor must be defined in the config"
                )

            tokens = self.preprocessor({"text": item["text"]}, *args, **kwargs)["text"]
            if self.prepend_bos_append_eos:
                length = min(len(tokens), self.max_length - 2)
                tokens = tokens[:length]
                tokens = [START_TOKEN] + tokens + [END_TOKEN]
                length += 2
            indices, tokens = self._map_strings_to_indices(tokens)
        else:
            raise AssertionError(
                "A dict with either 'text' or 'tokens' keys must be passed to the processor"
            )

        tokens, length = self._pad_tokens(tokens)

        return {"text": indices, "tokens": tokens, "length": length}

    def _pad_tokens(self, tokens):
        padded_tokens = [self.PAD_TOKEN] * self.max_length
        token_length = min(len(tokens), self.max_length)
        padded_tokens[:token_length] = tokens[:token_length]
        token_length = torch.tensor(token_length, dtype=torch.long)
        return padded_tokens, token_length

    def get_pad_index(self):
        """Get index of padding <pad> token in vocabulary.

        Returns:
            int: index of the padding token.

        """
        return self.vocab.get_pad_index()

    def get_vocab_size(self):
        """Get size of the vocabulary.

        Returns:
            int: size of the vocabulary.

        """
        return self.vocab.get_size()

    def get_vocab(self):
        return self.vocab

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.zeros(self.max_length, dtype=torch.long)
        output.fill_(self.vocab.get_pad_index())

        if self.remove_unk is True:
            tokens = [t for t in tokens if self.vocab.stoi.get(t) is not None]
        for idx, token in enumerate(tokens):
            output[idx] = self.vocab.stoi.get(token)
        return output, tokens


@registry.register_processor("glove")
class GloVeProcessor(VocabProcessor):
    """Inherits VocabProcessor, and returns GloVe vectors for each of the
    words. Maps them to index using vocab processor, and then gets GloVe vectors
    corresponding to those indices.

    Args:
        config (Configuration): Configuration parameters for GloVe same as
                             :func:`~VocabProcessor`.

    """

    def __init__(self, config, *args, **kwargs):
        if not hasattr(config, "vocab"):
            raise AttributeError(
                "Config passed to the processor has no attribute vocab"
            )
        vocab_processor_config = Configuration(config)
        # GloVeProcessor needs vocab type to be "intersected"
        vocab_processor_config.vocab.type = "intersected"

        if "vocab_file" not in vocab_processor_config.vocab:
            warnings.warn(
                "'vocab_file' key is not present in the config. Switching to pretrained vocab."
            )

            vocab_processor_config.vocab.type = "pretrained"

        super().__init__(vocab_processor_config, *args, **kwargs)

    def __call__(self, item):
        indices = super().__call__(item)["text"]
        embeddings = torch.zeros(
            (len(indices), self.vocab.get_embedding_dim()), dtype=torch.float
        )

        for idx, index in enumerate(indices):
            embeddings[idx] = self.vocab.vectors[index]

        return {"text": embeddings}


@registry.register_processor("fasttext")
class FastTextProcessor(VocabProcessor):
    """FastText processor, similar to GloVe processor but returns FastText vectors.

    Args:
        config (Configuration): Configuration values for the processor.

    """

    def __init__(self, config, *args, **kwargs):
        self._init_extras(config)
        self.config = config
        self._already_downloaded = False

    def _try_download(self):
        is_main_process = self._is_main_process()

        if self._already_downloaded:
            return

        if is_main_process:
            self.writer.write("Fetching fastText model for OCR processing")

        needs_download = False

        if not hasattr(self.config, "model_file"):
            if is_main_process:
                warnings.warn(
                    "'model_file' key is required but missing from FastTextProcessor's config."
                )
            needs_download = True

        model_file = self.config.model_file
        model_file = os.path.join(get_antmmf_root(), model_file)

        if not os.path.exists(model_file):
            if is_main_process:
                warnings.warn("No model file present at {}.".format(model_file))
            needs_download = True

        if needs_download:
            if is_main_process:
                self.writer.write("Downloading FastText bin", "info")
            model_file = self._download_model()

        synchronize()

        self._load_fasttext_model(model_file)
        self._already_downloaded = True

    def _download_model(self):
        is_main_process = self._is_main_process()

        model_file_path = os.path.join(
            get_antmmf_root(), ".vector_cache", "wiki.en.bin"
        )

        if not is_main_process:
            return model_file_path

        if os.path.exists(model_file_path):
            if is_main_process:
                self.writer.write(
                    "Vectors already present at {}.".format(model_file_path), "info"
                )
            return model_file_path

        import requests
        from antmmf.common.constants import FASTTEXT_WIKI_URL
        from tqdm import tqdm

        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        response = requests.get(FASTTEXT_WIKI_URL, stream=True)

        with open(model_file_path, "wb") as f:
            pbar = tqdm(
                total=int(response.headers["Content-Length"]) / 4096,
                miniters=50,
                disable=not is_main_process,
            )

            idx = 0
            for data in response.iter_content(chunk_size=4096):
                if data:
                    if idx % 50 == 0:
                        pbar.update(len(data))
                    f.write(data)
                    idx += 1

            pbar.close()

        if is_main_process:
            self.writer.write(
                "fastText bin downloaded at {}.".format(model_file_path), "info"
            )

        return model_file_path

    def _load_fasttext_model(self, model_file):
        from fasttext import load_model

        is_main_process = self._is_main_process()

        if is_main_process:
            self.writer.write("Loading fasttext model now from %s" % model_file)

        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)

        if is_main_process:
            self.writer.write("Finished loading fasttext model")

    def _is_main_process(self):
        return multiprocessing.current_process().name == "Process-1"

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.full(
            (self.max_length, self.model.get_dimension()),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self.stov[token])

        return output

    def __call__(self, item):
        self._try_download()
        return super().__call__(item)


@registry.register_processor("multi_hot_answer_from_vocab")
class MultiHotAnswerFromVocabProcessor(VQAAnswerProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def compute_answers_scores(self, answers_indices):
        scores = torch.zeros(self.get_vocab_size(), dtype=torch.float)
        scores[answers_indices] = 1
        scores[self.answer_vocab.UNK_INDEX] = 0
        return scores


@registry.register_processor("soft_copy_answer")
class SoftCopyAnswerProcessor(VQAAnswerProcessor):
    """Similar to Answer Processor but adds soft copy dynamic answer space to it.
    Read https://arxiv.org/abs/1904.08920 for extra information on soft copy
    and LoRRA.

    Args:
        config (Configuration): Configuration for soft copy processor.

    """

    DEFAULT_MAX_LENGTH = 50

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            self.max_length = self.DEFAULT_MAX_LENGTH
            warnings.warn(
                "'max_length' not defined in the config. Setting to default of {}".format(
                    self.DEFAULT_MAX_LENGTH
                )
            )

        self.context_preprocessor = None
        if hasattr(config, "context_preprocessor"):
            self.context_preprocessor = Processor(config.context_preprocessor)

    def get_vocab_size(self):
        """Size of Vocab + Size of Dynamic soft-copy based answer space

        Returns:
            int: Size of vocab + size of dynamic soft-copy answer space.

        """
        answer_vocab_nums = self.answer_vocab.num_vocab
        answer_vocab_nums += self.max_length

        return answer_vocab_nums

    def get_true_vocab_size(self):
        """Actual vocab size which only include size of the vocabulary file.

        Returns:
            int: Actual size of vocabs.

        """
        return self.answer_vocab.num_vocab

    def __call__(self, item):
        answers = item["answers"]
        scores = super().__call__({"answers": answers})

        indices = scores["answers_indices"]
        answers = scores["answers"]
        scores = scores["answers_scores"]

        tokens_scores = scores.new_zeros(self.max_length)
        tokens = item["tokens"]
        length = min(len(tokens), self.max_length)

        gt_answers = list(enumerate(answers))

        if self.context_preprocessor is not None:
            tokens = [
                self.context_preprocessor({"text": token})["text"] for token in tokens
            ]

        answer_counter = Counter(answers)

        for idx, token in enumerate(tokens[:length]):
            if answer_counter[token] == 0:
                continue
            accs = []

            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == token]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)

            tokens_scores[idx] = sum(accs) / len(accs)

        # Scores are already proper size, see L314. Now,
        # fix scores for soft copy candidates
        scores[-len(tokens_scores) :] = tokens_scores  # noqa
        return {
            "answers": answers,
            "answers_indices": indices,
            "answers_scores": scores,
        }


@registry.register_processor("simple_word")
class SimpleWordProcessor(BaseProcessor):
    """Tokenizes a word and processes it.

    Attributes:
        tokenizer (function): Type of tokenizer to be used.

    """

    def __init__(self, *args, **kwargs):
        from antmmf.utils.text_utils import word_tokenize

        self.tokenizer = word_tokenize

    def __call__(self, item, *args, **kwargs):
        return {"text": self.tokenizer(item["text"], *args, **kwargs)}


@registry.register_processor("simple_sentence")
class SimpleSentenceProcessor(BaseProcessor):
    """Tokenizes a sentence and processes it.

    Attributes:
        tokenizer (function): Type of tokenizer to be used.

    """

    def __init__(self, *args, **kwargs):
        from antmmf.utils.text_utils import tokenize

        self.tokenizer = tokenize

    def __call__(self, item, *args, **kwargs):
        tokens = self.tokenizer(item["text"], *args, **kwargs)
        return {"text": tokens}


@registry.register_processor("caption")
class CaptionProcessor(BaseProcessor):
    """Processes a caption with start, end and pad tokens and returns raw string.

    Args:
        config (Configuration): Configuration for caption processor.

    """

    def __init__(self, config, *args, **kwargs):
        if not hasattr(config, "vocab"):
            raise AttributeError(
                "config passed to the processor has no attribute vocab"
            )
        super(CaptionProcessor, self).__init__(config)
        self.vocab = Vocab(*args, **config.vocab, **kwargs)

    def __call__(self, item, *args, **kwargs):
        item = keep_till_eos(item)
        tokens = [
            self.vocab.get_itos()[w]
            for w in item
            if w
            not in {self.vocab.SOS_INDEX, self.vocab.EOS_INDEX, self.vocab.PAD_INDEX}
        ]
        caption = " ".join(tokens)
        return {"tokens": tokens, "caption": caption}


@registry.register_processor("masked_token")
class MaskedTokenProcessor(BaseProcessor):
    """
    support loading offline bert model, ${PYTORCH_TRANSFORMERS_CACHE} indicates absolute path to bert_model_dir
    bert_model_dir should have this hier structure:
    |____bert_model_dir
    | |____bert-base-uncased # bert_model_name
    | | |____config.json
    | | |____pytorch_model.bin
    | | |____vocab.txt

    """

    _CLS_TOKEN = CLS_TOKEN_STR
    _SEP_TOKEN = SEP_TOKEN_STR
    _MASK_TOKEN = "[MASK]"
    _PAD_TOKEN_ID = 0

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        tokenizer_config = config.tokenizer_config
        from antmmf.datasets.build import build_tokenizer

        self.config = config
        self._tokenizer = build_tokenizer(tokenizer_config)

        self._max_seq_length = (
            config.max_length if "max_length" in config else config.max_seq_length
        )
        assert self._max_seq_length is not None, "max_seq_length is not set in config"
        self._probability = getattr(config, "mask_probability", 0.15)
        self._trim_start_token = getattr(config, "trim_start_token", False)
        self._random_mask_chinese = config.get("random_mask_chinese", False)

        # select a random sliding window of the text sequence if
        # tokens are too long. Will not take effect for clip more
        # than ONE sentences at the same time.
        self._random_truncate = config.get("random_truncate", False)

        # perform Whole Word Masking:
        # https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L350-L358
        self._wwm = config.get("whole_word_masking", False)

        # intra_MLM for SNP-S3
        if config.get("intra_VTM", False) and config.intra_VTM.get("IW_MLM", False):
            print('\nMLM IW V3!\n')
            path = config.intra_VTM.HT_words_count_file_dir
            with open(path, 'r') as reader:
                HT_count_data = json.load(reader)

            self.word_rank_info = HT_count_data['rank']
            self.words_top_k = config.intra_VTM.words_top_k
            important_tokens = []
            for i in range(len(self.word_rank_info)):
                if self.word_rank_info[i] <= self.words_top_k:
                    important_tokens.append(i)
            self.important_words = important_tokens
            # 获得同词根的vocab列表
            same_lema_word_dir = config.intra_VTM.vocab_same_lema_dir
            with open(same_lema_word_dir, 'r') as reader:
                self.same_lema_list = json.load(reader)

    def get_vocab_size(self):
        return len(self._tokenizer)

    '''intra_MLM for SNP-S3'''
    def get_important_tag_list(self, tokens):
        IW_word_tag = []
        IW_word_lists = []
        other_token_idx_list = []

        for i in range(len(tokens)):
            id_raw = self._tokenizer.convert_tokens_to_ids(tokens[i])
            if self.word_rank_info[id_raw] <= self.words_top_k:
                IW_word_lists.append(i)
            else:
                other_token_idx_list.append(i)
            IW_word_tag.append(0)
        return IW_word_tag, IW_word_lists, other_token_idx_list, tokens

    def get_one_from_same_vocab(self, root_idx):
        root_list = self.same_lema_list[root_idx]
        if len(root_list) == 0:
            return self._tokenizer.convert_ids_to_tokens(root_idx)[0]
        else:
            root_rand_idx = random.randint(0, len(root_list) - 1)
            return root_list[root_rand_idx]

    def _random_word(self, tokens, probability=0.15):
        if self.config.get("intra_VTM", False) and self.config.intra_VTM.get("IW_MLM", False):
            return self._random_word_IW_MLM(tokens, probability=probability)
        else:
            return self._random_word_raw(tokens, probability=probability)

    def _random_word_raw(self, tokens, probability=0.15):
        labels = []
        from antmmf.utils.text_utils import is_chinese

        for idx, token in enumerate(tokens):
            prob = random.random()
            if self._random_mask_chinese and not is_chinese(token):
                # not make char as target other than chinese
                labels.append(-1)
            elif prob < probability:
                prob /= probability

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[idx] = self._MASK_TOKEN
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[idx] = self._tokenizer.convert_ids_to_tokens(
                        torch.randint(len(self._tokenizer), (1,), dtype=torch.long)
                    )[0]

                # rest 10% keep the original token as it is

                labels.append(self._tokenizer.convert_tokens_to_ids(token))
            else:
                labels.append(-1)

        return tokens, labels

    def _random_word_IW_MLM(self, tokens, probability=0.15):
        labels = []
        for i in range(len(tokens)):
            labels.append(-1)

        '''get masked words'''
        IW_word_tag, IW_word_lists, other_id_lists, tokens = self.get_important_tag_list(tokens)
        chosen_num = int(len(tokens) * probability)
        float_num = len(tokens) * probability - chosen_num

        if float_num >= 0.3:
            chosen_num += 1
        if chosen_num > len(IW_word_lists):
            chosen_idx_list = []
            for i in range(len(IW_word_lists)):
                chosen_idx_list.append(i)
            rest_num = chosen_num - len(IW_word_lists)
            tt_rand_list = random.sample(range(0, len(other_id_lists)), rest_num)
            for trl in tt_rand_list:
                tokens[trl] = self._MASK_TOKEN
                labels[trl] = self._tokenizer.convert_tokens_to_ids(tokens[trl])
        else:
            chosen_idx_list = random.sample(range(0, len(IW_word_lists)), chosen_num)
        for cil in chosen_idx_list:
            IW_word_tag[IW_word_lists[cil]] = 1

        #get masked tokens and labels
        for idx, token in enumerate(tokens):
            # if self._random_mask_chinese and not is_chinese(token):
            #     # not make char as target other than chinese
            #     labels.append(-1)
            if IW_word_tag[idx] == 1:
                new_prob = random.random()

                # 80% randomly change token to mask token
                if new_prob < 0.8:
                    tokens[idx] = self._MASK_TOKEN
                # 10% randomly change token to random token
                elif new_prob < 0.9:
                    new_idx = random.randint(0, len(self.important_words)-1)
                    root_idx = self.important_words[new_idx]
                    tokens[idx] = self.get_one_from_same_vocab(root_idx)

                # rest 10% keep the original token as it is

                labels[idx] = self._tokenizer.convert_tokens_to_ids(token)
            # else:
            #     labels.append(-1)

        return tokens, labels
    '''end intra_MLM for SNP-S3'''

    def _truncate_tokens(self, tokens, max_length, random_truncate=True):
        total = len(tokens)
        if random_truncate:
            # Refer to
            # LayoutLMv2: https://arxiv.org/abs/2012.14740
            # section 3.2 pretraining settings.
            truncate_point = max(total - max_length, 0)
            s = random.randint(0, truncate_point)
            e = s + max_length
            tokens = tokens[s:e]
        else:
            truncate_point = min(max_length, total)
            tokens = tokens[:truncate_point]

        return tokens

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer
        # sequence.
        if tokens_b is None:
            tokens_b = []
            tokens_a = self._truncate_tokens(
                tokens_a, max_length, random_truncate=self._random_truncate
            )
        else:
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return tokens_a, tokens_b

    def _whole_word_masking(self, tokens, labels):
        wwm_tokens, wwm_labels = tokens[:], labels[:]
        txt_len = len(tokens)
        wwm_cand = []
        for t_id in range(txt_len):
            if t_id == 0:
                continue
            if tokens[t_id].startswith("##"):
                back_trace = t_id - 1
                while back_trace >= 0 and tokens[back_trace].startswith("##"):
                    back_trace -= 1
                if back_trace >= 0 and labels[back_trace] != -1:
                    wwm_cand.append(t_id)
        for t_id in wwm_cand:
            wwm_labels[t_id] = self._tokenizer.convert_tokens_to_ids(tokens[t_id])
            wwm_tokens[t_id] = self._MASK_TOKEN
        return wwm_tokens, wwm_labels

    def _convert_to_indices(self, tokens_a, tokens_b=None, probability=0.15):
        tokens_a, label_a = self._random_word(tokens_a, probability=probability)

        if self._wwm:
            tokens_a, label_a = self._whole_word_masking(tokens_a, label_a)

        if self._trim_start_token:
            tokens = []
            segment_ids = []
            lm_label_ids = []
        else:
            tokens = [self._CLS_TOKEN]
            segment_ids = [0]
            lm_label_ids = [-1]

        tokens += tokens_a
        segment_ids += [0] * len(tokens_a)

        tokens.append(self._SEP_TOKEN)
        segment_ids.append(0)

        if tokens_b:
            tokens_b, label_b = self._random_word(tokens_b, probability=probability)
            lm_label_ids += label_a + [-1] + label_b + [-1]
            assert len(tokens_b) > 0
            tokens += tokens_b
            segment_ids += [1] * len(tokens_b)
            tokens.append(self._SEP_TOKEN)
            segment_ids.append(1)
        else:
            lm_label_ids += label_a + [-1]

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        source_len = len(input_ids)
        input_mask = [1] * source_len

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(self._PAD_TOKEN_ID)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == self._max_seq_length
        assert len(input_mask) == self._max_seq_length
        assert len(segment_ids) == self._max_seq_length
        assert len(lm_label_ids) == self._max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "lm_label_ids": lm_label_ids,
            "tokens": tokens,
            "source_len": source_len,
        }

    def __call__(self, item, probability=None):
        text_a = item["text_a"] if "text_a" in item else item["text"]
        text_b = item.get("text_b", None)

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if text_b:
            tokens_b = self._tokenizer.tokenize(text_b)

        if self._trim_start_token:
            content_len = self._max_seq_length - 1
        else:
            content_len = self._max_seq_length - 2
        tokens_a, tokens_b = self._truncate_seq_pair(tokens_a, tokens_b, content_len)
        # Enable dynamically changing probability,
        # For supporting both MLM task and Generation task with the same processor.
        prob = probability if probability is not None else self._probability
        output = self._convert_to_indices(tokens_a, tokens_b, probability=prob)
        if "is_correct" in item:
            output["is_correct"] = torch.tensor(item["is_correct"], dtype=torch.long)

        return output


@registry.register_processor("masked_layoutlm_tokenizer")
class MaskedLayoutlmTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def __call__(self, item):
        if "text" in item:
            text = item["text"]
        else:
            text = " ".join(item["tokens"])

        assert len(text) == len(item["bbox"])
        boxes = []
        tokens = []
        if len(text) == 0 or (
            len(text) == 1 and text[0] == ""
        ):  # handle case text=['']
            boxes = [[0, 0, 0, 0]]  # add pad_token for empty input
            tokens = [""]

        for token, bbox in zip(text, item["bbox"]):
            # Preprocess OCR tokens
            # preprocessor is None if not configured
            if getattr(self, "preprocessor", None):
                # ocr token need to remove [?,] before token
                token = self.preprocessor({"text": token})["text"]
            sub_tokens = self._tokenizer.tokenize(token)  # do subtoken
            for sub_token in sub_tokens:  # sub_token share same bbox
                boxes.append(bbox)
                tokens.append(sub_token)

        if self._trim_start_token:
            content_len = self._max_seq_length - 1
        else:
            content_len = self._max_seq_length - 2
        tokens, boxes = self._truncate_seq_pair(tokens, boxes, content_len)
        output = self._convert_to_indices(tokens, boxes, probability=self._probability)
        output[TEXT_MODALITY] = output["tokens"]
        output[CLS_ID_STR] = self._tokenizer.cls_token_id
        output[SEP_ID_STR] = self._tokenizer.sep_token_id

        return output

    def _truncate_seq_pair(self, tokens, bboxes, max_length):
        assert len(tokens) == len(bboxes)

        total = len(tokens)
        if self._random_truncate:
            # Refer to
            # LayoutLMv2: https://arxiv.org/abs/2012.14740
            # section 3.2 pretraining settings.
            truncate_point = max(total - max_length, 0)
            s = random.randint(0, truncate_point)
            e = s + max_length
            tokens = tokens[s:e]
            bboxes = bboxes[s:e]
        else:
            truncate_point = min(max_length, total)
            tokens = tokens[:truncate_point]
            bboxes = bboxes[:truncate_point]

        return tokens, bboxes

    def _convert_to_indices(self, tokens, bbox, probability=0.15):
        tokens_a, label_a = self._random_word(tokens, probability=probability)
        if self._trim_start_token:
            tokens = []
            bboxes = []
            segment_ids = []
            lm_label_ids = []
        else:
            tokens = [self._CLS_TOKEN]
            bboxes = [[0, 0, 1000, 1000]]
            segment_ids = [0]
            lm_label_ids = [-1]

        tokens += tokens_a
        segment_ids += [0] * len(tokens_a)
        bboxes += bbox

        tokens.append(self._SEP_TOKEN)
        segment_ids.append(0)
        bboxes.append([1000, 1000, 1000, 1000])

        lm_label_ids += label_a + [-1]

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        source_len = len(input_ids)
        input_mask = [1] * source_len

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)
            bboxes.append([0, 0, 0, 0])

        assert len(input_ids) == self._max_seq_length
        assert len(input_mask) == self._max_seq_length
        assert len(segment_ids) == self._max_seq_length
        assert len(lm_label_ids) == self._max_seq_length
        assert len(bboxes) == self._max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
        bboxes = torch.tensor(bboxes, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "lm_label_ids": lm_label_ids,
            "tokens": tokens,
            "source_len": source_len,
            "bboxes": bboxes,
        }


@registry.register_processor("masked_bert_tokenizer")
class MaskedBertTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = getattr(config, "mask_probability", 0)
        self._trim_start_token = getattr(config, "trim_start_token", False)

    def __call__(self, item, probability=None):
        if "text" in item:
            text_a = item["text"]
        else:
            text_a = " ".join(item["tokens"])

        tokens_a = self._tokenizer.tokenize(text_a)

        if self._trim_start_token:
            content_len = self._max_seq_length - 1
        else:
            content_len = self._max_seq_length - 2
        tokens_a, tokens_b = self._truncate_seq_pair(tokens_a, None, content_len)

        # Enable dynamically changing probability,
        # For supporting both MLM task and Generation task with the same processor.
        prob = probability if probability is not None else self._probability
        output = self._convert_to_indices(tokens_a, None, probability=prob)
        output[TEXT_MODALITY] = output["tokens"]
        output[CLS_ID_STR] = self._tokenizer.cls_token_id
        output[SEP_ID_STR] = self._tokenizer.sep_token_id
        output[LM_LABEL_IDS_STR] = output["lm_label_ids"]
        return output

    def tokenizer(self):
        return self._tokenizer


@registry.register_processor("masked_roberta_tokenizer")
class MaskedRobertaTokenizer(MaskedTokenProcessor):
    _CLS_TOKEN = "<s>"
    _SEP_TOKEN = "</s>"
    _MASK_TOKEN = "<mask>"
    _PAD_TOKEN_ID = 1  # roberta's pad_token_id == 1

    def __init__(self, config, *args, **kwargs):
        # https://huggingface.co/transformers/model_doc/xlmroberta.html
        # roberta is with different tokenization of above default (bert)
        super().__init__(config, *args, **kwargs)
        assert self._CLS_TOKEN == self._tokenizer.bos_token  # <s>
        assert self._SEP_TOKEN == self._tokenizer.sep_token  # </s>
        assert self._MASK_TOKEN == self._tokenizer.mask_token  # <mask>
        assert (
            self._PAD_TOKEN_ID == self._tokenizer.pad_token_id == 1
        )  # 1, roberta's pad_token_id == 1


@registry.register_processor("phoc")
class PhocProcessor(VocabProcessor):
    """
    Compute PHOC features from text tokens
    """

    def __init__(self, config, *args, **kwargs):
        self._init_extras(config)
        self.config = config

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        phoc_dim = 604
        output = torch.full(
            (self.max_length, phoc_dim),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(build_phoc(token))

        return output


@registry.register_processor("bert_tokenizer")
class BertTokenizerProcessor(MaskedBertTokenizer):
    """
    Tokenize a text string with BERT tokenizer
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = 0.0


@registry.register_processor("roberta_tokenizer")
class RoBERTaTokenizer(MaskedRobertaTokenizer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = 0


@registry.register_processor("clip_tokenizer")
class CLIPTokenizerProcessor(BaseProcessor):
    """
    Tokenize a text string with CLIP BERT tokenizer

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]

    """

    def __init__(self, config, *args, **kwargs):
        self.context_length = config.max_seq_length
        self.truncate = config.truncate
        from antmmf.modules.vision.backbone.clip.simple_tokenizer import SimpleTokenizer

        self._tokenizer = SimpleTokenizer()

    def __call__(self, item):
        text = item[TEXT_MODALITY]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)

        if len(tokens) > self.context_length:
            if self.truncate:
                tokens = tokens[: self.context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {text} is too long for context length {self.context_length}"
                )

        result[: len(tokens)] = torch.tensor(tokens)

        results = {"input_ids": result, TEXT_MODALITY: item[TEXT_MODALITY]}

        return results


@registry.register_processor("cn_clip_tokenizer")
class CNCLIPTokenizerProcessor(BaseProcessor):
    """
    Tokenize a text string with Chinese CLIP BERT tokenizer
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 52 as the context length
    Returns
    -------
    A one-dimensional tensor containing the resulting tokens, shape = [context_length]
    """

    def __init__(self, config, context_length=52, *args, **kwargs):
        self.context_length = context_length
        from antmmf.modules.vision.backbone.clip.cn_tokenizer import FullTokenizer

        self._tokenizer = FullTokenizer()

    def __call__(self, item):
        text = item[TEXT_MODALITY]
        tokens = (
            [self._tokenizer.vocab["[CLS]"]]
            + self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))[
                : self.context_length - 2
            ]
            + [self._tokenizer.vocab["[SEP]"]]
        )
        result = torch.zeros(self.context_length, dtype=torch.long)
        if len(tokens) > self.context_length:
            tokens = tokens[: self.context_length]
            tokens[-1] = self._tokenizer.vocab["[SEP]"]
        result[: len(tokens)] = torch.tensor(tokens)
        results = {"input_ids": result, TEXT_MODALITY: item[TEXT_MODALITY]}
        return results
