# Copyright (c) 2023 Ant Group and its affiliates.
"""
Text utils module contains implementations for various decoding strategies like
Greedy, Beam Search and Nucleus Sampling.

In your model's config you can specify ``inference`` attribute to use these strategies
in the following way:

.. code::

   model_attributes:
       some_model:
           inference:
               - type: greedy
               - params: {}
"""
import os
import sys
import re
from collections import Counter
from itertools import chain

import torch

from antmmf.common.registry import registry
from antmmf.utils.general import get_absolute_path

SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3


"""
refer to:
https://github.com/huggingface/transformers/blob/master/examples/\
research_projects/mlm_wwm/run_chinese_ref.py#L9
"""


def str_q2b(sentence):
    """
    全角转半角
    全角字符unicode编码从65281~65374 （十六进制 0xFF01 ~ 0xFF5E）
    半角字符unicode编码从33~126 （十六进制 0x21~ 0x7E）
    除空格外,全角/半角按unicode编码排序在顺序上是对应的
    """
    rstring = ""
    for uchar in sentence:
        inside_code = ord(uchar)
        # 空格 全角为0x3000 半角为0x0020
        if inside_code == 0x3000:
            inside_code = 0x0020
        # 中文句号
        elif inside_code == 0x3002:
            inside_code = 0x2E
        else:
            inside_code -= 0xFEE0
        if inside_code < 0x0020 or inside_code > 0x7E:
            rstring += uchar
        else:
            rstring += chr(inside_code)
    return rstring


def str_decode(sentence):
    """
    :param sentence: a sentence
    :return: a sentence has been decode into unicode type
    """
    PY_VERSION = sys.version_info[0]
    if PY_VERSION == 2:
        TEXT_TYPE = unicode
    else:
        TEXT_TYPE = str

    if not isinstance(sentence, TEXT_TYPE):
        try:
            sentence = sentence.decode("utf-8")
        except UnicodeDecodeError:
            sentence = sentence.decode("gbk", "ignore")
    return sentence


def replace_blank_with(sentence, repl):
    for i in range(100, 1, -1):
        sentence = sentence.replace(" " * i, repl)
    return sentence


def replace_simple_entity_with(sentence, repl):
    """
    replace special entities in sentence with repl,
    such as url, ip, email, phone number, html, time, date, jpg...
    """
    re_url = r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)"
    re_ip = r"((?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d))"
    re_email = r"([a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+[\.[a-zA-Z0-9_-]+]?)"
    re_phone = r"(1[3|4|5|6|7|8|9]\d{9})"
    re_html_tag = r'<[a-z0-9"/= :#;\-]*?>'
    re_html_escape = r"(&[a-zA-Z0-9]*;)"
    re_en_time = r"(\d{1,2}:\d{1,2})"
    re_en_date = r"(\d{4}-\d{1,2}-\d{1,2})"
    re_jpg = r"([-|a-zA-Z0-9]*.jpg)"
    re_kb = r"([\d]*[\.]?[[\d]+]?KB)"
    re_all = "|".join(
        [
            re_url,
            re_ip,
            re_email,
            re_html_tag,
            re_html_escape,
            re_phone,
            re_en_time,
            re_en_date,
            re_jpg,
            re_kb,
        ]
    )
    return re.sub(re_all, repl, sentence)


def replace_digit(sentence):
    re_digit = r"([\d]*[\.]?[[\d]+]?[%]?)"
    return re.sub(re_digit, "1", sentence)


def replace_time(sentence):
    re_zh_time = (
        r"(\d{1,2}[点|时]\d{1,2}分\d{1,2}秒|\d{1,2}[点|时]\d{1,2}分|\d{1,2}[点|时]|\d{1,2}分)"
    )
    re_zh_date = r"(\d{1,4}年\d{1,2}月\d{1,2}日|\d{1,4}年\d{1,2}月[份]?|\d{1,4}年|\d{1,2}月\d{1,2}日|\d{1,2}月[份]?)"
    re_all = "|".join([re_zh_time, re_zh_date])
    return re.sub(re_all, "T", sentence)


def replace_luanma_with(sentence, repl):
    # luanma = [u"◑",u"[[+_+]]",u"...",u"———",u"…",u"■",u"◆",u"→",u"▼",u"$$",u"★",u"</p>",u"<p>",u"<br>"]
    luanma = [
        "\n",
        "◑",
        "[[+_+]]",
        "...",
        "———",
        "…",
        "→",
        "▼",
        "$$",
        "★",
        "</p>",
        "<p>",
        "<br>",
        "<strong>",
        "</strong>",
    ]
    for m in luanma:
        sentence = sentence.replace(m, repl)
    return sentence


def to_lowercase_english(sentence):
    sentence = list(sentence)
    for i in range(len(sentence)):
        if sentence[i].isupper():
            sentence[i] = sentence[i].lower()
    return "".join(sentence)


def del_low_freq_char(sentence):
    new_sentence = list()
    # 删掉不是汉字、数字、英文、高频字符
    for char in sentence:
        if (
            ("\u4e00" <= char <= "\u9fff")
            or ("\u0030" <= char <= "\u0039")
            or ("\u0041" <= char <= "\u005a" or "\u0061" <= char <= "\u007a")
            or (char in "+, 、 :(.)%-《》;][!*?")
        ):
            new_sentence.append(char)
        else:
            if char in "“”【】":
                new_sentence.append({"“": '"', "”": '"', "【": "[", "】": "]"}[char])
    return "".join(new_sentence)


def not_hanzi_digit_english(char):
    if (
        ("\u4e00" <= char <= "\u9fff")
        or ("\u0030" <= char <= "\u0039")
        or ("\u0041" <= char <= "\u005a" or "\u0061" <= char <= "\u007a")
    ):
        return False
    else:
        return True


def entity_process(entity):
    # 实体开头结尾的符号类字符全部删除
    while PreProcessor.not_hanzi_digit_english(entity[0]):
        entity = entity[1:]
    while PreProcessor.not_hanzi_digit_english(entity[-1]):
        entity = entity[:-1]
    return entity


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word: str):
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return 0
    return 1


def generate_ngrams(tokens, n=1):
    """Generate ngrams for particular 'n' from a list of tokens
    Parameters
    ----------
    tokens : List[str]
        List of tokens for which the ngram are to be generated
    n : int
        n for which ngrams are to be generated
    Returns
    -------
    List[str]
        List of ngrams generated
    """
    shifted_tokens = (tokens[i:] for i in range(n))
    tuple_ngrams = zip(*shifted_tokens)
    return (" ".join(i) for i in tuple_ngrams)


def generate_ngrams_range(tokens, ngram_range=(1, 3)):
    """Generates and returns a list of ngrams for all n present in ngram_range.
    Parameters
    ----------
    tokens : List[str]
        List of string tokens for which ngram are to be generated
    ngram_range : List[int]
        List of 'n' for which ngrams are to be generated. For e.g. if
        ngram_range = (1, 4) then it will returns 1grams, 2grams and 3grams
    Returns
    -------
    List[str]
        List of ngrams for each n in ngram_range.
    """
    assert (
        len(ngram_range) == 2
    ), "'ngram_range' should be a tuple of two elements which is range of numbers"
    return chain(*(generate_ngrams(tokens, i) for i in range(*ngram_range)))


def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX, keep=["'s"], remove=[",", "?"]):
    sentence = sentence.lower()
    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def word_tokenize(word, remove=[",", "?"]):
    word = word.lower()

    for item in remove:
        word = word.replace(item, "")
    word = word.replace("'s", " 's")

    return word.strip()


def load_str_list(fname):
    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


def keep_till_eos(item):
    for idx, v in enumerate(item):
        if v == EOS_INDEX:
            item = item[:idx]
            break
    return item


class VocabDict:
    def __init__(self, vocab_file, data_root_dir=None):
        if not os.path.isabs(vocab_file) and data_root_dir is not None:
            vocab_file = get_absolute_path("{}/{}".format(data_root_dir, vocab_file))

        if not os.path.exists(vocab_file):
            raise RuntimeError(
                "Vocab file {} for vocab dict doesn't exist".format(vocab_file)
            )

        self.word_list = load_str_list(vocab_file)
        self._build()

    def _build(self):
        if UNK_TOKEN not in self.word_list:
            self.word_list = [UNK_TOKEN] + self.word_list

        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}

        # String (word) to integer (index) dict mapping
        self.stoi = self.word2idx_dict
        # Integer to string (word) reverse mapping
        self.itos = self.word_list
        self.num_vocab = len(self.word_list)

        self.UNK_INDEX = (
            self.word2idx_dict[UNK_TOKEN] if UNK_TOKEN in self.word2idx_dict else None
        )

        self.PAD_INDEX = (
            self.word2idx_dict[PAD_TOKEN] if PAD_TOKEN in self.word2idx_dict else None
        )

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def __len__(self):
        return len(self.word_list)

    def get_size(self):
        return len(self.word_list)

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return UNK_TOKEN

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_INDEX is not None:
            return self.UNK_INDEX
        else:
            raise ValueError(
                "word %s not in dictionary \
                             (while dictionary does not contain <unk>)"
                % w
            )

    def tokenize_and_index(self, sentence, keep=["'s"]):
        inds = [self.word2idx(w) for w in tokenize(sentence, keep=keep)]
        return inds


class VocabFromText(VocabDict):
    DEFAULT_TOKENS = [
        PAD_TOKEN,
        UNK_TOKEN,
        START_TOKEN,
        END_TOKEN,
    ]

    def __init__(
        self,
        sentences,
        min_count=1,
        regex=SENTENCE_SPLIT_REGEX,
        keep=[],
        remove=[],
        only_unk_extra=False,
    ):
        token_counter = Counter()

        for sentence in sentences:
            tokens = tokenize(sentence, regex=regex, keep=keep, remove=remove)
            token_counter.update(tokens)

        token_list = []
        for token in token_counter:
            if token_counter[token] >= min_count:
                token_list.append(token)

        extras = self.DEFAULT_TOKENS

        if only_unk_extra:
            extras = [UNK_TOKEN]

        self.word_list = extras + token_list
        self._build()


class TextDecoder:
    """Base class to be inherited by all decoding strategies. Contains
    implementations that are common for all strategies.

    Args:
        vocab (list): Collection of all words in vocabulary.

    """

    def __init__(self, vocab):
        self._vocab = vocab
        self._vocab_size = vocab.get_size()

        # Lists to store completed sequences and scores
        self._complete_seqs = []
        self._complete_seqs_scores = []

    def init_batch(self, sample_list):
        self.seqs = sample_list.answers.new_full(
            (self._decode_size, 1), SOS_INDEX, dtype=torch.long
        )

        sample_list.image_feature_0 = (
            sample_list.image_feature_0.unsqueeze(1)
            .expand(-1, self._decode_size, -1, -1)
            .squeeze(0)
        )
        return sample_list

    def add_next_word(self, seqs, prev_word_inds, next_word_inds):
        return torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

    def find_complete_inds(self, next_word_inds):
        incomplete_inds = []
        for ind, next_word in enumerate(next_word_inds):
            if next_word != EOS_INDEX:
                incomplete_inds.append(ind)
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        return complete_inds, incomplete_inds

    def update_data(self, data, prev_word_inds, next_word_inds, incomplete_inds):
        data["texts"] = next_word_inds[incomplete_inds].unsqueeze(1)
        h1 = data["state"]["td_hidden"][0][prev_word_inds[incomplete_inds]]
        c1 = data["state"]["td_hidden"][1][prev_word_inds[incomplete_inds]]
        h2 = data["state"]["lm_hidden"][0][prev_word_inds[incomplete_inds]]
        c2 = data["state"]["lm_hidden"][1][prev_word_inds[incomplete_inds]]
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        return data


@registry.register_decoder("beam_search")
class BeamSearch(TextDecoder):
    def __init__(self, vocab, config):
        super(BeamSearch, self).__init__(vocab)
        self._decode_size = config["inference"]["params"]["beam_length"]

    def init_batch(self, sample_list):
        setattr(
            self,
            "top_k_scores",
            sample_list.answers.new_zeros((self._decode_size, 1), dtype=torch.float),
        )
        return super().init_batch(sample_list)

    def decode(self, t, data, scores):
        # Add predicted scores to top_k_scores
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        scores = self.top_k_scores.expand_as(scores) + scores

        # Find next top k scores and words. We flatten the scores tensor here
        # and get the top_k_scores and their indices top_k_words
        if t == 0:
            self.top_k_scores, top_k_words = scores[0].topk(
                self._decode_size, 0, True, True
            )
        else:
            self.top_k_scores, top_k_words = scores.view(-1).topk(
                self._decode_size, 0, True, True
            )

        # Convert to vocab indices. top_k_words contain indices from a flattened
        # k x vocab_size tensor. To get prev_word_indices we divide top_k_words
        # by vocab_size to determine which index in the beam among k generated
        # the next top_k_word. To get next_word_indices we take top_k_words
        # modulo vocab_size index. For example :
        # vocab_size : 9491
        # top_k_words : [610, 7, 19592, 9529, 292]
        # prev_word_inds : [0, 0, 2, 1, 0]
        # next_word_inds : [610, 7, 610, 38, 292]
        prev_word_inds = top_k_words // self._vocab_size
        next_word_inds = top_k_words % self._vocab_size

        # Add new words to sequences
        self.seqs = self.add_next_word(self.seqs, prev_word_inds, next_word_inds)

        # Find completed sequences
        complete_inds, incomplete_inds = self.find_complete_inds(next_word_inds)

        # Add to completed sequences
        if len(complete_inds) > 0:
            self._complete_seqs.extend(self.seqs[complete_inds].tolist())
            self._complete_seqs_scores.extend(self.top_k_scores[complete_inds])

        # Reduce beam length
        self._decode_size -= len(complete_inds)

        # Proceed with incomplete sequences
        if self._decode_size == 0:
            return True, data, 0

        self.seqs = self.seqs[incomplete_inds]
        self.top_k_scores = self.top_k_scores[incomplete_inds].unsqueeze(1)

        # TODO: Make the data update generic for any type of model
        # This is specific to BUTD model only.
        data = self.update_data(data, prev_word_inds, next_word_inds, incomplete_inds)

        next_beam_length = len(prev_word_inds[incomplete_inds])

        return False, data, next_beam_length

    def get_result(self):
        if len(self._complete_seqs_scores) == 0:
            captions = torch.FloatTensor([0] * 5).unsqueeze(0)
        else:
            i = self._complete_seqs_scores.index(max(self._complete_seqs_scores))
            captions = torch.FloatTensor(self._complete_seqs[i]).unsqueeze(0)
        return captions


@registry.register_decoder("nucleus_sampling")
class NucleusSampling(TextDecoder):
    """Nucleus Sampling is a new text decoding strategy that avoids likelihood maximization.
    Rather, it works by sampling from the smallest set of top tokens which have a cumulative
    probability greater than a specified threshold.

    Present text decoding strategies like beam search do not work well on open-ended
    generation tasks (even on strong language models like GPT-2). They tend to repeat text
    a lot and the main reason behind it is that they try to maximize likelihood, which is a
    contrast from human-generated text which has a mix of high and low probability tokens.

    Nucleus Sampling is a stochastic approach and resolves this issue. Moreover, it improves
    upon other stochastic methods like top-k sampling by choosing the right amount of tokens
    to sample from. The overall result is better text generation on the same language model.

    Link to the paper introducing Nucleus Sampling (Section 6) - https://arxiv.org/pdf/1904.09751.pdf

    Args:
        vocab (list): Collection of all words in vocabulary.
        sum_threshold (float): Ceiling of sum of probabilities of tokens to sample from.
    """

    def __init__(self, vocab, config):
        super().__init__(vocab)
        self._decode_size = 1
        # Threshold for sum of probability
        self._threshold = config["inference"]["params"]["sum_threshold"]

    def decode(self, t, data, scores):
        # Convert scores to probabilities
        scores = torch.nn.functional.softmax(scores, dim=1)
        # Sort scores in descending order and then select the top m elements having sum more than threshold.
        # We get the top_m_scores and their indices top_m_words
        if t == 0:
            top_m_scores, top_m_words = scores[0].sort(0, True)
        else:
            top_m_scores, top_m_words = scores.view(-1).sort(0, True)

        last_index = 0
        score_sum = 0
        for score in top_m_scores:
            last_index += 1
            score_sum += score
            if score_sum >= self._threshold:
                break

        top_m_scores = torch.div(top_m_scores[:last_index], score_sum)
        top_m_words = top_m_words[:last_index]

        # Zero value inside prev_word_inds because we are predicting a single stream of output.
        prev_word_ind = torch.tensor([0])
        # Get next word based on probabilities of top m words.
        next_word_ind = top_m_words[torch.multinomial(top_m_scores, 1)]
        # Add next word to sequence

        self.seqs = self.add_next_word(self.seqs, prev_word_ind, next_word_ind)
        # Check if sequence is complete
        complete_inds, incomplete_inds = self.find_complete_inds(next_word_ind)
        # If sequence is complete then return
        if len(complete_inds) > 0:
            self._complete_seqs.extend(self.seqs[complete_inds].tolist())
            return True, data, 0

        self.seqs = self.seqs[incomplete_inds]

        data = self.update_data(data, prev_word_ind, next_word_ind, incomplete_inds)

        return False, data, 1

    def get_result(self):
        if len(self._complete_seqs) == 0:
            captions = torch.FloatTensor([0] * 5).unsqueeze(0)
        else:
            captions = torch.FloatTensor(self._complete_seqs[0]).unsqueeze(0)
        return captions


def bmeso_tag_to_spans(tags, ignore_labels=None):
    """
    给定一个tags的lis，比如['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)
    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()
    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        if tag == "<pad>":
            continue
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ("b", "s"):
            spans.append((label, [idx, idx]))
        elif (
            bmes_tag in ("m", "e")
            and prev_bmes_tag in ("b", "m")
            and label == spans[-1][0]
        ):
            spans[-1][1][1] = idx
        elif bmes_tag == "o":
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [
        (span[0], (span[1][0], span[1][1] + 1))
        for span in spans
        if span[0] not in ignore_labels
    ]


def bioes_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bioes_tag = None
    for idx, tag in enumerate(tags):
        if tag == "<pad>":
            continue
        tag = tag.lower()
        bioes_tag, label = tag[:1], tag[2:]
        if bioes_tag in ("b", "s"):
            spans.append((label, [idx, idx]))
        elif (
            bioes_tag in ("i", "e")
            and prev_bioes_tag in ("b", "i")
            and label == spans[-1][0]
        ):
            spans[-1][1][1] = idx
        elif bioes_tag == "o":
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bioes_tag = bioes_tag
    return [
        (span[0], (span[1][0], span[1][1] + 1))
        for span in spans
        if span[0] not in ignore_labels
    ]


def bio_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        if tag == "<pad>":
            continue
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == "b":
            spans.append((label, [idx, idx]))
        elif bio_tag == "i" and prev_bio_tag in ("b", "i") and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == "o":  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [
        (span[0], (span[1][0], span[1][1] + 1))
        for span in spans
        if span[0] not in ignore_labels
    ]


class BeamSearchNode(object):
    def __init__(
        self, hidden_state, context_vector, previous_node, word_id, log_prob, length
    ):
        self.h = hidden_state
        self.c = context_vector
        self.prev_node = previous_node
        self.word_id = word_id
        self.logp = log_prob
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return (
            self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
        )  # beam-search with punishment

    def __lt__(self, other):
        return self.leng < other.leng  # if score is same, then compare length

    def __gt__(self, other):
        return self.leng > other.leng
