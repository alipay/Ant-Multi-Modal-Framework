import numpy as np


_alphabet = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
}  # NoQA
_unigrams = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]
_bigrams = [
    "th",
    "he",
    "in",
    "er",
    "an",
    "re",
    "es",
    "on",
    "st",
    "nt",
    "en",
    "at",
    "ed",
    "nd",
    "to",
    "or",
    "ea",
    "ti",
    "ar",
    "te",
    "ng",
    "al",
    "it",
    "as",
    "is",
    "ha",
    "et",
    "se",
    "ou",
    "of",
    "le",
    "sa",
    "ve",
    "ro",
    "ra",
    "ri",
    "hi",
    "ne",
    "me",
    "de",
    "co",
    "ta",
    "ec",
    "si",
    "ll",
    "so",
    "na",
    "li",
    "la",
    "el",
]


def build_phoc(token):
    token = token.lower().strip()
    word = [c for c in token if c in _alphabet]

    phoc = np.zeros(604, dtype=np.float32)

    n = len(word)
    for index in range(n):

        char_occ0 = index / (n + 0.0)
        char_occ1 = (index + 1.0) / n

        char_index = -1
        for k in range(36):
            if _unigrams[k] == word[index]:
                char_index = k
                break
        assert char_index != -1, r"Error: unigram {} is unknown".format(word[index])

        # check unigram levels
        for level in range(2, 6):
            for region in range(level):
                region_occ0 = region / (level + 0.0)
                region_occ1 = (region + 1.0) / level
                overlap0 = max(char_occ0, region_occ0)
                overlap1 = min(char_occ1, region_occ1)
                kkk = (overlap1 - overlap0) / (char_occ1 - char_occ0 + 0.0)
                if kkk >= 0.5:
                    sum = 0
                    for l in range(2, 6):
                        if l < level:
                            sum += l
                    feat_vec_index = int(sum * 36 + region * 36 + char_index)
                    phoc[feat_vec_index] = 1

    # add _bigrams
    ngram_offset = 36 * 14
    for i in range(n - 1):
        ngram_index = -1
        for k in range(50):
            if _bigrams[k] == "".join(word[i : i + 2]):
                ngram_index = k
                break

        if ngram_index == -1:
            continue

        ngram_occ0 = i / (0.0 + n)
        ngram_occ1 = (i + 2.0) / n
        level = 2
        for region in range(level):
            region_occ0 = region / (level + 0.0)
            region_occ1 = (region + 1.0) / level
            overlap0 = max(ngram_occ0, region_occ0)
            overlap1 = min(ngram_occ1, region_occ1)
            if (overlap1 - overlap0) / (ngram_occ1 - ngram_occ0) >= 0.5:
                phoc[int(ngram_offset + region * 50 + ngram_index)] = 1

    return phoc
