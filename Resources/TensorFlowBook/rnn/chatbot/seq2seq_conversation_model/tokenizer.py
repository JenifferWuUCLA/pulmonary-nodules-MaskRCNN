# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import sys

from tensorflow.python.platform import gfile

from settings import VOCAB_DICT_FILE

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w.decode('utf8') for w in words if w]


# forward maximum matching word segmentation
_DICT = None
_MAX_WORD_LENGTH = 0


def fmm_tokenizer(sentence):
    global _DICT
    global _MAX_WORD_LENGTH
    if not _DICT:
        _DICT, _ = initialize_vocabulary(VOCAB_DICT_FILE)
        for v in _DICT:
            if len(v) > _MAX_WORD_LENGTH:
                _MAX_WORD_LENGTH = len(v)
        print(_MAX_WORD_LENGTH)

    words = []
    begin = 0
    while begin < len(sentence):
        end = min(begin + _MAX_WORD_LENGTH, len(sentence))
        while end > begin + 1:
            word = sentence[begin: end]
            # print (word)
            if word in _DICT:
                break
            end -= 1
        word = sentence[begin: end]
        words.append(word.encode('utf8'))
        begin = end
    return words


def create_vocabulary(vocabulary_path, data_path_patterns, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    vocab = {}
    if gfile.Exists(vocabulary_path):
        sys.stderr.write(
            'vocabulary path %s exsit. we will use the exised one\n' % vocabulary_path)
        return
    for data_f in glob.glob(data_path_patterns):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_f))
        with gfile.GFile(data_f, mode="r") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
    print('total vaca size: %s' % len(vocab))
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
    We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
        vocabulary_path: path to the file containing the vocabulary.

    Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
        ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip().decode('utf8') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)
