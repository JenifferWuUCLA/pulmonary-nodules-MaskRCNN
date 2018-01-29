# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from seq2seq_conversation_model.tokenizer import UNK_ID, _DIGIT_RE
from seq2seq_conversation_model.tokenizer import basic_tokenizer, \
    create_vocabulary, fmm_tokenizer, initialize_vocabulary
from tensorflow.python.platform import gfile


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: a string, the sentence to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    words = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if gfile.Exists(target_path):
        sys.stderr.write(
            'target path %s already exist! we will use the existed one.\n' % target_path)
    else:
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    tokens_file.write(
                        " ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, vocabulary_size, train_file='train',
                 dev_file='dev', use_fmm_tokenizer=False):
    """
    prepare dialog data. all the data should be put in the data_dir
    :param data_dir:
    :param train_file: train file prefix. there should be two train-file prefix files. By default, you should name the enquiry
                       file as train.enquiry, the answer file as train.answer. each line of these two file make a enquiry, answer pair.
    :param dev_file:  almost the same as train file, except  that it is used to internal evaluate
    :param vocabulary_size:
    :return:
    """
    ttokenizer = fmm_tokenizer if use_fmm_tokenizer else basic_tokenizer
    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    create_vocabulary(vocab_path, data_dir + '/test', vocabulary_size,
                      tokenizer=ttokenizer)

    # Create token ids for the training data.
    enquiry_train_ids_path = os.path.join(data_dir, train_file + (
        ".ids%d.enquiry" % vocabulary_size))
    answer_train_ids_path = os.path.join(data_dir, train_file + (
        ".ids%d.answer" % vocabulary_size))
    data_to_token_ids(os.path.join(data_dir, train_file + ".enquiry"),
                      enquiry_train_ids_path, vocab_path, ttokenizer)
    data_to_token_ids(os.path.join(data_dir, train_file + ".answer"),
                      answer_train_ids_path, vocab_path, ttokenizer)

    # Create token ids for the development data.
    enquiry_dev_ids_path = os.path.join(data_dir, dev_file + (
        ".ids%d.enquiry" % vocabulary_size))
    answer_dev_ids_path = os.path.join(data_dir, dev_file + (
        ".ids%d.answer" % vocabulary_size))
    data_to_token_ids(os.path.join(data_dir, dev_file + ".enquiry"),
                      enquiry_dev_ids_path, vocab_path, ttokenizer)
    data_to_token_ids(os.path.join(data_dir, dev_file + ".answer"),
                      answer_dev_ids_path, vocab_path, ttokenizer)

    return (enquiry_train_ids_path, answer_train_ids_path,
            enquiry_dev_ids_path, answer_dev_ids_path,
            vocab_path)
