# coding=utf8
import logging
import os
import random
import re
import numpy as np
import tensorflow as tf
from seq2seq_conversation_model import seq2seq_model
from seq2seq_conversation_model import data_utils
from seq2seq_conversation_model import tokenizer
from seq2seq_conversation_model.seq2seq_conversation_model import FLAGS, _buckets
from settings import SEQ2SEQ_MODEL_DIR

_LOGGER = logging.getLogger('track')

UNK_TOKEN_REPLACEMENT = [
    '？',
    '我不知道你在说什么',
    '什么鬼。。。',
    '宝宝不知道你在说什么呐。。。',
]


ENGLISHWORD_PATTERN = re.compile(r'[a-zA-Z0-9]')


def is_unichar_englishnum(char):
    return ENGLISHWORD_PATTERN.match(char)


def trim(s):
    """
    1. delete every space between chinese words
    2. suppress extra spaces
    :param s: some python string
    :return: the trimmed string
    """
    if not (isinstance(s, unicode) or isinstance(s, str)):
        return s
    unistr = s.decode('utf8') if type(s) != unicode else s
    unistr = unistr.strip()
    if not unistr:
        return ''
    trimmed_str = []
    if unistr[0] != ' ':
        trimmed_str.append(unistr[0])
    for ind in xrange(1, len(unistr) - 1):
        prev_char = unistr[ind - 1] if len(trimmed_str) == 0 else trimmed_str[-1]
        cur_char = unistr[ind]
        maybe_trim = cur_char == ' '
        next_char = unistr[ind + 1]
        if not maybe_trim:
            trimmed_str.append(cur_char)
        else:
            if is_unichar_englishnum(prev_char) and is_unichar_englishnum(next_char):
                trimmed_str.append(cur_char)
            else:
                continue
    if unistr[-1] != ' ':
        trimmed_str.append(unistr[-1])
    return ''.join(trimmed_str)


class Chatbot():
    """
    answer an enquiry using trained seq2seq model
    """

    def __init__(self, model_dir):
        # Create model and load parameters.
        self.session = tf.InteractiveSession()
        self.model = self.create_model(self.session, model_dir, True)
        self.model.batch_size = 1
        # Load vocabularies.
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        self.vocab, self.rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    def create_model(self, session, model_dir, forward_only):
        """Create conversation model and initialize or load parameters in session."""
        model = seq2seq_model.Seq2SeqModel(
            FLAGS.vocab_size, FLAGS.vocab_size, _buckets,
            FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
            use_lstm=FLAGS.use_lstm,
            forward_only=forward_only)

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            _LOGGER.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
            _LOGGER.info("Read model parameter succeed!")
        else:
            raise ValueError(
                "Failed to find legal model checkpoint files in %s" % model_dir)
        return model

    def generate_answer(self, enquiry):
        # Get token-ids for the input sentence.
        token_ids = data_utils.sentence_to_token_ids(enquiry, self.vocab, tokenizer.fmm_tokenizer)
        if len(token_ids) == 0:
            _LOGGER.error('lens of token ids of sentence %s is 0' % enquiry)
        # Which bucket does it belong to?
        bucket_id = min([b for b in xrange(len(_buckets))
                         if _buckets[b][0] > len(token_ids)])
        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        # Get output logits for the sentence.
        _, _, output_logits = self.model.step(self.session, encoder_inputs,
                                              decoder_inputs,
                                              target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if tokenizer.EOS_ID in outputs:
            outputs = outputs[:outputs.index(tokenizer.EOS_ID)]

        # Print out response sentence corresponding to outputs.
        answer = " ".join([self.rev_vocab[output] for output in outputs])
        if tokenizer._UNK in answer:
            answer = random.choice(UNK_TOKEN_REPLACEMENT)
        answer = trim(answer)
        return answer

    def close(self):
        self.session.close()


if __name__ == "__main__":
    m = Chatbot(SEQ2SEQ_MODEL_DIR + '/train/')
    response = m.generate_answer(u'我知道你不知道我知道你不知道我说的是什么意思')
    print response
