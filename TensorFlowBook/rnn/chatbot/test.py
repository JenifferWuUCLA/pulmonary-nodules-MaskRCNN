# coding=utf8
import logging
import random

import numpy as np
import tensorflow as tf

from seq2seq_conversation_model import seq2seq_model

_LOGGER = logging.getLogger('track')


def test_tokenizer():
    words = fmm_tokenizer(u'嘿，机器人同学，你都会些啥？')
    for w in words:
        print(w)


def test_conversation_model():
    """Test the conversation model."""
    with tf.Session() as sess:
        print("Self-test for neural conversation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                           5.0, 32, 0.3, 0.99, num_samples=8)
        sess.run(tf.initialize_all_variables())
        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])]
                    )
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       bucket_id, False)
        a, b, c = model.step(sess, encoder_inputs, decoder_inputs,
                             target_weights,
                             bucket_id, True)
        print (c)
        c = np.array(c)
        print (c.shape)
        outputs = [np.argmax(logit, axis=1) for logit in c]
        print (outputs)


if __name__ == "__main__":
    # test_tokenizer()
    test_conversation_model()
