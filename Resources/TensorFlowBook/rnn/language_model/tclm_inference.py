from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tclm_reader
import tensorflow as tf
from tclm import *


def main(_):
    # $ python tclm_inference.py --save_path=./tc.lm/
    if not FLAGS.save_path:
        raise ValueError("Must set --save_path to language model directory")

    test_data = [tclm_reader.BOF] * 500

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.name_scope("Train"):
            test_input = PTBInput(config=config, data=test_data,
                                  name="TestInput")
            with tf.variable_scope("Model", reuse=None,
                                   initializer=initializer):
                mtest = PTBModel(is_training=True, config=eval_config,
                                 input_=test_input)
            tf.summary.scalar("Training Loss", mtest.cost)
            tf.summary.scalar("Learning Rate", mtest.lr)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                sv.saver.restore(session, ckpt.model_checkpoint_path)
                test_perplexity = run_epoch(session, mtest)
                print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
