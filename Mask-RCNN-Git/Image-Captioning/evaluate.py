#! /usr/bin/python
# -*- coding: utf8 -*-



"""Evaluate the image captioning model.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from buildmodel import *


DIR = "/home/haodong/Workspace/image_captioning"
# MSCOCO_DIR = DIR + "/data/mscoco
MSCOCO_DIR = "/home/haodong/Workspace/image_captioning/data/mscoco"
MODEL_DIR = DIR + "/model"
# Disable GPU
# export CUDA_VISIBLE_DEVICES=""
# Enable 1 GPU
# export CUDA_VISIBLE_DEVICES=1
# File pattern of sharded TFRecord input files.
input_file_pattern = MSCOCO_DIR + "/val-?????-of-00004"
# Directory containing model checkpoints.
checkpoint_dir = MODEL_DIR + "/train"
# Directory to write event logs.
eval_dir = MODEL_DIR + "/eval"
# Interval between evaluation runs, seconds
eval_interval_secs = 600
# Number of examples for evaluation.
num_eval_examples = 10132
# Minimum global step to run evaluation.
min_global_step = 5000

# Whether to train inception submodel variables. If True : Fine Tune the Inception v3 Model
train_inception = False

mode = "eval"
assert mode in ["train", "eval", "inference"]

tf.logging.set_verbosity(tf.logging.INFO) # Enable tf.logging


def evaluate_model(sess, target_cross_entropy_losses, target_cross_entropy_loss_weights, global_step, summary_writer, summary_op):
  """Computes perplexity-per-word over the evaluation dataset.

  Summaries and perplexity-per-word are written out to the eval directory.

  Args:
    sess: Session object.
    model: Instance of ShowAndTellModel; the model to evaluate.
    global_step: Integer; global step of the model checkpoint.
    summary_writer: Instance of SummaryWriter.
    summary_op: Op for generating model summaries.
  """
  # Log model summaries on a single batch.
  summary_str = sess.run(summary_op)
  summary_writer.add_summary(summary_str, global_step)

  # Compute perplexity over the entire dataset.
  num_eval_batches = int(
      math.ceil(num_eval_examples / batch_size))

  start_time = time.time()
  sum_losses = 0.
  sum_weights = 0.
  for i in xrange(num_eval_batches):
    cross_entropy_losses, weights = sess.run([
        target_cross_entropy_losses,
        target_cross_entropy_loss_weights
    ])
    sum_losses += np.sum(cross_entropy_losses * weights)
    sum_weights += np.sum(weights)
    if not i % 100:
      tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                      num_eval_batches)
  eval_time = time.time() - start_time

  perplexity = math.exp(sum_losses / sum_weights)
  tf.logging.info("Perplexity = %f (%.2g sec)", perplexity, eval_time)

  # Log perplexity to the SummaryWriter.
  summary = tf.Summary()
  value = summary.value.add()
  value.simple_value = perplexity
  value.tag = "Perplexity"
  summary_writer.add_summary(summary, global_step)

  # Write the Events file to the eval directory.
  summary_writer.flush()
  tf.logging.info("Finished processing evaluation at global step %d.",
                  global_step)


def run_once(global_step, target_cross_entropy_losses, target_cross_entropy_loss_weights, saver, summary_writer, summary_op):
  """Evaluates the latest model checkpoint.

  Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
    saver: Instance of tf.train.Saver for restoring model Variables.
    summary_writer: Instance of SummaryWriter.
    summary_op: Op for generating model summaries.
  """
  # The lastest ckpt
  model_path = tf.train.latest_checkpoint(checkpoint_dir)
  # print(model_path)   # /home/dsigpu4/Samba/im2txt/model/train_tl/model.ckpt-20000
  # exit()
  if not model_path:
    tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                    checkpoint_dir)
    return

  with tf.Session() as sess:
    # Load model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", model_path)
    saver.restore(sess, model_path)
    # global_step = tf.train.global_step(sess, model.global_step.name)
    step = tf.train.global_step(sess, global_step.name)
    tf.logging.info("Successfully loaded %s at global step = %d.",
                    # os.path.basename(model_path), global_step)
                    os.path.basename(model_path), step)
    # if global_step < min_global_step:
    if step < min_global_step:
    #   tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
      tf.logging.info("Skipping evaluation. Global step = %d < %d", step,
                      min_global_step)
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Run evaluation on the latest checkpoint.
    try:
        evaluate_model(
             sess=sess,
             target_cross_entropy_losses=target_cross_entropy_losses,
             target_cross_entropy_loss_weights=target_cross_entropy_loss_weights,
             global_step=step,
             summary_writer=summary_writer,
             summary_op=summary_op)
    except Exception, e:  # pylint: disable=broad-except
      tf.logging.error("Evaluation failed.")
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  g = tf.Graph()
  with g.as_default():
    images, input_seqs, target_seqs, input_mask = Build_Inputs(mode, input_file_pattern)
    net_image_embeddings = Build_Image_Embeddings(mode, images, train_inception)
    net_seq_embeddings = Build_Seq_Embeddings(input_seqs)
    _, target_cross_entropy_losses, target_cross_entropy_loss_weights, network = \
            Build_Model(mode, net_image_embeddings, net_seq_embeddings, target_seqs, input_mask)

    global_step = tf.Variable(
        initial_value=0,
        dtype=tf.int32,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    # Create the summary operation and the summary writer.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(eval_dir)

    g.finalize()

    # Run a new evaluation run every eval_interval_secs.
    while True:
      start = time.time()
      tf.logging.info("Starting evaluation at " + time.strftime(
          "%Y-%m-%d-%H:%M:%S", time.localtime()))
      run_once(global_step, target_cross_entropy_losses,
                        target_cross_entropy_loss_weights,
                        saver, summary_writer,
                        summary_op)
      time_to_next_eval = start + eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


def main(unused_argv):
  assert input_file_pattern, "--input_file_pattern is required"
  assert checkpoint_dir, "--checkpoint_dir is required"
  assert eval_dir, "--eval_dir is required"
  run()


if __name__ == "__main__":
  tf.app.run()
