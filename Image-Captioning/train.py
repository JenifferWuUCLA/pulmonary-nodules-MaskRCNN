#! /usr/bin/python
# -*- coding: utf8 -*-


"""
TensorLayer implementation of Google's "Show and Tell: A Neural Image Caption Generator".

Before start, you need to download the inception_v3 ckpt model
and MSCOCO data as the following link :
https://github.com/tensorflow/models/tree/master/im2txt

Paper: http://arxiv.org/abs/1411.4555
"""

import tensorflow as tf
import tensorlayer as tl
import time
import numpy as np
from buildmodel import *

DIR = "/home/haodong/Workspace/image_captioning"

## DIR =========================================================================
# Directory containing preprocessed MSCOCO data.
# MSCOCO_DIR = DIR + "/data/mscoco"
MSCOCO_DIR = "/home/haodong/Workspace/image_captioning/data/mscoco"
# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT = DIR + "/data/inception_v3.ckpt"
# Directory to save the model.
MODEL_DIR = DIR + "/model"

# File pattern of sharded TFRecord input files.
input_file_pattern = MSCOCO_DIR + "/train-?????-of-00256"
# Path to a pretrained inception_v3 model. File containing an Inception v3
# checkpoint to initialize the variables of the Inception model. Must be
# provided when starting training for the first time.
inception_checkpoint_file = INCEPTION_CHECKPOINT
# Directory for saving and loading model checkpoints.
train_dir = MODEL_DIR + "/train"
# Whether to train inception submodel variables. If True : Fine Tune the Inception v3 Model
train_inception = False
# Number of training steps.
number_of_steps = 1000000
# Frequency at which loss and global step are logged.
log_every_n_steps = 1
# Build the model.
mode = "train"
assert mode in ["train", "eval", "inference"]


## Train Config ================= Don't Change =================================
# Number of examples per epoch of training data.
num_examples_per_epoch = 586363
# Optimizer for training the model.
optimizer = "SGD"
# Learning rate for the initial phase of training.
initial_learning_rate = 2.0
learning_rate_decay_factor = 0.5
num_epochs_per_decay = 8.0
# Learning rate when fine tuning the Inception v3 parameters.
train_inception_learning_rate = 0.0005
# If not None, clip gradients to this value.
clip_gradients = 5.0
# How many model checkpoints to keep.
max_checkpoints_to_keep = 5

tf.logging.set_verbosity(tf.logging.INFO) # Enable tf.logging

## =============================================================================
# Create training directory.
if not tf.gfile.IsDirectory(train_dir):
    # if not Directory for saving and loading model checkpoints, create it
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)
# Alternatively, you can use os
# if not os.path.exists(train_dir):
#     print("Creating training directory: %s"% train_dir)
#     os.makedirs(train_dir)

# Build the TensorFlow graph. ==================================================
g = tf.Graph()
with g.as_default():
    # with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    print("tl : Build Show and Tell Model")
    images, input_seqs, target_seqs, input_mask = Build_Inputs(mode, input_file_pattern)
    # ## Example of read data
    # from im2txt.inference_utils import vocabulary
    # # vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    # vocab = vocabulary.Vocabulary('/home/haodong/Workspace/image_captioning/data/mscoco/word_counts.txt')
    # print('vocab:',[vocab.id_to_word(w) for w in range(100)])
    # sess = tf.Session()#tf.InteractiveSession()
    # sess.run(tf.initialize_all_variables())
    # with tf.Session() as sess:
    #     sess.run(tf.initialize_all_variables())
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     for i in range(3):  # number of mini-batch (step)
    #         print("Step %d" % i)
    #         img_val, caps, tar, mask = sess.run([images, input_seqs, target_seqs, input_mask])
    #         print(img_val.shape, caps.shape, tar.shape, mask.shape)
    #         for i in range(len(caps)):    # print all sentence in a batch, Note : the length is Dynamic !
    #             sentence = [vocab.id_to_word(id) for id in caps[i]]
    #             print("input_seqs:"+ " ".join(sentence))
    #             sentence = [vocab.id_to_word(id) for id in tar[i]]
    #             print("target_seqs:"+ " ".join(sentence))
    #             print("input_mask: %s" % mask[i])
    #     coord.request_stop()
    #     coord.join(threads)
    #     sess.close()
    # # ((32, 299, 299, 3), (32, 18), (32, 18), (32, 18))
    # # input_seqs:<S> a figurine with a plastic witches head is standing in front of a computer keyboard . a
    # # target_seqs:a figurine with a plastic witches head is standing in front of a computer keyboard . </S> a
    # # input_mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
    # exit()
    # ## End of Example of read data
    # with tf.device('/gpu:0'):
    net_image_embeddings = Build_Image_Embeddings(mode, images, train_inception)
    net_seq_embeddings = Build_Seq_Embeddings(input_seqs)
    total_loss, _, _, network = Build_Model(mode, net_image_embeddings, net_seq_embeddings, target_seqs, input_mask)

    network.print_layers()

    tvar = tf.all_variables() # or tf.trainable_variables()
    for idx, v in enumerate(tvar):
      print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

    # Sets up the function to restore inception variables from checkpoint.  setup_inception_initializer()
    inception_variables = tf.get_collection(
            tf.GraphKeys.VARIABLES, scope="InceptionV3")

    # Sets up the global step Tensor. setup_global_step()
    print("tl : Sets up the Global Step")
    global_step = tf.Variable(
        initial_value=0,
        dtype=tf.int32,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

    # Set up the learning rate.
    learning_rate_decay_fn = None
    if train_inception:
        # when fine-tune
        learning_rate = tf.constant(train_inception_learning_rate)
    else:
        # when don't update inception_v3
        learning_rate = tf.constant(initial_learning_rate)
        if learning_rate_decay_factor > 0:
            num_batches_per_epoch = (num_examples_per_epoch / batch_size)
            decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate,
                global_step,
                decay_steps=decay_steps,
                decay_rate=learning_rate_decay_factor,
                staircase=True)
        learning_rate_decay_fn = _learning_rate_decay_fn

    # with tf.device('/gpu:0'):
        # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=optimizer,
            clip_gradients=clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    if mode != "inference":
        print("tl : Restore InceptionV3 model from: %s" % inception_checkpoint_file)
        saver = tf.train.Saver(inception_variables)
        saver.restore(sess, inception_checkpoint_file)
        print("tl : Restore the lastest ckpt model from: %s" % train_dir)
        try:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(train_dir)) # train_dir+"/model.ckpt-960000")
        except Exception:
            print("     Not ckpt found")

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)

print('Start training') # the 1st epoch will take a while
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for step in range(sess.run(global_step), number_of_steps+1):
    start_time = time.time()
    loss, _ = sess.run([total_loss, train_op])
    print("step %d: loss = %.4f (%.2f sec)" % (step, loss, time.time() - start_time))
    if (step % 10000) == 0 and step != 0:
        # save_path = saver.save(sess, MODEL_DIR+"/train/model.ckpt-"+str(step))
        save_path = saver.save(sess, MODEL_DIR+"/train/model.ckpt", global_step=step)
        tl.files.save_npz(network.all_params , name=MODEL_DIR+'/train/model_image_caption.npz')
coord.request_stop()
coord.join(threads)
