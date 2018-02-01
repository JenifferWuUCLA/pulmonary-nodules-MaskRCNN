#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3_arg_scope
slim = tf.contrib.slim

## Model Config ================= Don't Change =================================
# Image format ("jpeg" or "png").
image_format = "jpeg"
# Approximate number of values per input shard. Used to ensure sufficient
# mixing between shards in training.
values_per_input_shard = 2300
# Minimum number of shards to keep in the input queue.
input_queue_capacity_factor = 2
# Number of threads for prefetching SequenceExample protos.
num_input_reader_threads = 1
# Name of the SequenceExample context feature containing image data.
image_feature_name = "image/data"
# Name of the SequenceExample feature list containing integer captions.
caption_feature_name = "image/caption_ids"
# Number of unique words in the vocab (plus 1, for <UNK>).
# The default value is larger than the expected actual vocab size to allow
# for differences between tokenizer versions used in preprocessing. There is
# no harm in using a value greater than the actual vocab size, but using a
# value less than the actual vocab size will result in an error.
vocab_size = 12000
# Number of threads for image preprocessing. Should be a multiple of 2.
num_preprocess_threads = 4
# Batch size.
batch_size = 32
# Dimensions of Inception v3 input images.
resize_height=346   # before crop
resize_width=346
image_height = 299  # crop
image_width = 299
# Scale used to initialize model variables.
initializer_scale = 0.08
# LSTM input and output dimensionality, respectively.
embedding_size = 512
num_lstm_units = 512
# If < 1.0, the dropout keep probability applied to LSTM variables.
lstm_dropout_keep_prob = 0.7

# Initializer
initializer = tf.random_uniform_initializer(minval=-initializer_scale, maxval=initializer_scale)


def distort_image(image, thread_id):
  """Perform random distortions on an image.
  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
  Returns:````
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal"):#, values=[image]): # DH MOdify
  # with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  color_ordering = thread_id % 2
  with tf.name_scope("distort_color"):#, values=[image]): # DH MOdify
  # with tf.name_scope("distort_color", values=[image]): # DH MOdify
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image

def process_image(mode, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
    encoded_image: A scalar string Tensor; the encoded image.
    thread_id: Preprocessing thread id used to select the ordering of color
    distortions.

    Returns:
    A float32 Tensor of shape [height, width, 3]; the processed image.
    """

    def _process_image(encoded_image,
                      is_training,
                      height,
                      width,
                      resize_height=resize_height,
                      resize_width=resize_width,
                      thread_id=0,
                      image_format="jpeg"):
      """Decode an image, resize and apply random distortions.
      In training, images are distorted slightly differently depending on thread_id.
      Args:
        encoded_image: String Tensor containing the image.
        is_training: Boolean; whether preprocessing for training or eval.
        height: Height of the output image.
        width: Width of the output image.
        resize_height: If > 0, resize height before crop to final dimensions.
        resize_width: If > 0, resize width before crop to final dimensions.
        thread_id: Preprocessing thread id used to select the ordering of color
          distortions. There should be a multiple of 2 preprocessing threads.
        image_format: "jpeg" or "png".
      Returns:
        A float32 Tensor of shape [height, width, 3] with values in [-1, 1].
      Raises:
        ValueError: If image_format is invalid.
      """
      # Helper function to log an image summary to the visualizer. Summaries are
      # only logged in thread 0.
      def image_summary(name, image):
        if not thread_id:
          tf.image_summary(name, tf.expand_dims(image, 0))

      # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
      with tf.name_scope("decode"):#, values=[encoded_image]):   # DH modify
      # with tf.name_scope("decode", values=[encoded_image]):   # DH modify
        if image_format == "jpeg":
          image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == "png":
          image = tf.image.decode_png(encoded_image, channels=3)
        else:
          raise ValueError("Invalid image format: %s" % image_format)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image_summary("original_image", image)

      # Resize image.
      assert (resize_height > 0) == (resize_width > 0)
      if resize_height:
        try:
            image = tf.image.resize_images(image,
                                           size=[resize_height, resize_width],
                                           method=tf.image.ResizeMethod.BILINEAR)
        except:
            image = tf.image.resize_images(image,     # for TF 0.10
                                           new_height=resize_height,
                                           new_width=resize_width,
                                           method=tf.image.ResizeMethod.BILINEAR)

      # Crop to final dimensions.
      if is_training:
        image = tf.random_crop(image, [height, width, 3])
      else:
        # Central crop, assuming resize_height > height, resize_width > width.
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)

      image_summary("resized_image", image)

      # Randomly distort the image.
      if is_training:
        image = distort_image(image, thread_id)

      image_summary("final_image", image)

      # Rescale to [-1,1] instead of [0, 1]
      image = tf.sub(image, 0.5)
      image = tf.mul(image, 2.0)
      return image
    return _process_image(encoded_image,
                          is_training= mode == 'train', # If Traning, distort image; if None, crop central part; the size unchange.
                          height=image_height,
                          width=image_width,
                          thread_id=thread_id,
                          image_format=image_format)


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions, returns the images, input sequence and
  output sequence.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example 1
  -----------

    Actual captions in the batch ('-' denotes padded character):
    |      [
    |        [ 1 2 5 4 5 ],
    |        [ 1 2 3 4 - ],
    |        [ 1 2 3 - - ],
    |      ]
    |
    |    input_seqs:
    |      [
    |        [ 1 2 3 4 ],
    |        [ 1 2 3 - ],
    |        [ 1 2 - - ],
    |      ]
    |
    |    target_seqs:
    |      [
    |        [ 2 3 4 5 ],
    |        [ 2 3 4 - ],
    |        [ 2 3 - - ],
    |      ]
    |
    |    mask:
    |      [
    |        [ 1 1 1 1 ],
    |        [ 1 1 1 0 ],
    |        [ 1 1 0 0 ],
    |      ]

  Example 2
  -----------
  - input_seqs - <S> a figurine with a plastic witches head is standing in front of a computer keyboard . a
  - target_seqs - a figurine with a plastic witches head is standing in front of a computer keyboard . </S> a
  - input_mask - [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]

  Parameters
  -----------
    images_and_captions : A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size : Batch size.
    queue_capacity : Queue capacity.
    add_summaries : If true, add caption length summaries.

  Returns
  --------
    images : A Tensor of shape [batch_size, height, width, channels].
    input_seqs : An int32 Tensor of shape [batch_size, padded_length].
    target_seqs : An int32 Tensor of shape [batch_size, padded_length].
    mask : An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  enqueue_list = []
  for image, caption in images_and_captions:
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.sub(caption_length, 1), 0)

    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    enqueue_list.append([image, input_seq, target_seq, indicator])

  images, input_seqs, target_seqs, mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.scalar_summary("caption_length/batch_min", tf.reduce_min(lengths))
    tf.scalar_summary("caption_length/batch_max", tf.reduce_max(lengths))
    tf.scalar_summary("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, input_seqs, target_seqs, mask


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
            If True, shuffle, otherwise, no shuffle.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    print("shuffle for training")
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    print("no shuffle")
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.scalar_summary(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue



def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3"):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  # with tf.variable_scope(scope, "InceptionV3", [images]) as scope:    # Original      InceptionV3/Conv2d_1a_3x3/weights   ckpt don't have this variable
  with tf.variable_scope("InceptionV3") as scope:   # Hao Dong                      InceptionV3/Conv2d_1a_3x3/weights:0     ckpt have this variable
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)
        with tf.variable_scope("logits"):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_inception_model_training,
              scope="dropout")
          net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  # return net              # Original
  return net, end_points    # Hao Dong


def Build_Inputs(mode, input_file_pattern):
    """Input prefetching, preprocessing and batching.   build_inputs()

    Outputs:
      images
      input_seqs
      target_seqs (training and eval only)
      input_mask (training and eval only)
    """
    print("tl : Build Inputs = inference (placeholder),  train/eval (TFRecord tensor)")
    # Reader for the input data.
    reader = tf.TFRecordReader()
    if mode == "inference":
        # In inference mode, images and inputs are fed via placeholders.
        image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
        input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # 1 word id
                                  name="input_feed")
        # Process image and insert batch dimensions.
        images = tf.expand_dims(process_image(mode, image_feed), 0)
        input_seqs = tf.expand_dims(input_feed, 1)
        # No target sequences or input mask in inference mode.
        target_seqs = None
        input_mask = None

    elif mode in ["train", "eval"]:
        # Prefetch serialized SequenceExample protos.
        # input_queue = input_ops.prefetch_input_data(
        input_queue = prefetch_input_data(
              reader,
              input_file_pattern,
              is_training= mode == "train", # Hao Dong : if training,
              batch_size=batch_size,
              values_per_shard=values_per_input_shard,
              input_queue_capacity_factor=input_queue_capacity_factor,
              num_reader_threads=num_input_reader_threads)

        # Image processing and random distortion. Split across multiple threads
        # with each thread applying a slightly different distortion.
        assert num_preprocess_threads % 2 == 0
        images_and_captions = []
        for thread_id in range(num_preprocess_threads):
            serialized_sequence_example = input_queue.dequeue()
            context, sequence = tf.parse_single_sequence_example(
                    serialized_sequence_example,
                    context_features={
                    image_feature_name: tf.FixedLenFeature([], dtype=tf.string)
                    },
                    sequence_features={
                    caption_feature_name: tf.FixedLenSequenceFeature([], dtype=tf.int64),
                    })
            encoded_image = context[image_feature_name]
            caption = sequence[caption_feature_name]

            image = process_image(mode, encoded_image, thread_id=thread_id)
            images_and_captions.append([image, caption])

        # Batch inputs.
        queue_capacity = (2 * num_preprocess_threads * batch_size)
        images, input_seqs, target_seqs, input_mask = (
              batch_with_dynamic_pad(images_and_captions,
                                               batch_size=batch_size,
                                               queue_capacity=queue_capacity))

        images = images
        input_seqs = input_seqs
        target_seqs = target_seqs
        input_mask = input_mask
    else:
        raise Exception("model in [\"train\", \"eval\", \"inference\"]")
    if mode == 'inference':
        return images, input_seqs, target_seqs, input_mask, input_feed
    else:
        return images, input_seqs, target_seqs, input_mask

def Build_Image_Embeddings(mode, images, train_inception):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    print("tl : Build Image Embeddings = InceptionV3 + Dense Layer / uses SlimNetsLayer and DenseLayer instead")

    with slim.arg_scope(inception_v3_arg_scope()):
        net_img_in = tl.layers.InputLayer(images, name='input_image_layer')
        network = tl.layers.SlimNetsLayer(layer=net_img_in, slim_layer=inception_v3,
                                          slim_args= {
                                                 'trainable' : train_inception,
                                                 'is_training' : mode == 'train',
                                                },
                                            name='',
                                            )
    network = tl.layers.DenseLayer(network,
                                    n_units = embedding_size,
                                    act = tf.identity,
                                    W_init = initializer,
                                    b_init = None,          # no biases
                                    name='image_embedding')
    return network

def Build_Seq_Embeddings(input_seqs):
    """Builds the input sequence embeddings.

    Inputs:
    self.input_seqs

    Outputs:
    self.seq_embeddings
    """
    print("tl : Build Seq Embedding")
    print("     EmbeddingInputlayer")
    network = tl.layers.EmbeddingInputlayer(
        inputs = input_seqs,
        vocabulary_size = vocab_size,
        embedding_size = embedding_size,
        E_init = initializer,
        name = 'seq_embedding')

    return  network

def Build_Model(mode, net_image_embeddings, net_seq_embeddings, target_seqs, input_mask):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    print("tl : Build Model = image_embeddings + seq_embeddings + LSTMs + Dropout")
    print('     LSTM')
    if mode == 'inference':
        with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
            tl.layers.set_name_reuse(True)
            net_image_embeddings = tl.layers.ReshapeLayer(net_image_embeddings, shape=(-1, 1, embedding_size))
            print(net_image_embeddings.outputs)
            # exit()
            net_img_rnn = tl.layers.DynamicRNNLayer(net_image_embeddings,
                                      cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
                                      n_hidden = num_lstm_units,
                                      dropout = None,
                                      initial_state = None,
                                      sequence_length = tf.ones([batch_size]),
                                      return_seq_2d = True,   # stack denselayer after it
                                      name = 'embed',
                                      )
            lstm_scope.reuse_variables()
            # exit()

            # # In inference mode, use concatenated states for convenient feeding and fetching.
            # Placeholder for feeding a batch of concatenated states.
            state_feed = tf.placeholder(dtype=tf.float32,
                                        shape=[None, sum(net_img_rnn.cell.state_size)],
                                        name="state_feed")
            state_tuple = tf.split(1, 2, state_feed)
            state_tuple = tf.nn.rnn_cell.LSTMStateTuple(state_tuple[0], state_tuple[1])

            net_seq_rnn = tl.layers.DynamicRNNLayer(net_seq_embeddings,
                          cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
                          n_hidden = num_lstm_units,
                          dropout = None,
                          initial_state = state_tuple,  # different with training
                          sequence_length = tf.ones([batch_size]),
                          return_seq_2d = True,   # stack denselayer after it
                          name = 'embed',
                          )
            network = net_seq_rnn
            network.all_layers = net_image_embeddings.all_layers + network.all_layers
            network.all_params = net_image_embeddings.all_params + network.all_params
    else:
        with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
            if mode == 'train':
                dropout = lstm_dropout_keep_prob
            else:
                dropout = None
            net_image_embeddings = tl.layers.ReshapeLayer(net_image_embeddings, shape=(batch_size, 1, embedding_size))
            net_img_rnn = tl.layers.DynamicRNNLayer(net_image_embeddings,
                                      cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
                                      n_hidden = num_lstm_units,
                                      initializer = initializer,
                                      dropout = dropout,
                                      initial_state = None,
                                      sequence_length = tf.ones([batch_size]),
                                      return_seq_2d = True,   # stack denselayer after it
                                      name = 'embed',
                                      )
            # Then, uses the hidden state which contains image info as the initial_state when feeding the sentence.
            lstm_scope.reuse_variables()
            tl.layers.set_name_reuse(True)
            network = tl.layers.DynamicRNNLayer(net_seq_embeddings,
                                      cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
                                      n_hidden = num_lstm_units,
                                      initializer = initializer,
                                      dropout = dropout,
                                      initial_state = net_img_rnn.final_state,      # feed in hidden state after feeding image
                                      sequence_length = tf.reduce_sum(input_mask, 1),
                                      return_seq_2d = True,     # stack denselayer after it
                                      name = 'embed',
                                      )
            network.all_layers = net_image_embeddings.all_layers + network.all_layers
            network.all_params = net_image_embeddings.all_params + network.all_params

    print('     Output layer = Dense Layer')
    network = tl.layers.DenseLayer(network, n_units=vocab_size, act=tf.identity, W_init=initializer, name="logits") # TL
    logits = network.outputs

    # network.print_layers()

    if mode == "inference":
      softmax = tf.nn.softmax(logits, name="softmax")
      return softmax, net_img_rnn, net_seq_rnn, state_feed
    else:
      batch_loss, losses, weights, _ = tl.cost.cross_entropy_seq_with_mask(logits, target_seqs, input_mask, return_details=True)    # TL

      tf.contrib.losses.add_loss(batch_loss)
      total_loss = tf.contrib.losses.get_total_loss()

      # Add summaries.
      tf.scalar_summary("batch_loss", batch_loss)
      tf.scalar_summary("total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

      total_loss = total_loss
      target_cross_entropy_losses = losses  # Used in evaluation.
      target_cross_entropy_loss_weights = weights  # Used in evaluation.
      return total_loss, target_cross_entropy_losses, target_cross_entropy_loss_weights, network














#
