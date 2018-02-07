import tensorflow as tf
import ops as op


def input_placeholder(image_size, image_channel, label_cnt):
    with tf.name_scope('inputlayer'):
        inputs = tf.placeholder("float", [None, image_size, image_size, image_channel], 'inputs')
        labels = tf.placeholder("float", [None, label_cnt], 'labels')
    dropout_keep_prob = tf.placeholder("float", None, 'keep_prob')
    learning_rate = tf.placeholder("float", None, name='learning_rate')

    return inputs, labels, dropout_keep_prob, learning_rate


def inference2(inputs, dropout_keep_prob, label_cnt):

  # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        #print_activations(conv1)
        #parameters += [kernel, biases]

  # lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool1
    pool1 = tf.nn.max_pool(lrn1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
    #print_activations(pool1)

  # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        #parameters += [kernel, biases]
        #print_activations(conv2)

  # lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool2
    pool2 = tf.nn.max_pool(lrn2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
    #print_activations(pool2)

  # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        #parameters += [kernel, biases]
        #print_activations(conv3)

  # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        #parameters += [kernel, biases]
        #print_activations(conv4)

  # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        #parameters += [kernel, biases]
        #print_activations(conv5)


    # fc layer 1
    with tf.name_scope('fc1layer'):
        fc1 = op.fc(conv5, 32, 1.0)
        fc1 = tf.nn.dropout(fc1, dropout_keep_prob)

    # fc layer 2
    with tf.name_scope('fc2layer'):
        #fc2 = op.fc(fc1, 4096, 1.0)
        fc2 = op.fc(fc1, 16, 1.0)
        fc2 = tf.nn.dropout(fc2, dropout_keep_prob)

    # fc layer 3 - output
    with tf.name_scope('fc3layer'):
        return op.fc(fc2, label_cnt, 1.0, None)	
		
		
	
def inference(inputs, dropout_keep_prob, label_cnt):
    # todo: change lrn parameters
    # conv layer 1
    with tf.name_scope('conv1layer'):
        conv1 = op.conv(inputs, 7, 96, 3)
        conv1 = op.lrn(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # conv layer 2
    with tf.name_scope('conv2layer'):
        conv2 = op.conv(conv1, 5, 256, 1, 1.0)
        conv2 = op.lrn(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # conv layer 3
    with tf.name_scope('conv3layer'):
        conv3 = op.conv(conv2, 3, 384, 1)

    # conv layer 4
    with tf.name_scope('conv4layer'):
        conv4 = op.conv(conv3, 3, 384, 1, 1.0)

    # conv layer 5
    with tf.name_scope('conv5layer'):
        conv5 = op.conv(conv4, 3, 256, 1, 1.0)
        conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # fc layer 1
    with tf.name_scope('fc1layer'):
        fc1 = op.fc(conv5, 4096, 1.0)
        #fc1 = op.fc(conv4, 32, 1.0)
        fc1 = tf.nn.dropout(fc1, dropout_keep_prob)

    # fc layer 2
    with tf.name_scope('fc2layer'):
        fc2 = op.fc(fc1, 4096, 1.0)
        #fc2 = op.fc(fc1, 16, 1.0)
        fc2 = tf.nn.dropout(fc2, dropout_keep_prob)

    # fc layer 3 - output
    with tf.name_scope('fc3layer'):
        return op.fc(fc2, label_cnt, 1.0, None)


def accuracy(logits, labels):
    # accuracy
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        
    return accuracy


def loss(logits, labels):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
       
    return loss


def train_rms_prop(loss, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp'):
    return tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon, use_locking, name).minimize(loss)
	
def onehot(indices, depth, on_value=None, off_value=None,
            axis=None, dtype=None, name=None):
  """Returns a one-hot tensor.
  The locations represented by indices in `indices` take value `on_value`,
  while all other locations take value `off_value`.
  `on_value` and `off_value` must have matching data types. If `dtype` is also
  provided, they must be the same data type as specified by `dtype`.
  If `on_value` is not provided, it will default to the value `1` with type
  `dtype`
  If `off_value` is not provided, it will default to the value `0` with type
  `dtype`
  If the input `indices` is rank `N`, the output will have rank `N+1`. The
  new axis is created at dimension `axis` (default: the new axis is appended
  at the end).
  If `indices` is a scalar the output shape will be a vector of length `depth`
  If `indices` is a vector of length `features`, the output shape will be:
  ```
    features x depth if axis == -1
    depth x features if axis == 0
  ```
  If `indices` is a matrix (batch) with shape `[batch, features]`, the output
  shape will be:
  ```
    batch x features x depth if axis == -1
    batch x depth x features if axis == 1
    depth x batch x features if axis == 0
  ```
  If `dtype` is not provided, it will attempt to assume the data type of
  `on_value` or `off_value`, if one or both are passed in. If none of
  `on_value`, `off_value`, or `dtype` are provided, `dtype` will default to the
  value `tf.float32`.
  Note: If a non-numeric data type output is desired (`tf.string`, `tf.bool`,
  etc.), both `on_value` and `off_value` _must_ be provided to `one_hot`.
  Examples
  =========
  Suppose that
  ```python
    indices = [0, 2, -1, 1]
    depth = 3
    on_value = 5.0
    off_value = 0.0
    axis = -1
  ```
  Then output is `[4 x 3]`:
  ```python
    output =
    [5.0 0.0 0.0]  // one_hot(0)
    [0.0 0.0 5.0]  // one_hot(2)
    [0.0 0.0 0.0]  // one_hot(-1)
    [0.0 5.0 0.0]  // one_hot(1)
  ```
  Suppose that
  ```python
    indices = [[0, 2], [1, -1]]
    depth = 3
    on_value = 1.0
    off_value = 0.0
    axis = -1
  ```
  Then output is `[2 x 2 x 3]`:
  ```python
    output =
    [
      [1.0, 0.0, 0.0]  // one_hot(0)
      [0.0, 0.0, 1.0]  // one_hot(2)
    ][
      [0.0, 1.0, 0.0]  // one_hot(1)
      [0.0, 0.0, 0.0]  // one_hot(-1)
    ]
  ```
  Using default values for `on_value` and `off_value`:
  ```python
    indices = [0, 1, 2]
    depth = 3
  ```
  The output will be
  ```python
    output =
    [[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]]
  ```
  Args:
    indices: A `Tensor` of indices.
    depth: A scalar defining the depth of the one hot dimension.
    on_value: A scalar defining the value to fill in output when `indices[j]
      = i`. (default: 1)
    off_value: A scalar defining the value to fill in output when `indices[j]
      != i`. (default: 0)
    axis: The axis to fill (default: -1, a new inner-most axis).
    dtype: The data type of the output tensor.
  Returns:
    output: The one-hot tensor.
  Raises:
    TypeError: If dtype of either `on_value` or `off_value` don't match `dtype`
    TypeError: If dtype of `on_value` and `off_value` don't match one another
  """
  with op.name_scope(name, "one_hot", [indices, depth, on_value, off_value,
                                        axis, dtype]) as name:
    on_exists = on_value is not None
    off_exists = off_value is not None

    on_dtype = op.convert_to_tensor(on_value).dtype.base_dtype if on_exists \
                  else None
    off_dtype = op.convert_to_tensor(off_value).dtype.base_dtype if off_exists\
                  else None

    if on_exists or off_exists:
      if dtype is not None:
        # Ensure provided on_value and/or off_value match dtype
        if (on_exists and on_dtype != dtype):
          raise TypeError("dtype {0} of on_value does not match " \
                          "dtype parameter {1}".format(on_dtype, dtype))
        if (off_exists and off_dtype != dtype):
          raise TypeError("dtype {0} of off_value does not match " \
                          "dtype parameter {1}".format(off_dtype, dtype))
      else:
        # dtype not provided: automatically assign it
        dtype = on_dtype if on_exists else off_dtype
    elif dtype is None:
      # None of on_value, off_value, or dtype provided. Default dtype to float32
      dtype = dtypes.float32

    if not on_exists:
      # on_value not provided: assign to value 1 of type dtype
      on_value = op.convert_to_tensor(1, dtype, name="on_value")
      on_dtype = dtype
    if not off_exists:
      # off_value not provided: assign to value 0 of type dtype
      off_value = op.convert_to_tensor(0, dtype, name="off_value")
      off_dtype = dtype

    if on_dtype != off_dtype:
      raise TypeError("dtype {0} of on_value does not match " \
                      "dtype {1} of off_value".format(on_dtype, off_dtype))

    return gen_array_ops._one_hot(indices, depth, on_value, off_value, axis,
                                  name)