import tensorflow as tf


def conv(inputs, kernel_size, output_num, stride_size=1, init_bias=0.0, conv_padding='SAME', stddev=0.01,
         activation_func=tf.nn.relu):
    input_size = inputs.get_shape().as_list()[-1]
    conv_weights = tf.Variable(
        tf.random_normal([kernel_size, kernel_size, input_size, output_num], dtype=tf.float32, stddev=stddev),
        name='weights')
    conv_biases = tf.Variable(tf.constant(init_bias, shape=[output_num], dtype=tf.float32), 'biases')
    conv_layer = tf.nn.conv2d(inputs, conv_weights, [1, stride_size, stride_size, 1], padding=conv_padding)
    conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
    if activation_func:
        conv_layer = activation_func(conv_layer)
    return conv_layer


def fc(inputs, output_size, init_bias=0.0, activation_func=tf.nn.relu, stddev=0.01):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 4:
        fc_weights = tf.Variable(
            tf.random_normal([input_shape[1] * input_shape[2] * input_shape[3], output_size], dtype=tf.float32,
                             stddev=stddev),
            name='weights')
        inputs = tf.reshape(inputs, [-1, fc_weights.get_shape().as_list()[0]])
    else:
        fc_weights = tf.Variable(tf.random_normal([input_shape[-1], output_size], dtype=tf.float32, stddev=stddev),
                                 name='weights')

    fc_biases = tf.Variable(tf.constant(init_bias, shape=[output_size], dtype=tf.float32), name='biases')
    fc_layer = tf.matmul(inputs, fc_weights)
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)
    if activation_func:
        fc_layer = activation_func(fc_layer)
    return fc_layer


def lrn(inputs, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(inputs, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)