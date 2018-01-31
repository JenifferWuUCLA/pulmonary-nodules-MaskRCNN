import tensorflow as tf
import tflearn


def generator(input_image):
    relu = tf.nn.relu
    conv2d = tflearn.conv_2d

    def batch_norm(x):
        mean, var = tf.nn.moments(x, axes=[1, 2, 3])
        return tf.nn.batch_normalization(x, mean, var, 0, 1, 1e-5)

    def deconv2d(x, n_filter, ksize, strides=1):
        _, h, w, _ = x.get_shape().as_list()
        output_shape = [strides * h, strides * w]
        return tflearn.conv_2d_transpose(x, n_filter, ksize, output_shape,
                                         strides)

    def res_block(x):
        net = relu(batch_norm(conv2d(x, 128, 3)))
        net = batch_norm(conv2d(net, 128, 3))
        return x + net

    net = relu(batch_norm(conv2d(input_image, 32, 9)))
    net = relu(batch_norm(conv2d(net, 64, 4, strides=2)))
    net = relu(batch_norm(conv2d(net, 128, 4, strides=2)))
    for i in range(5):
        net = res_block(net)
    net = relu(batch_norm(deconv2d(net, 64, 4, strides=2)))
    net = relu(batch_norm(deconv2d(net, 32, 4, strides=2)))
    net = deconv2d(net, 3, 9)
    return net
