import os
import sys

import numpy as np
import scipy.io
import tensorflow as tf
from PIL import Image

# Define command line args
tf.app.flags.DEFINE_string('style_image', 'starry_night.jpg', 'style image')
tf.app.flags.DEFINE_string('content_image', 'flower.jpg', 'content image')
tf.app.flags.DEFINE_integer('epochs', 500, 'training epochs')
tf.app.flags.DEFINE_float('learning_rate', 0.5, 'learning rate')
FLAGS = tf.app.flags.FLAGS

# Define hyper-parameters
STYLE_WEIGHT = 10.
CONTENT_WEIGHT = 1.
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYERS = ['relu4_2']
_vgg_params = None


def vgg_params():
    # Load pre-trained VGG19 params
    global _vgg_params
    if _vgg_params is None:
        file = 'imagenet-vgg-verydeep-19.mat'
        if os.path.isfile(file):
            _vgg_params = scipy.io.loadmat(file)
        else:
            sys.stderr.write('Please download imagenet-vgg-verydeep-19.mat from'
                             ' http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat\n')
            sys.exit(1)
    return _vgg_params


def vgg19(input_image):
    # VGG19 network
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    weights = vgg_params()['layers'][0]
    net = input_image
    network = {}
    for i, name in enumerate(layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet weights: [width, height, in_channels, out_channels]
            # tensorflow weights: [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            conv = tf.nn.conv2d(net, tf.constant(kernels),
                                strides=(1, 1, 1, 1), padding='SAME',
                                name=name)
            net = tf.nn.bias_add(conv, bias.reshape(-1))
            net = tf.nn.relu(net)
        elif layer_type == 'pool':
            net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1),
                                 strides=(1, 2, 2, 1),
                                 padding='SAME')
        network[name] = net
    return network


def content_loss(target_features, content_features):
    _, height, width, channel = content_features.get_shape().as_list()
    content_size = height * width * channel
    return tf.nn.l2_loss(target_features - content_features) / content_size


def style_loss(target_features, style_features):
    _, height, width, channel = target_features.get_shape().as_list()
    size = height * width * channel
    target_features = tf.reshape(target_features, (-1, channel))
    target_gram = tf.matmul(tf.transpose(target_features),
                            target_features) / size
    style_features = tf.reshape(style_features, (-1, channel))
    style_gram = tf.matmul(tf.transpose(style_features),
                           style_features) / size
    gram_size = channel * channel
    return tf.nn.l2_loss(target_gram - style_gram) / gram_size


def total_loss(content_image, style_image, target_image):
    style_feats = vgg19([style_image])
    content_feats = vgg19([content_image])
    target_feats = vgg19([target_image])
    loss = 0.0
    for layer in CONTENT_LAYERS:
        layer_loss = content_loss(target_feats[layer], content_feats[layer])
        loss += CONTENT_WEIGHT * layer_loss
    for layer in STYLE_LAYERS:
        layer_loss = style_loss(target_feats[layer], style_feats[layer])
        loss += STYLE_WEIGHT * layer_loss
    return loss


def stylize(style_image, content_image, learning_rate=0.1, epochs=500):
    # target is initialized with content image
    target = tf.Variable(content_image, dtype=tf.float32)
    style_input = tf.constant(style_image, dtype=tf.float32)
    content_input = tf.constant(content_image, dtype=tf.float32)
    cost = total_loss(content_input, style_input, target)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(epochs):
            _, loss, target_image = sess.run([train_op, cost, target])
            print("iter:%d, loss:%.9f" % (i, loss))
            if (i + 1) % 100 == 0:
                # save target image every 100 iterations
                image = np.clip(target_image + 128.0, 0, 255).astype(np.uint8)
                Image.fromarray(image).save("out/neural_%d.jpg" % (i + 1))


if __name__ == '__main__':
    # images are preprocessed to be zero-center
    style = Image.open(FLAGS.style_image)
    style = np.array(style).astype(np.float32) - 128.0
    content = Image.open(FLAGS.content_image)
    content = np.array(content).astype(np.float32) - 128.0
    stylize(style, content, FLAGS.learning_rate, FLAGS.epochs)
