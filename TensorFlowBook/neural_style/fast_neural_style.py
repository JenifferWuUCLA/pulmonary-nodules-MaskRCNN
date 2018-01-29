import glob
import importlib
import os
import sys

import numpy as np
import scipy.io
import tensorflow as tf
from PIL import Image
from skimage import transform

# Define command line args
tf.app.flags.DEFINE_string('style_image', 'starry_night.jpg', 'style image')
tf.app.flags.DEFINE_string('content_dir', '.', 'content images directory')
tf.app.flags.DEFINE_string('generator', 'johnson', 'johnson | texture_net')
tf.app.flags.DEFINE_integer('epochs', 5000, 'training epochs')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
tf.app.flags.DEFINE_integer('image_size', 256, 'image size')
tf.app.flags.DEFINE_integer('batch_size', 16, 'mini-batch size')
FLAGS = tf.app.flags.FLAGS

# Define hyper-parameters
STYLE_WEIGHT = 1.
CONTENT_WEIGHT = 1.
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYERS = ['relu4_2']
_vgg_params = None


def crop_image(image, shape):
    factor = float(min(shape[:2])) / min(image.shape[:2])
    new_size = [int(image.shape[0] * factor), int(image.shape[1] * factor)]
    if new_size[0] < shape[0]:
        new_size[0] = shape[0]
    if new_size[1] < shape[0]:
        new_size[1] = shape[0]
    resized_image = transform.resize(image, new_size)
    sample = np.asarray(resized_image) * 256
    if shape[0] < sample.shape[0] or shape[1] < sample.shape[1]:
        xx = int((sample.shape[0] - shape[0]))
        yy = int((sample.shape[1] - shape[1]))
        x_start = xx / 2
        y_start = yy / 2
        x_end = x_start + shape[0]
        y_end = y_start + shape[1]
        sample = sample[x_start:x_end, y_start:y_end, :]
    return sample


def preprocess_image(image, shape):
    return crop_image(image, shape).astype(np.float32) - 128.0


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
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5')
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
    # Calculate content loss
    _, height, width, channel = content_features.get_shape().as_list()
    content_size = height * width * channel
    return tf.nn.l2_loss(target_features - content_features) / content_size


def style_loss(target_features, style_features):
    # Calculate style loss
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
    # Total loss
    style_feats = vgg19([style_image])
    content_feats = vgg19(content_image)
    target_feats = vgg19(target_image)
    loss = 0.0
    for layer in CONTENT_LAYERS:
        layer_loss = content_loss(target_feats[layer], content_feats[layer])
        loss += CONTENT_WEIGHT * layer_loss
    for layer in STYLE_LAYERS:
        layer_loss = style_loss(target_feats[layer], style_feats[layer])
        loss += STYLE_WEIGHT * layer_loss
    return loss


def train(style, contents, image_shape,
          generator_name="johnson",
          batch_size=16, learning_rate=0.1, epochs=500):
    # target is initialized with content image
    style_name = os.path.splitext(os.path.basename(style))[0]
    style_image = np.array(Image.open(style)).astype(np.float32) - 128.0
    style_input = tf.constant(style_image, dtype=tf.float32)
    content_input_shape = [None, ] + image_shape
    content_input = tf.placeholder(tf.float32, shape=content_input_shape)

    # import generator
    generator_module = importlib.import_module(generator_name)
    target = generator_module.generator(content_input)
    saver = tf.train.Saver()

    cost = total_loss(content_input, style_input, target)
    # use Adam algorithm to optimize the total cost
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(epochs):
            batches = len(contents) / batch_size
            for batch in range(batches):
                images = contents[batch * batch_size: (batch + 1) * batch_size]
                _, loss = sess.run([train_op, cost],
                                   feed_dict={content_input: images})
                print("iter:%d, batch:%d, loss:%.9f" % (i, batch, np.sum(loss)))
        saver.save(sess, '%s_%s.ckpt' % (generator_name, style_name))


if __name__ == '__main__':
    # images are preprocessed to be zero-center
    image_shape = [FLAGS.image_size, FLAGS.image_size, 3]
    contents = []
    for f in glob.glob(FLAGS.content_dir + "/*.jpg"):
        img = np.array(Image.open(f))
        contents.append(preprocess_image(img, image_shape))
    train(FLAGS.style_image, contents, image_shape,
          batch_size=FLAGS.batch_size,
          learning_rate=FLAGS.learning_rate,
          epochs=FLAGS.epochs)
