import csv
import datetime
import math
import operator
import os.path
import random
import sys

import numpy as np
import tensorflow as tf
import torchfile
from PIL import Image
from skimage import img_as_float
from sklearn.model_selection import StratifiedKFold

# from tensorflow.python.framework import ops

np.set_printoptions(threshold=np.inf)

NUM_CLASSES = 2
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]


class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    EXTRACTED FROM: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Parameters
    ----------
    actual : ndarray
             A list of elements that are to be predicted (order doesn't matter)
    predicted : ndarray
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input_data lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

            # if not actual:
            # return 0.0

    return score / min(len(actual), k)


def print_params(list_params):
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for i in xrange(1, len(sys.argv)):
        print list_params[i - 1] + '= ' + sys.argv[i]
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


def softmax(array):
    expa = np.exp(array)
    sumexp = np.sum(expa, axis=1)
    sumexp_repeat = np.repeat(sumexp, NUM_CLASSES, axis=0).reshape((sumexp.shape[0], NUM_CLASSES))
    soft_calc = np.divide(expa, sumexp_repeat)
    return soft_calc


'''
PROCESS IMAGES
'''


def normalize_images(data, mean_full, std_full=None):
    data[:, :, :, 0] = np.subtract(data[:, :, :, 0], mean_full[0])
    data[:, :, :, 1] = np.subtract(data[:, :, :, 1], mean_full[1])
    data[:, :, :, 2] = np.subtract(data[:, :, :, 2], mean_full[2])

    if std_full is not None:
        data[:, :, :, 0] = np.divide(data[:, :, :, 0], std_full[0])
        data[:, :, :, 1] = np.divide(data[:, :, :, 1], std_full[1])
        data[:, :, :, 2] = np.divide(data[:, :, :, 2], std_full[2])


def compute_image_mean(data):
    mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    std_full = np.std(data, axis=0, ddof=1)[0, 0, :]

    return mean_full, std_full


def read_csv_file(path, is_test=False):
    img_names = []
    img_classes = []
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            img_names.append(row[0])
            if is_test is False:
                img_classes.append(int(row[1]))

    return img_names, img_classes


def crop_center_image(img, x_rand, y_rand, final_crop_size):
    # img.crop((left, top, right, bottom))
    return img.crop((x_rand, y_rand, x_rand + final_crop_size, y_rand + final_crop_size))


'''
def random_and_resize(img, final_crop_size=64, num_rands=3, resize_to=None):
    imgs = []

    img_size = img.size
    for i in xrange(num_rands):
        x_rand = np.random.randint(0, img_size[0])
        y_rand = np.random.randint(0, img_size[1])
        # im = random_crop_image(img, x_rand, y_rand, final_crop_size)

        if resize_to is not None:
            im = im.resize((resize_to, resize_to), Image.ANTIALIAS)
            print im.size
            im.load()
        else:
            im.load()

        imgs.append(img_as_float(im))

    return imgs
'''


def load_images(dataset_path, img_names, img_classes, resize_to=224, is_test=False):
    data = []
    classes = []
    names = []

    for i in xrange(len(img_names) - 1, -1, -1):
        try:
            img = Image.open(dataset_path + img_names[i] + '.jpg')
        except:
            print BatchColors.FAIL + "Error: Cannot find image " + dataset_path + img_names[i] + \
                  '.jpg' + BatchColors.ENDC
            del img_names[i]
            del img_classes[i]
            continue

        width, height = img.size
        min_size = (height if width > height else width)

        left = (width - min_size) / 2
        top = (height - min_size) / 2
        right = (width + min_size) / 2
        bottom = (height + min_size) / 2
        # print width, height, min_size, left, top, right, bottom
        cropped_img = img.crop((left, top, right, bottom))
        w, h = cropped_img.size
        if w != h:
            print cropped_img.size
            return

        img_resize = cropped_img.resize((resize_to, resize_to), Image.ANTIALIAS)
        img_resize.load()

        img_float = img_as_float(img_resize)
        data.append(img_float)
        names.append(img_names[i])
        if is_test is False:
            classes.append(img_classes[i])

            # DATA AUGMENTATION
            # data.append(np.fliplr(img_float))
            # classes.append(img_classes[i])

    data_arr = np.asarray(data)
    classes_arr = np.asarray(classes)
    names_arr = np.asarray(names)
    print data_arr.shape, classes_arr.shape, names_arr.shape

    return data_arr, classes_arr, names_arr


'''
TensorFlow
'''


def load_t7(session, data_path, network_variable_names, num_blocks):
    f = torchfile.load(data_path)

    first = True
    block_bool = False
    index_module = 0
    block_counter = 0

    i = 1
    while i < len(network_variable_names):
        if 'init' in network_variable_names[i].name:
            continue
        if block_bool is False:
            # print block_counter, len(num_blocks)
            if first is True:
                # print 'initial'
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(
                    np.swapaxes(np.swapaxes(np.swapaxes(f['modules'][index_module]['weight'], 0, 2), 1, 3), 2, 3)))
                index_module += 1
                i += 1

                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(f['modules'][index_module]['running_mean']))
                i += 1
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(f['modules'][index_module]['running_var']))
                i += 1
                index_module += 3
                first = False
            elif block_counter == len(num_blocks):
                # print 'final'
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(f['modules'][index_module]['running_mean']))
                i += 1
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(f['modules'][index_module]['running_var']))
                i = len(network_variable_names) + 1
            else:
                # print 'transition'
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(f['modules'][index_module]['running_mean']))
                i += 1
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(f['modules'][index_module]['running_var']))
                i += 1
                index_module += 2

                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(
                    np.swapaxes(np.swapaxes(np.swapaxes(f['modules'][index_module]['weight'], 0, 2), 1, 3), 2, 3)))
                i += 1
                index_module += 2

            block_bool = True
        elif block_bool is True:
            '''print f['modules'][i] ## concat
            print f['modules'][4]['modules'][0] ## identity
            print f['modules'][4]['modules'][1] ## sequential
            print f['modules'][4]['modules'][1]['modules'][0] ## BN + bias
            print f['modules'][4]['modules'][1]['modules'][1] ## RELU
            print f['modules'][4]['modules'][1]['modules'][2] ## conv
            print f['modules'][4]['modules'][1]['modules'][3]['running_mean']['weight']['running_var']['bias']
            print f['modules'][4]['modules'][1]['modules'][4] ## RELU
            print f['modules'][4]['modules'][1]['modules'][5]['weight'] ## conv'''

            '''
            dense1/denseblock1/moving_mean:0
            dense1/denseblock1/moving_variance:0
            dense1/denseblock1/bottleneck/conv/weights:0
            dense1/denseblock1/bottleneck/moving_mean:0
            dense1/denseblock1/bottleneck/moving_variance:0
            dense1/denseblock1/conv/weights:0
            '''

            # print 'block', block_counter+1
            for j in xrange(num_blocks[block_counter]):
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(
                    f['modules'][index_module]['modules'][1]['modules'][0]['running_mean']))
                i += 1
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(
                    f['modules'][index_module]['modules'][1]['modules'][0]['running_var']))
                i += 1
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(np.swapaxes(
                    np.swapaxes(np.swapaxes(f['modules'][index_module]['modules'][1]['modules'][2]['weight'], 0, 2), 1,
                                3), 2, 3)))
                i += 1

                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(
                    f['modules'][index_module]['modules'][1]['modules'][3]['running_mean']))
                i += 1
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(
                    f['modules'][index_module]['modules'][1]['modules'][3]['running_var']))
                i += 1
                # print network_variable_names[i].name
                session.run(network_variable_names[i].assign(np.swapaxes(
                    np.swapaxes(np.swapaxes(f['modules'][index_module]['modules'][1]['modules'][5]['weight'], 0, 2), 1,
                                3), 2, 3)))
                i += 1

                index_module += 1

            block_bool = False
            block_counter += 1


def load_npy(session, data_path, ignore_missing=False, ignore_params=None):
    """Load npy of network weights
        Args:
            session: The current TensorFlow session
            data_path: The path to the numpy-serialized network weights
            ignore_missing: If true, serialized weights for missing layers are ignored
            ignore_params: name of the layer to ignore (for example, for fine-tuning, one should ignore the last
            fully connected with number of classes is different.
        Returns:
            Precision @ k
        Raises:
            ValueError: existing variable not in ignore_missing
        """
    data_dict = np.load(data_path).item()
    for op_name in data_dict:
        if ignore_params is None or op_name not in ignore_params:
            # print op_name
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    # print param_name
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    print BatchColors.OKGREEN + "Model loaded!" + BatchColors.ENDC


def leaky_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def _variable_on_cpu(name, shape, ini):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=ini, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, ini, wd):
    var = _variable_on_cpu(name, shape, ini)
    # tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    # tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    # tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        try:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        except:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv_layer(input_data, kernel_shape, name, weight_decay, is_training, strides, group=1, pad='SAME',
                activation='leaky_relu', batch_norm=True, is_activated=True, has_bias=True):
    with tf.variable_scope(name) as scope:
        kernel_shape[2] = kernel_shape[2] / group
        weights = _variable_with_weight_decay('weights', shape=kernel_shape,
                                              ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                              wd=weight_decay)
        if has_bias is True:
            biases = _variable_on_cpu('biases', kernel_shape[-1], tf.constant_initializer(0.1))

        if group == 1:
            conv_op = tf.nn.conv2d(input_data, weights, strides, padding=pad)
        else:
            try:
                input_data_groups = tf.split(3, group, input_data)
                kernel_groups = tf.split(3, group, weights)
            except:
                input_data_groups = tf.split(input_data, group, 3)
                kernel_groups = tf.split(weights, group, 3)

            output_groups = [tf.nn.conv2d(i, k, strides, padding=pad) for i, k in zip(input_data_groups, kernel_groups)]
            try:
                conv_op = tf.concat(3, output_groups)
            except:
                conv_op = tf.concat(output_groups, 3)

        if has_bias is True:
            conv_op_add_bias = tf.nn.bias_add(conv_op, biases)
        else:
            conv_op_add_bias = conv_op

        if is_activated is True:
            if batch_norm is True:
                if activation == 'leaky_relu':
                    conv_act = leaky_relu(_batch_norm(conv_op_add_bias, is_training, scope=scope))
                elif activation == 'relu':
                    conv_act = tf.nn.relu(_batch_norm(conv_op_add_bias, is_training, scope=scope))
            else:
                if activation == 'leaky_relu':
                    conv_act = leaky_relu(conv_op_add_bias)
                elif activation == 'relu':
                    conv_act = tf.nn.relu(conv_op_add_bias)
        elif is_activated is False:
            conv_act = conv_op_add_bias

        return conv_act


def maxpool2d(x, k, strides=1, pad='SAME', name=None):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding=pad, name=name)


def avgpool2d(x, k, strides=1, pad='SAME', name=None):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding=pad, name=name)


def _batch_norm(input_data, is_training, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type ## , scope=scope+'/batchnorn'
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=True, center=False,
                                                        updates_collections=None,
                                                        scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=False, center=False,
                                                        updates_collections=None, scope=scope, reuse=True)
                   )


'''
alex_net
'''


def alex_net(x, dropout, is_training, image_size, weight_decay):
    # .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    # .lrn(2, 2e-05, 0.75, name='norm1')
    # .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')

    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, image_size, image_size, 3])
    # print x.get_shape()

    # Convolution Layer 1
    conv1 = _conv_layer(x, kernel_shape=[11, 11, 3, 96], name='conv1', weight_decay=weight_decay,
                        is_training=is_training,
                        pad='VALID', strides=[1, 4, 4, 1], batch_norm=True, is_activated=True)
    norm1 = tf.nn.local_response_normalization(conv1, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75, name='norm1')

    # Max Pooling (down-sampling)
    pool1 = maxpool2d(norm1, k=3, strides=2, name='pool1', pad='VALID')

    '''
     .conv(5, 5, 256, 1, 1, group=2, name='conv2')
     .lrn(2, 2e-05, 0.75, name='norm2')
     .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
     '''

    # Convolution Layer 2
    conv2 = _conv_layer(pool1, kernel_shape=[5, 5, 96, 256], name='conv2', weight_decay=weight_decay,
                        is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], group=2, batch_norm=True,
                        is_activated=True)
    norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75, name='norm2')

    # Max Pooling (down-sampling)
    pool2 = maxpool2d(norm2, k=3, strides=2, name='pool2', pad='VALID')

    '''
     .conv(3, 3, 384, 1, 1, name='conv3')
     .conv(3, 3, 384, 1, 1, group=2, name='conv4')
     .conv(3, 3, 256, 1, 1, group=2, name='conv5')
     .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
     '''
    # Convolution Layer 3,4,5
    conv3 = _conv_layer(pool2, kernel_shape=[3, 3, 256, 384], name='conv3', weight_decay=weight_decay,
                        is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv4 = _conv_layer(conv3, kernel_shape=[3, 3, 384, 384], name='conv4', weight_decay=weight_decay,
                        is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], group=2, batch_norm=True,
                        is_activated=True)
    conv5 = _conv_layer(conv4, kernel_shape=[3, 3, 384, 256], name='conv5', weight_decay=weight_decay,
                        is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], group=2, batch_norm=True,
                        is_activated=True)

    # Max Pooling (down-sampling)
    pool3 = maxpool2d(conv5, k=3, strides=2, name='pool5', pad='VALID')

    '''
     .fc(4096, name='fc6')
     .fc(4096, name='fc7')
     .fc(1000, relu=False, name='fc8')
     .softmax(name='prob'))
    '''
    # Fully connected layer 1
    with tf.variable_scope('fc6') as scope:
        reshape = tf.reshape(pool3, [-1, 6 * 6 * 256])
        # dim = reshape.get_shape()[0].value
        # print 'dim fc1', dim

        weights = _variable_with_weight_decay('weights', shape=[6 * 6 * 256, 4096],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              wd=weight_decay)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))

        # Apply Dropout
        drop_fc1 = tf.nn.dropout(reshape, dropout)
        fc1 = leaky_relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training,
                                     scope=scope.name))  # name=scope.name)

    # Fully connected layer 2
    with tf.variable_scope('fc7') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              wd=weight_decay)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))

        # Apply Dropout
        drop_fc2 = tf.nn.dropout(fc1, dropout)
        fc2 = leaky_relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training,
                                     scope=scope.name))  # name=scope.name)

    # Output, class prediction
    with tf.variable_scope('init_softmax') as scope:
        weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              wd=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    # softmax
    return fc2, logits


'''
vgg16
'''


def vgg16(x, dropout, is_training, image_size, weight_decay):
    # .conv(3, 3, 64, 1, 1, name='conv1_1')
    # .conv(3, 3, 64, 1, 1, name='conv1_2')
    # .max_pool(2, 2, 2, 2, name='pool1')

    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, image_size, image_size, 3])  # 224x224x3

    # Convolution Layer 1
    conv1_1 = _conv_layer(x, kernel_shape=[3, 3, 3, 64], name='conv1_1', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv1_2 = _conv_layer(conv1_1, kernel_shape=[3, 3, 64, 64], name='conv1_2', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)

    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv1_2, k=2, strides=2, name='pool1')

    '''
    .conv(3, 3, 128, 1, 1, name='conv2_1')
    .conv(3, 3, 128, 1, 1, name='conv2_2')
    .max_pool(2, 2, 2, 2, name='pool2')
    '''

    # Convolution Layer 2
    conv2_1 = _conv_layer(pool1, kernel_shape=[3, 3, 64, 128], name='conv2_1', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv2_2 = _conv_layer(conv2_1, kernel_shape=[3, 3, 128, 128], name='conv2_2', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)

    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2_2, k=2, strides=2, name='pool2')

    '''
    .conv(3, 3, 256, 1, 1, name='conv3_1')
    .conv(3, 3, 256, 1, 1, name='conv3_2')
    .conv(3, 3, 256, 1, 1, name='conv3_3')
    .max_pool(2, 2, 2, 2, name='pool3')
    '''

    # Convolution Layer 3
    conv3_1 = _conv_layer(pool2, kernel_shape=[3, 3, 128, 256], name='conv3_1', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv3_2 = _conv_layer(conv3_1, kernel_shape=[3, 3, 256, 256], name='conv3_2', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv3_3 = _conv_layer(conv3_2, kernel_shape=[3, 3, 256, 256], name='conv3_3', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)

    # Max Pooling (down-sampling)
    pool3 = maxpool2d(conv3_3, k=2, strides=2, name='pool3')

    '''
    .conv(3, 3, 512, 1, 1, name='conv4_1')
    .conv(3, 3, 512, 1, 1, name='conv4_2')
    .conv(3, 3, 512, 1, 1, name='conv4_3')
    .max_pool(2, 2, 2, 2, name='pool4')
    '''

    # Convolution Layer 4
    conv4_1 = _conv_layer(pool3, kernel_shape=[3, 3, 256, 512], name='conv4_1', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv4_2 = _conv_layer(conv4_1, kernel_shape=[3, 3, 512, 512], name='conv4_2', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv4_3 = _conv_layer(conv4_2, kernel_shape=[3, 3, 512, 512], name='conv4_3', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)

    # Max Pooling (down-sampling)
    pool4 = maxpool2d(conv4_3, k=2, strides=2, name='pool4')

    '''
    .conv(3, 3, 512, 1, 1, name='conv5_1')
    .conv(3, 3, 512, 1, 1, name='conv5_2')
    .conv(3, 3, 512, 1, 1, name='conv5_3')
    .max_pool(2, 2, 2, 2, name='pool5')
    '''

    # Convolution Layer 4
    conv5_1 = _conv_layer(pool4, kernel_shape=[3, 3, 512, 512], name='conv5_1', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv5_2 = _conv_layer(conv5_1, kernel_shape=[3, 3, 512, 512], name='conv5_2', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)
    conv5_3 = _conv_layer(conv5_2, kernel_shape=[3, 3, 512, 512], name='conv5_3', weight_decay=weight_decay,
                          is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True, is_activated=True)

    # Max Pooling (down-sampling)
    pool5 = maxpool2d(conv5_3, k=2, strides=2, name='pool5')

    '''
    .fc(4096, name='fc6')
    .fc(4096, name='fc7')
    .fc(1000, relu=False, name='fc8')
    '''
    # Fully connected layer 1
    with tf.variable_scope('fc6') as scope:
        reshape = tf.reshape(pool5, [-1, 7 * 7 * 512])
        # dim = reshape.get_shape()[0].value
        # print 'dim fc1', dim

        weights = _variable_with_weight_decay('weights', shape=[7 * 7 * 512, 4096],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              wd=weight_decay)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))

        # Apply Dropout
        drop_fc1 = tf.nn.dropout(reshape, dropout)
        fc1 = leaky_relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training,
                                     scope=scope.name))  # name=scope.name)

    # Fully connected layer 2
    with tf.variable_scope('fc7') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              wd=weight_decay)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))

        # Apply Dropout
        drop_fc2 = tf.nn.dropout(fc1, dropout)
        fc2 = leaky_relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training,
                                     scope=scope.name))  # name=scope.name)

    # Output, class prediction
    with tf.variable_scope('init_softmax') as scope:
        weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              wd=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    # softmax
    return fc2, logits


'''
google_net
'''


def inception_module(input_data, is_training, weight_decay, output_shapes, names):
    # (self.feed('pool2_3x3_s2')
    # .conv(1, 1, 64, 1, 1, name='inception_3a_1x1'))
    # (self.feed('pool2_3x3_s2')
    #     .conv(1, 1, 96, 1, 1, name='inception_3a_3x3_reduce')
    #     .conv(3, 3, 128, 1, 1, name='inception_3a_3x3'))
    # (self.feed('pool2_3x3_s2')
    #     .conv(1, 1, 16, 1, 1, name='inception_3a_5x5_reduce')
    #    .conv(5, 5, 32, 1, 1, name='inception_3a_5x5'))
    # (self.feed('pool2_3x3_s2')
    #     .max_pool(3, 3, 1, 1, name='inception_3a_pool')
    #     .conv(1, 1, 32, 1, 1, name='inception_3a_pool_proj'))
    # (self.feed('inception_3a_1x1',
    #          'inception_3a_3x3',
    #          'inception_3a_5x5',
    #          'inception_3a_pool_proj')
    #    .concat(3, name='inception_3a_output')
    conv1_1 = _conv_layer(input_data, kernel_shape=[1, 1, input_data.get_shape()[-1], output_shapes[0]], name=names[0],
                          weight_decay=weight_decay, is_training=is_training, pad='SAME', strides=[1, 1, 1, 1],
                          batch_norm=True, is_activated=True)

    conv2_1 = _conv_layer(input_data, kernel_shape=[1, 1, input_data.get_shape()[-1], output_shapes[1]], name=names[1],
                          weight_decay=weight_decay, is_training=is_training, pad='SAME', strides=[1, 1, 1, 1],
                          batch_norm=True, is_activated=True)
    conv2_2 = _conv_layer(conv2_1, kernel_shape=[3, 3, output_shapes[1], output_shapes[2]], name=names[2],
                          weight_decay=weight_decay, is_training=is_training, pad='SAME', strides=[1, 1, 1, 1],
                          batch_norm=True, is_activated=True)

    conv3_1 = _conv_layer(input_data, kernel_shape=[1, 1, input_data.get_shape()[-1], output_shapes[3]], name=names[3],
                          weight_decay=weight_decay, is_training=is_training, pad='SAME', strides=[1, 1, 1, 1],
                          batch_norm=True, is_activated=True)
    conv3_2 = _conv_layer(conv3_1, kernel_shape=[5, 5, output_shapes[3], output_shapes[4]], name=names[4],
                          weight_decay=weight_decay, is_training=is_training, pad='SAME', strides=[1, 1, 1, 1],
                          batch_norm=True, is_activated=True)

    pool = maxpool2d(input_data, k=3, strides=1, name=names[5])
    conv4_1 = _conv_layer(pool, kernel_shape=[1, 1, input_data.get_shape()[-1], output_shapes[5]], name=names[6],
                          weight_decay=weight_decay, is_training=is_training, pad='SAME', strides=[1, 1, 1, 1],
                          batch_norm=True, is_activated=True)

    try:
        return tf.concat(concat_dim=3, values=[conv1_1, conv2_2, conv3_2, conv4_1], name=names[7])
    except:
        return tf.concat(axis=3, values=[conv1_1, conv2_2, conv3_2, conv4_1], name=names[7])


def google_net(x, is_training, image_size, weight_decay):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, image_size, image_size, 3])  # 224x224x3

    '''(self.feed('data')
         .conv(7, 7, 64, 2, 2, name='conv1_7x7_s2')
         .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
         .lrn(2, 2e-05, 0.75, name='pool1_norm1')
         .conv(1, 1, 64, 1, 1, name='conv2_3x3_reduce')
         .conv(3, 3, 192, 1, 1, name='conv2_3x3')
         .lrn(2, 2e-05, 0.75, name='conv2_norm2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2_3x3_s2')
    '''
    conv1_7x7_s2 = _conv_layer(x, kernel_shape=[7, 7, 3, 64], name='conv1_7x7_s2', weight_decay=weight_decay,
                               is_training=is_training, pad='SAME', strides=[1, 2, 2, 1], batch_norm=True,
                               is_activated=True)
    pool1_3x3_s2 = maxpool2d(conv1_7x7_s2, k=3, strides=2, name='pool1_3x3_s2')
    pool1_norm1 = tf.nn.local_response_normalization(pool1_3x3_s2, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75,
                                                     name='pool1_norm1')

    conv2_3x3_reduce = _conv_layer(pool1_norm1, kernel_shape=[1, 1, 64, 64], name='conv2_3x3_reduce',
                                   weight_decay=weight_decay, is_training=is_training, pad='SAME', strides=[1, 1, 1, 1],
                                   batch_norm=True, is_activated=True)
    conv2_3x3 = _conv_layer(conv2_3x3_reduce, kernel_shape=[3, 3, 64, 192], name='conv2_3x3', weight_decay=weight_decay,
                            is_training=is_training, pad='SAME', strides=[1, 1, 1, 1], batch_norm=True,
                            is_activated=True)
    conv2_norm2 = tf.nn.local_response_normalization(conv2_3x3, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75,
                                                     name='pool1_norm1')
    pool2_3x3_s2 = maxpool2d(conv2_norm2, k=3, strides=2, name='pool2_3x3_s2', pad='VALID')

    '''
    (self.feed('pool2_3x3_s2')
     .conv(1, 1, 64, 1, 1, name='inception_3a_1x1'))

    (self.feed('pool2_3x3_s2')
         .conv(1, 1, 96, 1, 1, name='inception_3a_3x3_reduce')
         .conv(3, 3, 128, 1, 1, name='inception_3a_3x3'))

    (self.feed('pool2_3x3_s2')
         .conv(1, 1, 16, 1, 1, name='inception_3a_5x5_reduce')
         .conv(5, 5, 32, 1, 1, name='inception_3a_5x5'))

    (self.feed('pool2_3x3_s2')
         .max_pool(3, 3, 1, 1, name='inception_3a_pool')
         .conv(1, 1, 32, 1, 1, name='inception_3a_pool_proj'))

    (self.feed('inception_3a_1x1', 
               'inception_3a_3x3', 
               'inception_3a_5x5', 
               'inception_3a_pool_proj')
         .concat(3, name='inception_3a_output')
    '''
    inception1 = inception_module(pool2_3x3_s2, is_training, weight_decay, output_shapes=[64, 96, 128, 16, 32, 32],
                                  names=['inception_3a_1x1', 'inception_3a_3x3_reduce', 'inception_3a_3x3',
                                         'inception_3a_5x5_reduce', 'inception_3a_5x5', 'inception_3a_pool',
                                         'inception_3a_pool_proj', 'inception_3a_output'])

    '''
    (self.feed('inception_3a_output')
         .conv(1, 1, 128, 1, 1, name='inception_3b_1x1'))
         
    (self.feed('inception_3a_output')
         .conv(1, 1, 128, 1, 1, name='inception_3b_3x3_reduce')
         .conv(3, 3, 192, 1, 1, name='inception_3b_3x3'))

    (self.feed('inception_3a_output')
         .conv(1, 1, 32, 1, 1, name='inception_3b_5x5_reduce')
         .conv(5, 5, 96, 1, 1, name='inception_3b_5x5'))

    (self.feed('inception_3a_output')
         .max_pool(3, 3, 1, 1, name='inception_3b_pool')
         .conv(1, 1, 64, 1, 1, name='inception_3b_pool_proj'))

    (self.feed('inception_3b_1x1', 
               'inception_3b_3x3', 
               'inception_3b_5x5', 
               'inception_3b_pool_proj')
         .concat(3, name='inception_3b_output')
         
    '''
    inception2 = inception_module(inception1, is_training, weight_decay, output_shapes=[128, 128, 192, 32, 96, 64],
                                  names=['inception_3b_1x1', 'inception_3b_3x3_reduce', 'inception_3b_3x3',
                                         'inception_3b_5x5_reduce', 'inception_3b_5x5', 'inception_3b_pool',
                                         'inception_3b_pool_proj', 'inception_3b_output'])

    '''
    .max_pool(3, 3, 2, 2, name='pool3_3x3_s2')
    '''
    pool3_3x3_s2 = maxpool2d(inception2, k=3, strides=2, name='pool3_3x3_s2')

    '''
    (self.feed('pool3_3x3_s2')
        .conv(1, 1, 192, 1, 1, name='inception_4a_1x1'))

    (self.feed('pool3_3x3_s2')
         .conv(1, 1, 96, 1, 1, name='inception_4a_3x3_reduce')
         .conv(3, 3, 208, 1, 1, name='inception_4a_3x3'))

    (self.feed('pool3_3x3_s2')
         .conv(1, 1, 16, 1, 1, name='inception_4a_5x5_reduce')
         .conv(5, 5, 48, 1, 1, name='inception_4a_5x5'))

    (self.feed('pool3_3x3_s2')
         .max_pool(3, 3, 1, 1, name='inception_4a_pool')
         .conv(1, 1, 64, 1, 1, name='inception_4a_pool_proj'))

    (self.feed('inception_4a_1x1', 
               'inception_4a_3x3', 
               'inception_4a_5x5', 
               'inception_4a_pool_proj')
         .concat(3, name='inception_4a_output')
         # Middle output not supported
         .avg_pool(5, 5, 3, 3, padding='VALID', name='loss1_ave_pool')
         .conv(1, 1, 128, 1, 1, name='loss1_conv')
         .fc(1024, name='loss1_fc')
         .fc(1000, relu=False, name='loss1_classifier')
         .softmax(name='loss1_loss'))
    '''
    inception3 = inception_module(pool3_3x3_s2, is_training, weight_decay, output_shapes=[192, 96, 208, 16, 48, 64],
                                  names=['inception_4a_1x1', 'inception_4a_3x3_reduce', 'inception_4a_3x3',
                                         'inception_4a_5x5_reduce', 'inception_4a_5x5', 'inception_4a_pool',
                                         'inception_4a_pool_proj', 'inception_4a_output'])

    '''
    (self.feed('inception_4a_output')
         .conv(1, 1, 160, 1, 1, name='inception_4b_1x1'))

    (self.feed('inception_4a_output')
         .conv(1, 1, 112, 1, 1, name='inception_4b_3x3_reduce')
         .conv(3, 3, 224, 1, 1, name='inception_4b_3x3'))

    (self.feed('inception_4a_output')
         .conv(1, 1, 24, 1, 1, name='inception_4b_5x5_reduce')
         .conv(5, 5, 64, 1, 1, name='inception_4b_5x5'))

    (self.feed('inception_4a_output')
         .max_pool(3, 3, 1, 1, name='inception_4b_pool')
         .conv(1, 1, 64, 1, 1, name='inception_4b_pool_proj'))

    (self.feed('inception_4b_1x1', 
               'inception_4b_3x3', 
               'inception_4b_5x5', 
               'inception_4b_pool_proj')
         .concat(3, name='inception_4b_output')
    '''
    inception4 = inception_module(inception3, is_training, weight_decay, output_shapes=[160, 112, 224, 24, 64, 64],
                                  names=['inception_4b_1x1', 'inception_4b_3x3_reduce', 'inception_4b_3x3',
                                         'inception_4b_5x5_reduce', 'inception_4b_5x5', 'inception_4b_pool',
                                         'inception_4b_pool_proj', 'inception_4b_output'])

    '''
    (self.feed('inception_4b_output')
         .conv(1, 1, 128, 1, 1, name='inception_4c_1x1'))
    (self.feed('inception_4b_output')
         .conv(1, 1, 128, 1, 1, name='inception_4c_3x3_reduce')
         .conv(3, 3, 256, 1, 1, name='inception_4c_3x3'))

    (self.feed('inception_4b_output')
         .conv(1, 1, 24, 1, 1, name='inception_4c_5x5_reduce')
         .conv(5, 5, 64, 1, 1, name='inception_4c_5x5'))

    (self.feed('inception_4b_output')
         .max_pool(3, 3, 1, 1, name='inception_4c_pool')
         .conv(1, 1, 64, 1, 1, name='inception_4c_pool_proj'))

    (self.feed('inception_4c_1x1', 
               'inception_4c_3x3', 
               'inception_4c_5x5', 
               'inception_4c_pool_proj')
         .concat(3, name='inception_4c_output')
    '''
    inception5 = inception_module(inception4, is_training, weight_decay, output_shapes=[128, 128, 256, 24, 64, 64],
                                  names=['inception_4c_1x1', 'inception_4c_3x3_reduce', 'inception_4c_3x3',
                                         'inception_4c_5x5_reduce', 'inception_4c_5x5', 'inception_4c_pool',
                                         'inception_4c_pool_proj', 'inception_4c_output'])

    '''
    (self.feed('inception_4c_output')
         .conv(1, 1, 112, 1, 1, name='inception_4d_1x1'))
    (self.feed('inception_4c_output')
         .conv(1, 1, 144, 1, 1, name='inception_4d_3x3_reduce')
         .conv(3, 3, 288, 1, 1, name='inception_4d_3x3'))

    (self.feed('inception_4c_output')
         .conv(1, 1, 32, 1, 1, name='inception_4d_5x5_reduce')
         .conv(5, 5, 64, 1, 1, name='inception_4d_5x5'))

    (self.feed('inception_4c_output')
         .max_pool(3, 3, 1, 1, name='inception_4d_pool')
         .conv(1, 1, 64, 1, 1, name='inception_4d_pool_proj'))

    (self.feed('inception_4d_1x1', 
               'inception_4d_3x3', 
               'inception_4d_5x5', 
               'inception_4d_pool_proj')
         .concat(3, name='inception_4d_output')
         .avg_pool(5, 5, 3, 3, padding='VALID', name='loss2_ave_pool')
         .conv(1, 1, 128, 1, 1, name='loss2_conv')
         .fc(1024, name='loss2_fc')
         .fc(1000, relu=False, name='loss2_classifier')
         .softmax(name='loss2_loss'))
    '''
    inception6 = inception_module(inception5, is_training, weight_decay, output_shapes=[112, 144, 288, 32, 64, 64],
                                  names=['inception_4d_1x1', 'inception_4d_3x3_reduce', 'inception_4d_3x3',
                                         'inception_4d_5x5_reduce', 'inception_4d_5x5', 'inception_4d_pool',
                                         'inception_4d_pool_proj', 'inception_4d_output'])

    '''
    (self.feed('inception_4d_output')
         .conv(1, 1, 256, 1, 1, name='inception_4e_1x1'))

    (self.feed('inception_4d_output')
         .conv(1, 1, 160, 1, 1, name='inception_4e_3x3_reduce')
         .conv(3, 3, 320, 1, 1, name='inception_4e_3x3'))

    (self.feed('inception_4d_output')
         .conv(1, 1, 32, 1, 1, name='inception_4e_5x5_reduce')
         .conv(5, 5, 128, 1, 1, name='inception_4e_5x5'))

    (self.feed('inception_4d_output')
         .max_pool(3, 3, 1, 1, name='inception_4e_pool')
         .conv(1, 1, 128, 1, 1, name='inception_4e_pool_proj'))

    (self.feed('inception_4e_1x1', 
               'inception_4e_3x3', 
               'inception_4e_5x5', 
               'inception_4e_pool_proj')
         .concat(3, name='inception_4e_output')
    '''
    inception7 = inception_module(inception6, is_training, weight_decay, output_shapes=[256, 160, 320, 32, 128, 128],
                                  names=['inception_4e_1x1', 'inception_4e_3x3_reduce', 'inception_4e_3x3',
                                         'inception_4e_5x5_reduce', 'inception_4e_5x5', 'inception_4e_pool',
                                         'inception_4e_pool_proj', 'inception_4e_output'])

    '''
    .max_pool(3, 3, 2, 2, name='pool4_3x3_s2')
    '''
    pool4_3x3_s2 = maxpool2d(inception7, k=3, strides=2, name='pool4_3x3_s2')

    '''
    (self.feed('pool4_3x3_s2')
         .conv(1, 1, 256, 1, 1, name='inception_5a_1x1'))
    (self.feed('pool4_3x3_s2')
         .conv(1, 1, 160, 1, 1, name='inception_5a_3x3_reduce')
         .conv(3, 3, 320, 1, 1, name='inception_5a_3x3'))

    (self.feed('pool4_3x3_s2')
         .conv(1, 1, 32, 1, 1, name='inception_5a_5x5_reduce')
         .conv(5, 5, 128, 1, 1, name='inception_5a_5x5'))

    (self.feed('pool4_3x3_s2')
         .max_pool(3, 3, 1, 1, name='inception_5a_pool')
         .conv(1, 1, 128, 1, 1, name='inception_5a_pool_proj'))

    (self.feed('inception_5a_1x1', 
               'inception_5a_3x3', 
               'inception_5a_5x5', 
               'inception_5a_pool_proj')
         .concat(3, name='inception_5a_output')
    '''
    inception8 = inception_module(pool4_3x3_s2, is_training, weight_decay, output_shapes=[256, 160, 320, 32, 128, 128],
                                  names=['inception_5a_1x1', 'inception_5a_3x3_reduce', 'inception_5a_3x3',
                                         'inception_5a_5x5_reduce', 'inception_5a_5x5', 'inception_5a_pool',
                                         'inception_5a_pool_proj', 'inception_5a_output'])

    '''
    (self.feed('inception_5a_output')
         .conv(1, 1, 384, 1, 1, name='inception_5b_1x1'))
    (self.feed('inception_5a_output')
         .conv(1, 1, 192, 1, 1, name='inception_5b_3x3_reduce')
         .conv(3, 3, 384, 1, 1, name='inception_5b_3x3'))

    (self.feed('inception_5a_output')
         .conv(1, 1, 48, 1, 1, name='inception_5b_5x5_reduce')
         .conv(5, 5, 128, 1, 1, name='inception_5b_5x5'))

    (self.feed('inception_5a_output')
         .max_pool(3, 3, 1, 1, name='inception_5b_pool')
         .conv(1, 1, 128, 1, 1, name='inception_5b_pool_proj'))

    (self.feed('inception_5b_1x1', 
               'inception_5b_3x3', 
               'inception_5b_5x5', 
               'inception_5b_pool_proj')
         .concat(3, name='inception_5b_output')
    '''
    inception9 = inception_module(inception8, is_training, weight_decay, output_shapes=[384, 192, 384, 48, 128, 128],
                                  names=['inception_5b_1x1', 'inception_5b_3x3_reduce', 'inception_5b_3x3',
                                         'inception_5b_5x5_reduce', 'inception_5b_5x5', 'inception_5b_pool',
                                         'inception_5b_pool_proj', 'inception_5b_output'])

    '''
     .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5_7x7_s1')
     .fc(1000, relu=False, name='loss3_classifier')
     .softmax(name='loss3_loss3'))
     '''
    pool5_7x7_s1 = avgpool2d(inception9, k=7, strides=1, pad='VALID', name='pool5_7x7_s1')
    # pool5_7x7_s1 = tf.Print(pool5_7x7_s1, [tf.shape(pool5_7x7_s1)], message='Shape of pool5_7x7_s1')

    # Output, class prediction
    with tf.variable_scope('init_softmax') as scope:
        reshape = tf.reshape(pool5_7x7_s1, [-1, 1024])

        weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              wd=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)

    # softmax
    return pool5_7x7_s1, logits


'''
RESNET
'''


def block(input_data, filter_output, is_training, weight_decay, diff_stride=1, bottleneck=True):
    filters_in = input_data.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed.
    # That is the case when bottleneck=False but when bottleneck is
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = (4 if bottleneck is True else 1)
    final_filter_out = m * filter_output

    shortcut = input_data  # branch 1

    input_data = _conv_layer(input_data, kernel_shape=[1, 1, filters_in, filter_output], name='a',
                             weight_decay=weight_decay,
                             is_training=is_training, pad='SAME', activation='relu',
                             strides=[1, diff_stride, diff_stride, 1],
                             batch_norm=True, is_activated=True, has_bias=False)

    input_data = _conv_layer(input_data, kernel_shape=[3, 3, filter_output, filter_output], name='b',
                             weight_decay=weight_decay,
                             is_training=is_training, pad='SAME', activation='relu', strides=[1, 1, 1, 1],
                             batch_norm=True,
                             is_activated=True, has_bias=False)

    input_data = _conv_layer(input_data, kernel_shape=[1, 1, filter_output, final_filter_out], name='c',
                             weight_decay=weight_decay,
                             is_training=is_training, pad='SAME', activation='relu', strides=[1, 1, 1, 1],
                             batch_norm=True,
                             is_activated=False, has_bias=False)

    if final_filter_out != filters_in or diff_stride != 1:
        shortcut = _conv_layer(shortcut, kernel_shape=[1, 1, filters_in, final_filter_out], name='shortcut',
                               weight_decay=weight_decay, is_training=is_training, pad='SAME', activation='relu',
                               strides=[1, diff_stride, diff_stride, 1], batch_norm=True, is_activated=False,
                               has_bias=False)

    return tf.nn.relu(input_data + shortcut)


def stack(input_data, filter_output, is_training, weight_decay, diff_stride, num_blocks):
    for n in xrange(num_blocks):
        with tf.variable_scope('block%d' % (n + 1)):
            input_data = block(input_data, filter_output, is_training, weight_decay,
                               diff_stride=(diff_stride if n == 0 else 1),
                               bottleneck=True)
    return input_data


def resnet(x, is_training, weight_decay, input_image_size, num_blocks=[3, 4, 6, 3]):
    # c = Config()
    # c['bottleneck'] = bottleneck
    # c['is_training'] = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
    # c['ksize'] = 3
    # c['stride'] = 1
    # c['use_bias'] = use_bias
    # c['fc_units_out'] = num_classes
    # c['num_blocks'] = num_blocks
    # c['stack_stride'] = 2

    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, input_image_size, input_image_size, 3])  # 224x224x3

    '''with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)'''
    input_data = _conv_layer(x, kernel_shape=[7, 7, 3, 64], name='scale1', weight_decay=weight_decay,
                             is_training=is_training,
                             pad='SAME', activation='relu', strides=[1, 2, 2, 1], batch_norm=True, is_activated=True,
                             has_bias=False)

    with tf.variable_scope('scale2'):
        '''x = _max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)'''
        input_data = maxpool2d(input_data, k=3, strides=3)
        input_data = stack(input_data, filter_output=64, is_training=is_training, weight_decay=weight_decay,
                           diff_stride=1,
                           num_blocks=num_blocks[0])

    with tf.variable_scope('scale3'):
        '''c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = stack(x, c)'''
        input_data = stack(input_data, filter_output=128, is_training=is_training, weight_decay=weight_decay,
                           diff_stride=2,
                           num_blocks=num_blocks[1])

    with tf.variable_scope('scale4'):
        '''c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)'''
        input_data = stack(input_data, filter_output=256, is_training=is_training, weight_decay=weight_decay,
                           diff_stride=2,
                           num_blocks=num_blocks[2])

    with tf.variable_scope('scale5'):
        '''c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)'''
        input_data = stack(input_data, filter_output=512, is_training=is_training, weight_decay=weight_decay,
                           diff_stride=2,
                           num_blocks=num_blocks[3])

    # post-net
    input_data = tf.reduce_mean(input_data, reduction_indices=[1, 2], name="avg_pool")

    with tf.variable_scope('init_softmax'):
        weights = _variable_with_weight_decay('weights', [input_data.get_shape()[1], NUM_CLASSES],
                                              ini=tf.truncated_normal_initializer(stddev=0.00004), wd=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.zeros_initializer)
        logits = tf.nn.xw_plus_b(input_data, weights, biases)

    return input_data, logits


'''
DENSENET
'''

'''
function add_transition(model, n_channels, nOutChannels, opt, last, pool_size)
  if opt.optMemory >= 3 then     
     model:add(nn.JoinTable(2))
  end

  model:add(cudnn.Spatialbatch_normalization(n_channels))
  model:add(cudnn.ReLU(true))      
  if last then
     model:add(cudnn.SpatialAveragePooling(pool_size, pool_size))
     model:add(nn.Reshape(n_channels))      
  else
     model:add(cudnn.SpatialConvolution(n_channels, nOutChannels, 1, 1, 1, 1, 0, 0))
     if opt.drop_rate > 0 then model:add(nn.Dropout(opt.drop_rate)) end
     model:add(cudnn.SpatialAveragePooling(2, 2))
  end      
end
'''


def add_transition(input_data, n_channels, filter_output, is_training, weight_decay, pool_size=None, last=False,
                   drop_rate=0.0,
                   scope=None):
    input_data = tf.nn.relu(_batch_norm(input_data, is_training, scope=scope))
    if last is True:
        input_data = avgpool2d(input_data, k=pool_size, pad='VALID', name='end_avg_pool')
        print int(n_channels)
        input_data = tf.reshape(input_data, shape=[-1, 1 * 1 * int(n_channels)])
    else:
        input_data = _conv_layer(input_data, kernel_shape=[1, 1, n_channels, filter_output], name='conv',
                                 weight_decay=weight_decay,
                                 is_training=is_training, pad='VALID', activation='relu', strides=[1, 1, 1, 1],
                                 batch_norm=False, is_activated=False, has_bias=False)
        if drop_rate > 0.0:
            input_data = tf.nn.dropout(input_data, drop_rate)
        input_data = avgpool2d(input_data, k=2, strides=2, pad='VALID', name='avg_pool')

    return input_data


'''
function dense_connect_layer_standard(n_channels, opt)
   local net = nn.Sequential()

   net:add(ShareGradInput(cudnn.Spatialbatch_normalization(n_channels), 'first'))
   net:add(cudnn.ReLU(true))   
   if opt.bottleneck then
      net:add(cudnn.SpatialConvolution(n_channels, 4 * opt.growth_rate, 1, 1, 1, 1, 0, 0))
      n_channels = 4 * opt.growth_rate
      if opt.drop_rate > 0 then net:add(nn.Dropout(opt.drop_rate)) end
      net:add(cudnn.Spatialbatch_normalization(n_channels))
      net:add(cudnn.ReLU(true))      
   end
   net:add(cudnn.SpatialConvolution(n_channels, opt.growth_rate, 3, 3, 1, 1, 1, 1))
   if opt.drop_rate > 0 then net:add(nn.Dropout(opt.drop_rate)) end

   return net
end
'''


def dense_connect_layer_standard(input_data, n_channels, growth_rate, is_training, weight_decay, bottleneck=True,
                                 drop_rate=0.0,
                                 scope=None):
    input_data = tf.nn.relu(_batch_norm(input_data, is_training, scope=scope))

    if bottleneck is True:
        # nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
        # net:add(cudnn.SpatialConvolution(n_channels, 4 * opt.growth_rate, 1, 1, 1, 1, 0, 0))
        with tf.variable_scope('bottleneck') as sp:
            input_data = _conv_layer(input_data, kernel_shape=[1, 1, n_channels, 4 * growth_rate], name='conv',
                                     weight_decay=weight_decay, is_training=is_training, pad='VALID', activation='relu',
                                     strides=[1, 1, 1, 1], batch_norm=False, is_activated=False, has_bias=False)
            n_channels = 4 * growth_rate

            if drop_rate > 0.0:
                input_data = tf.nn.dropout(input_data, drop_rate)
            input_data = tf.nn.relu(_batch_norm(input_data, is_training, scope=sp))

    # nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
    # net:add(cudnn.SpatialConvolution(n_channels, opt.growth_rate, 3, 3, 1, 1, 1, 1))
    input_data = _conv_layer(input_data, kernel_shape=[3, 3, n_channels, growth_rate], name='conv',
                             weight_decay=weight_decay,
                             is_training=is_training, pad='SAME', activation='relu', strides=[1, 1, 1, 1],
                             batch_norm=False,
                             is_activated=False, has_bias=False)

    if drop_rate > 0.0:
        input_data = tf.nn.dropout(input_data, drop_rate)

    return input_data


'''
function add_layer(model, n_channels, opt)
  if opt.optMemory >= 3 then
     model:add(nn.DenseConnectLayerCustom(n_channels, opt))
  else
     model:add(nn.Concat(2)
        :add(nn.Identity())
        :add(dense_connect_layer_standard(n_channels, opt)))      
  end
end
'''


def add_layer(input_data, n_channels, growth_rate, drop_rate, is_training, weight_decay, scope=None):
    aux = input_data
    out = dense_connect_layer_standard(input_data, n_channels, growth_rate, is_training, weight_decay, bottleneck=True,
                                       drop_rate=drop_rate, scope=scope)
    try:
        out_final = tf.concat([out, aux], 3)
    except:
        out_final = tf.concat(concat_dim=3, values=[out, aux])

    return out_final


'''
local function add_dense_block(model, n_channels, opt, N)
    for i = 1, N do 
        add_layer(model, n_channels, opt)
        n_channels = n_channels + opt.growth_rate
    end
return n_channels
'''


def add_dense_block(input_data, n_channels, growth_rate, drop_rate, is_training, weight_decay, num_blocks):
    for n in xrange(num_blocks):
        with tf.variable_scope('denseblock%d' % (n + 1)) as scope:
            input_data = add_layer(input_data, n_channels, growth_rate, drop_rate, is_training, weight_decay, scope)
            n_channels = n_channels + growth_rate
    return input_data, n_channels


def densenet(x, is_training, weight_decay, input_image_size, reduction, drop_rate, growth_rate,
             num_blocks=[6, 12, 24, 16]):
    x = tf.reshape(x, shape=[-1, input_image_size, input_image_size, 3])  # 224x224x3

    '''
    model:add(cudnn.SpatialConvolution(3, n_channels, 7,7, 2,2, 3,3))
    model:add(cudnn.Spatialbatch_normalization(n_channels))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
    '''
    input_data = _conv_layer(x, kernel_shape=[7, 7, 3, 64], name='dense0', weight_decay=weight_decay,
                             is_training=is_training,
                             pad='SAME', activation='relu', strides=[1, 2, 2, 1], batch_norm=True, is_activated=True,
                             has_bias=False)
    input_data = maxpool2d(input_data, k=3, strides=2)

    # --Dense-Block 1 and transition (56x56)
    # n_channels = add_dense_block(model, n_channels, opt, stages[1])
    # add_transition(model, n_channels, math.floor(n_channels*reduction), opt)
    # n_channels = math.floor(n_channels*reduction)
    with tf.variable_scope('dense1') as scope:
        input_data, n_channels = add_dense_block(input_data, n_channels=64, growth_rate=growth_rate,
                                                 drop_rate=drop_rate,
                                                 is_training=is_training, weight_decay=weight_decay,
                                                 num_blocks=num_blocks[0])
        input_data = add_transition(input_data, n_channels, math.floor(n_channels * reduction), is_training,
                                    weight_decay, last=False,
                                    drop_rate=drop_rate, scope=scope)
        n_channels = math.floor(n_channels * reduction)

    # --Dense-Block 2 and transition (28x28)
    # n_channels = add_dense_block(model, n_channels, opt, stages[2])
    # add_transition(model, n_channels, math.floor(n_channels*reduction), opt)
    # n_channels = math.floor(n_channels*reduction)
    with tf.variable_scope('dense2') as scope:
        input_data, n_channels = add_dense_block(input_data, n_channels, growth_rate=growth_rate, drop_rate=drop_rate,
                                                 is_training=is_training, weight_decay=weight_decay,
                                                 num_blocks=num_blocks[1])
        input_data = add_transition(input_data, n_channels, math.floor(n_channels * reduction), is_training,
                                    weight_decay, last=False,
                                    drop_rate=drop_rate, scope=scope)
        n_channels = math.floor(n_channels * reduction)

    # --Dense-Block 3 and transition (14x14)
    # n_channels = add_dense_block(model, n_channels, opt, stages[3])
    # add_transition(model, n_channels, math.floor(n_channels*reduction), opt)
    # n_channels = math.floor(n_channels*reduction)
    with tf.variable_scope('dense3') as scope:
        input_data, n_channels = add_dense_block(input_data, n_channels, growth_rate=growth_rate, drop_rate=drop_rate,
                                                 is_training=is_training, weight_decay=weight_decay,
                                                 num_blocks=num_blocks[2])
        input_data = add_transition(input_data, n_channels, math.floor(n_channels * reduction), is_training,
                                    weight_decay, last=False,
                                    drop_rate=drop_rate, scope=scope)
        n_channels = math.floor(n_channels * reduction)

    # --Dense-Block 4 and transition (7x7)
    # n_channels = add_dense_block(model, n_channels, opt, stages[4])
    # add_transition(model, n_channels, n_channels, opt, true, 7)
    with tf.variable_scope('dense4') as scope:
        input_data, n_channels = add_dense_block(input_data, n_channels, growth_rate=growth_rate, drop_rate=drop_rate,
                                                 is_training=is_training, weight_decay=weight_decay,
                                                 num_blocks=num_blocks[3])
        input_data = add_transition(input_data, n_channels, n_channels, is_training, weight_decay, pool_size=7,
                                    last=True,
                                    drop_rate=drop_rate, scope=scope)

    with tf.variable_scope('init_softmax'):
        weights = _variable_with_weight_decay('weights', [input_data.get_shape()[1], NUM_CLASSES],
                                              ini=tf.truncated_normal_initializer(stddev=0.00004), wd=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.zeros_initializer)
        logits = tf.nn.xw_plus_b(input_data, weights, biases)

    return input_data, logits


def loss_def(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


'''
Extract features from the trained network
'''


def test(sess, data, classes, n_input_data, batch_size, x, y, keep_prob, is_training, pred, acc_mean, step):
    all_predcs = []
    cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    true_count = 0.0

    for i in xrange(0,
                    (len(classes) / batch_size if len(classes) % batch_size == 0 else (len(classes) / batch_size) + 1)):
        bx = np.reshape(data[i * batch_size:min(i * batch_size + batch_size, len(classes))], (-1, n_input_data))
        by = classes[i * batch_size:min(i * batch_size + batch_size, len(classes))]

        preds_val, acc_mean_val = sess.run([pred, acc_mean],
                                           feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})
        true_count += acc_mean_val

        all_predcs = np.concatenate((all_predcs, preds_val))

        for j in xrange(len(preds_val)):
            cm_test[by[j]][preds_val[j]] += 1

    _sum = 0.0
    for i in xrange(len(cm_test)):
        _sum += (cm_test[i][i] / float(np.sum(cm_test[i])) if np.sum(cm_test[i]) != 0 else 0)

    print("---- Iter " + str(step) + " -- Time " + str(datetime.datetime.now().time()) +
          " -- Test: Overall Accuracy= " + "{:.6f}".format(true_count / float(len(classes))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
          " Confusion Matrix= " + np.array_str(cm_test).replace("\n", "")
          )


def feature_extraction(sess, data, classes, n_input_data, batch_size, x, y, keep_prob, is_training, features):
    # all_features = []
    first = True

    for i in xrange(0, (len(data) / batch_size if len(data) % batch_size == 0 else (len(data) / batch_size) + 1)):
        bx = np.reshape(data[i * batch_size:min(i * batch_size + batch_size, len(data))], (-1, n_input_data))
        by = classes[i * batch_size:min(i * batch_size + batch_size, len(data))]

        _features = sess.run(features, feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})

        if first is True:
            all_features = _features
            first = False
        else:
            all_features = np.concatenate((all_features, _features))

    return np.asarray(all_features)


def main_feature_extraction(batch_size, class_test, class_train, data_test, data_train, features, fold, is_training,
                            keep_prob, n_input_data, names_test, names_train, output_path, sess, test_index,
                            train_index, x, y):
    all_features_train = feature_extraction(sess, data_train, class_train, n_input_data, batch_size, x, y,
                                            keep_prob,
                                            is_training, features)
    all_features = feature_extraction(sess, data_test, class_test, n_input_data, batch_size, x, y, keep_prob,
                                      is_training, features)
    print BatchColors.OKGREEN + "Features extracted!" + BatchColors.ENDC
    assert len(all_features_train) == len(train_index)
    assert len(all_features) == len(test_index)
    feat_file = open(output_path + 'fold' + str(fold) + '/all_featuresTest_fold' + str(fold) + '.txt', 'w')
    # print all_features.shape
    # print all_features[0,:]
    for f in xrange(len(all_features)):
        feat_file.write(
            names_test[f] + ' ' + np.array_str(all_features[f, :]).replace("\n", " ").replace("[",
                                                                                              "").replace(
                "]", "") + '\n')
    feat_file.close()
    feat_file2 = open(output_path + 'fold' + str(fold) + '/all_features_train_fold' + str(fold) + '.txt',
                      'w')
    # print all_features_train.shape
    # print all_features_train[0,:]
    for f in xrange(len(all_features_train)):
        feat_file2.write(
            names_train[f] + ' ' + np.array_str(all_features_train[f, :]).replace("\n", " ").replace("[",
                                                                                                     "").replace(
                "]", "") + '\n')
    feat_file2.close()


def generate_ranking(batch_size, class_test, data_test, is_training, keep_prob, logits, n_input_data, names_test, sess,
                     x, y):
    all_logits = feature_extraction(sess, data_test, class_test, n_input_data, batch_size, x, y, keep_prob,
                                    is_training, logits)
    all_logits_softmax = softmax(all_logits)
    # print all_logits_softmax[0,:], all_logits_softmax[1,:]
    # all_logits_softmax[2,:], all_logits_softmax[3,:], all_logits_softmax[4,:]
    zipped = zip(class_test, names_test, all_logits_softmax[:, 1])
    zipped.sort(key=operator.itemgetter(2), reverse=True)
    # relevance_list = np.asarray([x[0] for x in zipped])
    relevance_list_name = np.asarray([x[1] for x in zipped])
    relevant = np.asarray([x[1] for x in zipped if x[0] == 1])
    # relevance_predict = np.asarray([x[2] for x in zipped])

    ''' TREC EVAL TEST '''
    # print relevance_list.shape, relevance_list_name.shape, relevant.shape, relevance_predict.shape
    # qid  iter  docno  rel
    # Q1046   0   PNGImages/dolphin/image_0041.png    0
    '''relFile = open(output_path+'fold'+str(fold)+'/relFile_fold'+str(fold)+'.txt', 'w')
                    for x in zipped:
                        relFile.write('Q01 0 '+x[1]+' '+str(x[0])+'\n')
                    relFile.close()
                    #qid iter   docno      rank  sim   run_id
                    #030  Q0  ZF08-175-870  0   4238   prise1
                    resultFile = open(output_path+'fold'+str(fold)+'/resultFile_fold'+str(fold)+'.txt', 'w')
                    rank = 0
                    for x in zipped:
                        resultFile.write('Q01 0 '+x[1]+' '+str(rank)+' '+str(x[2])+' alex_net\n')
                        rank += 1
                    resultFile.close()'''
    ''' TREC EVAL TEST '''

    for k in [50, 100, 250, 480]:
        # print "Calculating Average Precision@" + str(k)
        print apk(relevant, relevance_list_name, k=k)
        # print average_precision(relevance_list[:k])
    return all_logits


def generate_results(sess, x, y, keep_prob, is_training, logits, n_input_data, batch_size, test_data, test_classes,
                     names_testing):
    all_logits = feature_extraction(sess, test_data, test_classes, n_input_data, batch_size, x, y, keep_prob,
                                    is_training, logits)

    all_logits_softmax = softmax(all_logits)

    zipped = zip(test_classes, names_testing, all_logits_softmax[:, 1])
    zipped.sort(key=operator.itemgetter(2), reverse=True)
    relevance_list_name = np.asarray([x[1] for x in zipped])
    relevant = np.asarray([x[1] for x in zipped if x[0] == 1])

    for k in [50, 100, 250, 480]:
        print apk(relevant, relevance_list_name, k=k)


def fold_ranking(sess, x, y, keep_prob, is_training, n_input_data, logits, batch_size, test_classes, test_data,
                 names_testing, output_path):
    all_logits = feature_extraction(sess, test_data, test_classes, n_input_data, batch_size, x, y, keep_prob,
                                    is_training, logits)

    all_logits_softmax = softmax(all_logits)

    # print test_classes.shape
    # print names_testing.shape
    # print all_logits_softmax.shape

    zipped = zip(test_classes, names_testing, all_logits_softmax[:, 1])
    zipped.sort(key=operator.itemgetter(2), reverse=True)
    relevance_list_name = np.asarray([x[1] for x in zipped])
    relevant = np.asarray([x[1] for x in zipped if x[0] == 1])

    # print np.asarray(relevant).shape
    # print np.asarray(relevance_list_name).shape
    for k in [50, 100, 250, 480]:
        print apk(relevant, relevance_list_name, k=k)

    out_file = open(output_path + 'ME17MST_DIRSM_MultiBrasil_run4.1.txt', 'w')
    for i in xrange(len(relevance_list_name)):
        out_file.write(relevance_list_name[i] + "\n")
    out_file.close()

    return all_logits


'''
python classification.py /home/mediaeval17/DIRSM/ /home/mediaeval17/DIRSM/folds_224/
/home/mediaeval17/caffe-tensorflow/models/vgg16/vgg16.npy /home/mediaeval17/DIRSM/ 0.01 0.005 200 200000 vgg16
'''


def main():
    list_params = ['dataset_path', 'folds_path(for input_data or output)', 'models_path',
                   'output_path(for model, images, etc)',
                   'learningRate', 'weight_decay', 'batch_size', 'niter',
                   'process[finetuning|feature_extraction|cnn_ranking|testing]',
                   'net_type[alex_net|vgg16|google_net|resnet[50|101|152]|densenet[121|169|201|161]]']
    if len(sys.argv) < len(list_params) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
    print_params(list_params)

    # training images path
    index = 1
    dataset_path = sys.argv[index]
    # folds path
    index = index + 1
    folds_path = sys.argv[index]
    # folds path
    index = index + 1
    models_path = sys.argv[index]
    # output path
    index = index + 1
    output_path = sys.argv[index]

    # Parameters
    index = index + 1
    lr_initial = float(sys.argv[index])
    index = index + 1
    weight_decay = float(sys.argv[index])
    index = index + 1
    batch_size = int(sys.argv[index])
    index = index + 1
    niter = int(sys.argv[index])
    index = index + 1
    process = sys.argv[index]
    index = index + 1
    net_type = sys.argv[index]
    input_image_size = (227 if net_type == 'alex_net' else 224)

    has_different_lr = True
    dynamic_norm = True

    # PROCESS IMAGES
    img_names, img_classes = read_csv_file(dataset_path + 'devset_images_gt.csv')
    print len(img_names), len(img_classes)
    data, classes, names = load_images(dataset_path + 'devset_images/', img_names, img_classes,
                                       resize_to=input_image_size)
    img_names, img_classes = np.asarray(img_names), np.asarray(img_classes)
    print img_names.shape, img_classes.shape, data.shape, classes.shape, names.shape

    if 'testing' in process:
        test_img_names, test_img_classes = read_csv_file(dataset_path + 'test/testset_images_gt.csv',
                                                         is_test=False)
        # read_csv_file(dataset_path+'test/testset_images.csv', is_test=True)
        print len(test_img_names), len(test_img_classes)
        test_data, test_classes, names_testing = load_images(dataset_path + 'test/testset_images/', test_img_names,
                                                             test_img_classes, resize_to=input_image_size,
                                                             is_test=False)
        # load_images(dataset_path+'test/testset_images/', test_img_names, img_classes,resize_to=input_image_size,True)
        test_img_names, test_img_classes = np.asarray(test_img_names), np.asarray(test_img_classes)
        print test_img_names.shape, test_img_classes.shape, test_data.shape, test_classes.shape, names_testing.shape

    fold = 0
    # skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    # for train_index, test_index in skf.split(data, classes):
    first = True
    for cur_fold in xrange(1, 6):
        fold += 1
        # print folds_path+'fold'+str(fold)+'/trainData_fold'+str(fold)+'.npy'
        # print os.path.isfile(folds_path+'fold'+str(fold)+'/trainData_fold'+str(fold)+'.npy')

        if os.path.isfile(folds_path + 'fold' + str(fold) + '/trainIndex_fold' + str(fold) + '.npy'):
            # TRAIN
            # data_train = np.load(folds_path+'fold'+str(fold)+'/trainData_fold'+str(fold)+'.npy')
            # class_train = np.load(folds_path+'fold'+str(fold)+'/trainLabel_fold'+str(fold)+'.npy')
            train_index = np.load(folds_path + 'fold' + str(fold) + '/trainIndex_fold' + str(fold) + '.npy')
            # TEST
            # data_test = np.load(folds_path+'fold'+str(fold)+'/testData_fold'+str(fold)+'.npy')
            # class_test = np.load(folds_path+'fold'+str(fold)+'/testLabel_fold'+str(fold)+'.npy')
            test_index = np.load(folds_path + 'fold' + str(fold) + '/testIndex_fold' + str(fold) + '.npy')
            print BatchColors.OKGREEN + "Folds loaded!" + BatchColors.ENDC
        else:
            # TRAIN
            # np.save(folds_path+'fold'+str(fold)+'/trainData_fold'+str(fold)+'.npy', data_train)
            # np.save(folds_path+'fold'+str(fold)+'/trainNames_fold'+str(fold)+'.npy', names_train)
            # np.save(folds_path+'fold'+str(fold)+'/trainLabel_fold'+str(fold)+'.npy', class_train)
            np.save(folds_path + 'fold' + str(fold) + '/trainIndex_fold' + str(fold) + '.npy', train_index)
            # TEST
            # np.save(folds_path+'fold'+str(fold)+'/testData_fold'+str(fold)+'.npy', data_test)
            # np.save(folds_path+'fold'+str(fold)+'/testNames_fold'+str(fold)+'.npy', names_test)
            # np.save(folds_path+'fold'+str(fold)+'/testLabel_fold'+str(fold)+'.npy', class_test)
            np.save(folds_path + 'fold' + str(fold) + '/testIndex_fold' + str(fold) + '.npy', test_index)
            print BatchColors.OKGREEN + "Folds saved!" + BatchColors.ENDC

        data_train, data_test = data[train_index], data[test_index]
        class_train, class_test = classes[train_index], classes[test_index]
        names_train, names_test = img_names[train_index], img_names[test_index]

        if not os.path.isfile(folds_path + 'fold' + str(fold) + '/trainNames_fold' + str(fold) + '.txt'):
            names_train_file = open(folds_path + 'fold' + str(fold) + '/trainNames_fold' + str(fold) + '.txt', 'w')
            for f in xrange(len(names_train)):
                names_train_file.write(names_train[f] + '\n')
            names_train_file.close()

            names_test_file = open(folds_path + 'fold' + str(fold) + '/testNames_fold' + str(fold) + '.txt', 'w')
            for f in xrange(len(names_test)):
                names_test_file.write(names_test[f] + '\n')
            names_test_file.close()

            print BatchColors.OKGREEN + "File names saved!" + BatchColors.ENDC

        # np.save(folds_path+'fold'+str(fold)+'/trainNames_fold'+str(fold)+'.npy', names_train)
        # np.save(folds_path+'fold'+str(fold)+'/testNames_fold'+str(fold)+'.npy', names_test)

        print data_train.shape, class_train.shape, names_train.shape
        print data_test.shape, class_test.shape, names_test.shape

        ###################
        epoch_number = 1000  # int(len(data_train)/batch_size) # 1 epoch = images / batch
        val_inteval = 1000  # int(len(data_train)/batch_size)
        display_step = 50  # math.ceil(int(len(training_classes)/batch_size)*0.01)
        ###################

        if dynamic_norm is True:
            mean_full, std_full = compute_image_mean(data_train)
            normalize_images(data_train, mean_full, std_full)
            normalize_images(data_test, mean_full, std_full)
            if 'testing' in process:
                print 'normalizing'
                normalize_images(test_data, mean_full, std_full)
        else:
            normalize_images(data_train, IMAGENET_MEAN_BGR, std_full=None)
            normalize_images(data_test, IMAGENET_MEAN_BGR, std_full=None)
            if 'testing' in process:
                normalize_images(test_data, IMAGENET_MEAN_BGR, std_full=None)

        # TRAIN NETWORK
        # Network Parameters
        n_input_data = input_image_size * input_image_size * 3  # RGB
        dropout = 0.5  # Dropout, probability to keep units

        # tf Graph input_data
        x = tf.placeholder(tf.float32, [None, n_input_data], name='data_input_data')
        y = tf.placeholder(tf.int32, [None], name='class_input_data')
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        is_training = tf.placeholder(tf.bool, [], name='is_training')
        global_step = tf.Variable(0, name='init_global_step', trainable=False)

        # CONVNET
        if net_type == 'alex_net':
            features, logits = alex_net(x, keep_prob, is_training, input_image_size, weight_decay)
            ignored = ['fc8']
        elif net_type == 'google_net':
            features, logits = google_net(x, is_training, input_image_size, weight_decay)
            ignored = ['loss1_ave_pool', 'loss1_conv', 'loss1_fc', 'loss1_classifier', 'loss1_loss', 'loss2_ave_pool',
                       'loss2_conv', 'loss2_fc', 'loss2_classifier', 'loss2_loss', 'loss3_loss3', 'loss3_classifier']
        elif net_type == 'vgg16':
            features, logits = vgg16(x, keep_prob, is_training, input_image_size, weight_decay)
            ignored = ['fc8']
        elif net_type == 'resnet50':
            features, logits = resnet(x, is_training, weight_decay, input_image_size, num_blocks=[3, 4, 6, 3])
        elif net_type == 'resnet101':
            features, logits = resnet(x, is_training, weight_decay, input_image_size, num_blocks=[3, 4, 23, 3])
        elif net_type == 'resnet152':
            features, logits = resnet(x, is_training, weight_decay, input_image_size, num_blocks=[3, 8, 36, 3])
        elif net_type == 'densenet121':
            features, logits = densenet(x, is_training, weight_decay, input_image_size, reduction=0.5, drop_rate=0.0,
                                        growth_rate=32, num_blocks=[6, 12, 24, 16])
        elif net_type == 'densenet169':
            features, logits = densenet(x, is_training, weight_decay, input_image_size, reduction=0.5, drop_rate=0.0,
                                        growth_rate=32, num_blocks=[6, 12, 32, 32])
        elif net_type == 'densenet201':
            features, logits = densenet(x, is_training, weight_decay, input_image_size, reduction=0.5, drop_rate=0.0,
                                        growth_rate=32, num_blocks=[6, 12, 48, 32])
        elif net_type == 'densenet161':
            features, logits = densenet(x, is_training, weight_decay, input_image_size, reduction=0.5, drop_rate=0.0,
                                        growth_rate=48, num_blocks=[6, 12, 36, 24])
        else:
            print BatchColors.FAIL + "Error: Cannot find identify network type: " + net_type + BatchColors.ENDC
            return

        # Define loss and optimizer
        if process == 'finetuning':
            loss = loss_def(logits, y)

            lr = tf.train.exponential_decay(lr_initial, global_step, 20000, 0.1, staircase=True)
            if has_different_lr is False:
                print BatchColors.WARNING + "All layers will be optimized with SAME learning rate." + BatchColors.ENDC
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss,
                                                                                                global_step=global_step)
            else:
                print BatchColors.WARNING + "Layers will be optimized with DIFFERENT learning rate." + BatchColors.ENDC
                if net_type == 'alex_net' or net_type == 'vgg16':
                    var_list1 = [k for k in tf.all_variables() if
                                 k.name.startswith('conv') or k.name.startswith('pool') or k.name.startswith(
                                     'fc')]
                elif net_type == 'google_net':
                    var_list1 = [k for k in tf.all_variables() if
                                 k.name.startswith('conv') or k.name.startswith('pool') or k.name.startswith(
                                     'inception')]
                elif net_type == 'resnet50' or net_type == 'resnet101' or net_type == 'resnet152':
                    var_list1 = [k for k in tf.all_variables() if k.name.startswith('scale')]
                elif net_type == 'densenet121':  # or net_type == 'resnet101' or net_type == 'resnet152':
                    var_list1 = [k for k in tf.all_variables() if k.name.startswith('dense')]
                var_list2 = [k for k in tf.all_variables() if k.name.startswith('init_')]  # ['init_softmax']

                opt1 = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
                opt2 = tf.train.MomentumOptimizer(learning_rate=lr * 10, momentum=0.9)

                grads = tf.gradients(loss, var_list1 + var_list2)
                grads1 = grads[:len(var_list1)]
                grads2 = grads[len(var_list1):]

                train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
                train_op2 = opt2.apply_gradients(zip(grads2, var_list2))

                optimizer = tf.group(train_op1, train_op2)

            # Evaluate model
            correct = tf.nn.in_top_k(logits, y, 1)
            # Return the number of true entries
            acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))
            pred = tf.argmax(logits, 1)
            # correct_pred = tf.equal(pred, tf.argmax(y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Initializing the variables
            # init = tf.initialize_all_variables()
            if net_type == 'alex_net' or net_type == 'vgg16' or net_type == 'google_net':
                init = tf.initialize_variables([k for k in tf.all_variables() if k.name.startswith(
                    'init') or 'Momentum' in k.name or 'moving_mean' in k.name or 'moving_variance' in k.name])
                # or 'batchnorn' in k.name
            elif 'densenet' in net_type or 'resnet' in net_type:
                init = tf.initialize_variables(
                    [k for k in tf.all_variables() if k.name.startswith('init') or 'Momentum' in k.name])
            else:
                init = tf.initialize_variables([k for k in tf.all_variables() if k.name.startswith('init')])

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()  # (max_to_keep=0)

            # Launch the graph
            shuffle = np.asarray(random.sample(xrange(len(class_train)), len(class_train)))

        if 'ckpt' in models_path and (net_type == 'resnet50' or net_type == 'resnet101' or net_type == 'resnet152'):
            saver_restore = tf.train.Saver(
                [k for k in tf.all_variables() if (k.name.startswith('scale') and 'Momentum' not in k.name)])
        else:
            saver_restore = tf.train.Saver()

        with tf.Session() as sess:
            if 'npy' in models_path:
                print BatchColors.OKGREEN + "Loading NPY model from " + models_path + BatchColors.ENDC
                load_npy(sess, models_path, ignore_params=ignored)
            elif 't7' in models_path:
                print BatchColors.OKGREEN + "Loading T7 model from " + models_path + BatchColors.ENDC
                load_t7(sess, models_path, tf.all_variables(), num_blocks=[6, 12, 24, 16])
            elif 'ckpt' in models_path:
                print BatchColors.OKGREEN + "Loading CKPT model from " + models_path + BatchColors.ENDC
                saver_restore.restore(sess, models_path)
            elif process == 'feature_extraction' or process == 'cnn_ranking' or 'testing' in process:
                print BatchColors.OKGREEN + "Loading final model from " + models_path + 'fold' + str(
                    fold) + '/model_final' + BatchColors.ENDC
                saver_restore.restore(sess, models_path + 'fold' + str(fold) + '/model_final')
            # saver_restore.restore(sess, models_path+'fold'+str(fold)+'/model-20000')
            else:
                print BatchColors.FAIL + "Error: Cannot load model: " + models_path + BatchColors.ENDC
                # return

            if process == 'finetuning':
                sess.run(init)
                it = 0
                # count_batch_balance = 0
                epoch_mean = 0.0
                epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
                batch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

                # Keep training until reach max iterations
                for step in xrange(1, niter + 1):
                    batch_x = np.reshape(
                        data_train[shuffle[it * batch_size:min(it * batch_size + batch_size, len(data_train))]],
                        (-1, n_input_data))
                    batch_y = class_train[shuffle[it * batch_size:min(it * batch_size + batch_size, len(data_train))]]
                    _, batch_loss, batch_correct, batch_predcs = sess.run([optimizer, loss, acc_mean, pred],
                                                                          feed_dict={x: batch_x, y: batch_y,
                                                                                     keep_prob: dropout,
                                                                                     is_training: True})

                    epoch_mean += batch_correct
                    for j in range(len(batch_predcs)):
                        epoch_cm_train[batch_y[j]][batch_predcs[j]] += 1

                    if step != 0 and step % display_step == 0:
                        # Calculate batch loss and accuracy
                        for j in range(len(batch_predcs)):
                            batch_cm_train[batch_y[j]][batch_predcs[j]] += 1

                        _sum = 0.0
                        for i in xrange(len(batch_cm_train)):
                            _sum += (batch_cm_train[i][i] / float(np.sum(batch_cm_train[i])) if np.sum(
                                batch_cm_train[i]) != 0 else 0)

                        print("Iter " + str(step) + " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
                              " Absolut Right Pred= " + str(int(batch_correct)) +
                              " Overall Accuracy= " + "{:.4f}".format(batch_correct / float(len(batch_y))) +
                              " Normalized Accuracy= " + "{:.4f}".format(_sum / float(NUM_CLASSES)) +
                              " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                              )
                        batch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

                    if step % epoch_number == 0:
                        _sum = 0.0
                        for i in xrange(len(epoch_cm_train)):
                            _sum += (epoch_cm_train[i][i] / float(np.sum(epoch_cm_train[i])) if np.sum(
                                epoch_cm_train[i]) != 0 else 0)

                        print("Iter " + str(step) + " -- Training Epoch:" +
                              " Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(len(class_train))) +
                              " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
                              " Confusion Matrix= " + np.array_str(epoch_cm_train).replace("\n", "")
                              )

                        epoch_mean = 0.0
                        epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

                    if step != 0 and step % val_inteval == 0:
                        # Test
                        saver.save(sess, output_path + 'fold' + str(fold) + '/model', global_step=step)
                        test(sess, data_test, class_test, n_input_data, batch_size, x, y, keep_prob, is_training, pred,
                             acc_mean, step)

                    if min(it * batch_size + batch_size, len(data_train)) == len(data_train) or len(
                            data_train) == it * batch_size + batch_size:
                        # print("New shuffle!!")
                        shuffle = np.asarray(random.sample(xrange(len(class_train)), len(class_train)))
                        it = -1
                    it += 1

                # FINISH
                print BatchColors.OKGREEN + "Optimization Finished!" + BatchColors.ENDC

                # Test: Final
                saver.save(sess, output_path + 'fold' + str(fold) + '/model_final')
                test(sess, data_test, class_test, n_input_data, batch_size, x, y, keep_prob, is_training, pred,
                     acc_mean,
                     step)

            elif process == 'feature_extraction':

                main_feature_extraction(batch_size, class_test, class_train, data_test, data_train, features, fold,
                                        is_training, keep_prob, n_input_data, names_test, names_train, output_path,
                                        sess, test_index, train_index, x, y)

            elif process == 'cnn_ranking':
                all_logits = generate_ranking(batch_size, class_test, data_test, is_training, keep_prob, logits,
                                              n_input_data, names_test, sess, x, y)

            elif process == 'testing_fold':
                all_logits = fold_ranking(sess, x, y, keep_prob, is_training, n_input_data, logits, batch_size,
                                          test_classes, test_data, names_testing, output_path)

            elif process == 'testing_all':
                all_logits_unique = feature_extraction(sess, test_data, test_classes, n_input_data, batch_size, x, y,
                                                       keep_prob,
                                                       is_training, logits)
                print all_logits_unique.shape
                if first is True:
                    all_logits = all_logits_unique
                    first = False
                else:
                    all_logits += all_logits_unique
                if cur_fold == 5:
                    all_logits = all_logits / 5.0
                    all_logits_softmax = softmax(all_logits)

                    zipped = zip(test_classes, names_testing, all_logits_softmax[:, 1])
                    zipped.sort(key=operator.itemgetter(2), reverse=True)

                    relevance_list_name = np.asarray([x[1] for x in zipped])
                    relevant = np.asarray([x[1] for x in zipped if x[0] == 1])
                    print relevance_list_name.shape

                    for k in [50, 100, 250, 480]:
                        print apk(relevant, relevance_list_name, k=k)

                    print 'writing to file... 1.1'
                    out_file = open(output_path + 'ME17MST_DIRSM_MultiBrasil_run1.1.txt', 'w')
                    for i in xrange(len(relevance_list_name)):
                        out_file.write(relevance_list_name[i] + "\n")
                    out_file.close()

            elif process == 'testing':
                generate_results(sess, x, y, keep_prob, is_training, logits, n_input_data, batch_size, test_data,
                                 test_classes, names_testing)

            else:
                print BatchColors.FAIL + "Error: Cannot identify process: " + process + BatchColors.ENDC
                return

        tf.reset_default_graph()


if __name__ == "__main__":
    main()
