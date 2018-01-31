from __future__ import absolute_import, division, print_function

import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell


def build_data(n):
    xs = []
    ys = []
    for i in range(0, 2000):
        k = random.uniform(1, 50)

        x = [[np.sin(k + j)] for j in range(0, n)]
        y = [np.sin(k + n)]

        # x[i] = sin(k + i) (i = 0, 1, ..., n-1)
        # y[i] = sin(k + n)
        xs.append(x)
        ys.append(y)

    train_x = np.array(xs[0: 1500])
    train_y = np.array(ys[0: 1500])
    test_x = np.array(xs[1500:])
    test_y = np.array(ys[1500:])
    return (train_x, train_y, test_x, test_y)


length = 10
time_step_size = length
vector_size = 1
batch_size = 10
test_size = 10

# build data
(train_x, train_y, test_x, test_y) = build_data(length)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

X = tf.placeholder("float", [None, length, vector_size])
Y = tf.placeholder("float", [None, 1])

# get lstm_size and output predicted value
W = tf.Variable(tf.random_normal([10, 1], stddev=0.01))
B = tf.Variable(tf.random_normal([1], stddev=0.01))


def seq_predict_model(X, w, b, time_step_size, vector_size):
    # input X shape: [batch_size, time_step_size, vector_size]
    # transpose X to [time_step_size, batch_size, vector_size]
    X = tf.transpose(X, [1, 0, 2])
    # reshape X to [time_step_size * batch_size, vector_size]
    X = tf.reshape(X, [-1, vector_size])
    # split X, array[time_step_size], shape: [batch_size, vector_size]
    X = tf.split(X, time_step_size, 0)

    # LSTM model with state_size = 10
    cell = core_rnn_cell.BasicLSTMCell(num_units=10,
                                       forget_bias=1.0,
                                       state_is_tuple=True)
    outputs, _states = core_rnn.static_rnn(cell, X, dtype=tf.float32)

    # Linear activation
    return tf.matmul(outputs[-1], w) + b, cell.state_size


pred_y, _ = seq_predict_model(X, W, B, time_step_size, vector_size)
loss = tf.square(tf.subtract(Y, pred_y))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('./log', sess.graph)

    # train
    for i in range(50):
        # train
        for end in range(batch_size, len(train_x), batch_size):
            begin = end - batch_size
            x_value = train_x[begin: end]
            y_value = train_y[begin: end]
            sess.run(train_op, feed_dict={X: x_value, Y: y_value})

        # randomly select validation set from test set
        test_indices = np.arange(len(test_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0: test_size]
        x_value = test_x[test_indices]
        y_value = test_y[test_indices]

        # eval in validation set
        val_loss = np.mean(sess.run(loss,
                                    feed_dict={X: x_value, Y: y_value}))
        print('Run %s' % i, val_loss)

    for b in range(0, len(test_x), test_size):
        x_value = test_x[b: b + test_size]
        y_value = test_y[b: b + test_size]
        pred = sess.run(pred_y, feed_dict={X: x_value})
        for i in range(len(pred)):
            print(pred[i], y_value[i], pred[i] - y_value[i])
