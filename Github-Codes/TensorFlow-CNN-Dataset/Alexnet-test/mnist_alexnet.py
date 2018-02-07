from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#用于显示网络每一层网络的尺寸#
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def weight_variable(shape):#weight变量，制造一些随机噪声来打破完全对称
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):#bias变量，增加了一些小的正值（0.1），避免死亡节点
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])

parameters = []

# conv1
with tf.name_scope('conv1') as scope:#将scope内生成的Variable自动命名为conv1/xxx，便于区分不同卷积层之间的组件
    kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 8], dtype=tf.float32, stddev=1e-1), name='weights')#初始化卷积核参数
    conv = tf.nn.conv2d(x_image, kernel, [1, 2, 2, 1], padding='SAME')#对输入的images完成卷积操作，28x28图片，用5x5的卷积核，每两步一卷积，模式为Same，最后输出的是14x14的图片，计算方法参见网页收藏
    biases = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32), trainable=True, name='biases')#biases初始化为0
    bias = tf.nn.bias_add(conv, biases)#将参数conv和偏置biases加起来
    conv1 = tf.nn.relu(bias, name=scope)#用激活函数relu对结果进行非线性处理
    print_activations(conv1)#打印该层结构
    parameters += [kernel, biases]#将这一层可训练的参数kernel、biases添加到parameters中


# pool1
#lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1') #对前面输出的tensor conv1进行LRN处理 但会降低反馈速度
pool1 = tf.nn.max_pool(conv1,#最大池化处理
                       ksize=[1, 2, 2, 1],#池化尺寸
                       strides=[1, 2, 2, 1],#取样步长
                       padding='SAME',
                       name='pool1')
print_activations(pool1)

# conv2
with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 8, 32], dtype=tf.float32, stddev=1e-1), name='weights')#通道数为上一层卷积核数量(所提取的特征数)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
print_activations(conv2)

# pool2
#lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
pool2 = tf.nn.max_pool(conv2,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool2')
print_activations(pool2)

# conv3
with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([2, 2, 32, 64], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

# conv4
with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([2, 2, 64, 32], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

# conv5
with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([2, 2, 32, 32], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

# pool5
pool5 = tf.nn.max_pool(conv5,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool5')
print_activations(pool5)

flattened = tf.reshape(pool5, [-1, 2*2*32])#对pool5输出的tensor进行变形，将其转化为1D的向量，然后连接全连接层

# dense1
with tf.name_scope('dense1') as scope:
    W_dense1 = weight_variable([2*2*32, 128])#将6x6x256个神经元与4096个神经元(输出)全连接
    b_dense1 = bias_variable([128])
    dense1 = tf.nn.relu(tf.matmul(flattened, W_dense1) + b_dense1, name=scope)#当然也可以用上述卷积层用的bias_add函数
    parameters += [W_dense1, b_dense1]
    print_activations(dense1)


# drop1
drop1 = tf.nn.dropout(dense1, keep_prob, name='drop1')
print_activations(drop1)


# dense2
with tf.name_scope('dense2') as scope:
    W_dense2 = weight_variable([128, 128])
    b_dense2 = bias_variable([128])
    dense2 = tf.nn.relu(tf.matmul(drop1, W_dense2) + b_dense2, name=scope)
    parameters += [W_dense2, b_dense2]
    print_activations(dense2)

# drop2
drop2 = tf.nn.dropout(dense2, keep_prob, name='drop2')
print_activations(drop2)


# dense3(softmax)
with tf.name_scope('dense3') as scope:
    W_dense3 = weight_variable([128, 10])###最后输出为1000类
    b_dense3 = bias_variable([10])
    dense3 = tf.nn.softmax(tf.matmul(drop2, W_dense3) + b_dense3, name=scope)#当然也可以用上述卷积层用的bias_add函数
    parameters += [W_dense3, b_dense3]
    print_activations(dense3)

'''
#第一层 —— 卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层 —— 卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第三层 —— 全连接层 1024个隐含节点
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#第四层 —— Dropout层 减轻过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#第五层 —— Softmax层 得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
'''

'''定义损失函数cross entropy，优化器使用Adam'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(dense3), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

'''定义评测准确率的操作'''
correct_prediction = tf.equal(tf.argmax(dense3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#开始训练#
tf.global_variables_initializer().run()
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
