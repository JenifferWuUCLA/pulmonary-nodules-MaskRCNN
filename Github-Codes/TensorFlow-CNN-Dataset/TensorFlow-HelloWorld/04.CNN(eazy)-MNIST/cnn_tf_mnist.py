# -*- coding: utf-8 -*-
'''卷积神经网络测试MNIST数据'''

# ########导入MNIST数据########
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 创建默认InteractiveSession
sess = tf.InteractiveSession()


# ########卷积网络会有很多的权重和偏置需要创建，先定义好初始化函数以便复用########
# 给权重制造一些随机噪声打破完全对称（比如截断的正态分布噪声，标准差设为0.1）
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 因为我们要使用ReLU，也给偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# #######卷积层、池化层接下来重复使用的，分别定义创建函数########
# tf.nn.conv2d是TensorFlow中的2维卷积函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 使用2*2的最大池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# #######正式设计卷积神经网络之前先定义placeholder########
# x是特征，y_是真实label。将图片数据从1D转为2D。使用tensor的变形函数tf.reshape
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# #######设计卷积神经网络########
# 第一层卷积
# 卷积核尺寸为5*5,1个颜色通道，32个不同的卷积核
W_conv1 = weight_variable([5, 5, 1, 32])
# 用conv2d函数进行卷积操作，加上偏置
b_conv1 = bias_variable([32])
# 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 对卷积的输出结果进行池化操作
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积（和第一层大致相同，卷积核为64，这一层卷积会提取64种特征）
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层。隐含节点数1024。使用ReLU激活函数
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了防止过拟合，在输出层之前加Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层。添加一个softmax层，就像softmax regression一样。得到概率输出。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# #######模型训练设置########
# 定义loss function为cross entropy，优化器使用Adam，并给予一个比较小的学习速率1e-4
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义评测准确率的操作
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# #######开始训练过程########
# 初始化所有参数
tf.global_variables_initializer().run()

# 训练（设置训练时Dropout的kepp_prob比率为0.5。mini-batch为50，进行2000次迭代训练，参与训练样本5万）
# 其中每进行100次训练，对准确率进行一次评测keep_prob设置为1，用以实时监测模型的性能
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print "-->step %d, training accuracy %.4f" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# 全部训练完成之后，在最终测试集上进行全面测试，得到整体的分类准确率
print "卷积神经网络在MNIST数据集正确率: %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
