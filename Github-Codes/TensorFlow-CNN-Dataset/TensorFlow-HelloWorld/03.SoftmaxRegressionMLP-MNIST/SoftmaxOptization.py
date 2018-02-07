# coding: UTF-8
# TensorFlow实现Softmax Regression识别手写数字（多层感知机）
import tensorflow as tf

########加载数据集########
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建一个TensorFlow默认的InteractiveSession，这样后面执行各项操作就无须指定Session了。
sess = tf.InteractiveSession()

########给隐含层的参数设置Variable并进行初始化########
# 输入节点数
in_units = 784

# 隐含层输出节点数（在此模型中隐含层节点数设在200-1000范围内的结果区别不大）
h1_units = 300

# 隐含层的权重初始化为截断的正态分布，标准差为0.1，这一步可以通过tf.truncated_normal简单的实现
# 因为模型使用的是ReLU，所以需要使用正态分布给参数加一点噪声，来打破完全对称并且避免0梯度
w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))

# 隐含层的偏置都设置为0
b1 = tf.Variable(tf.zeros([h1_units]))

# 最后的输出层直接将权重和偏置全部初始化为0即可
w2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

########定义输入x的placeholder########
# Dropout的比率keep_prob（即保留节点的概率）是不一样的，通常在训练时小于1，而预测时等于1，所以也把Dropout的比率也作为计算图的输入，并定义成一个placeholder
x = tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)

########定义模型结构########
# 实现一个激活函数为ReLU的隐含层
hidden1 = tf.nn.relu(tf.matmul(x,w1)+b1)
# 实现Droopout功能，即随机将一部分节点设置为0，防止过拟合
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
# 计算Softmax每个分类的概率值
y = tf.nn.softmax(tf.matmul(hidden1_drop,w2) + b2)

########定义损失函数和选择优化器来优化loss########
# 对于多分类问题，通常使用cross-entropy作为loss function
y_ = tf.placeholder(tf.float32,[None,10]) 
# '''先定义一个placeholder，输入是真是的label，用来计算cross-entropy'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices = [1]))

# 选择自适应优化器Adagrad，并把学习速率设置为0.3
train_step = tf.train.AdagradOptimizer(0,3).minimize(cross_entropy)

########训练步骤########
# 使用TensorFlow的全局参数初始化器
tf.global_variables_initializer().run()

# （Dropout）这里与之前有些不同，我们加入了keep_prob作为计算图的输入，，并且在训练时设为保留75%的节点，其余节点置为0。
# 因为加了隐含层我们需要需要更多的训练迭代来优化模型参数。采用3000个batch，每个batch包含100条样本，一共30万的样本，相当于对全数据进行五轮迭代。
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys,keep_prob:0.75})

########模型准确率评测########
# tf.argmax是从一个tensor中寻找最大值的序号，tf.argmax(y,1)预测数字中概率最大那一个，tf.argmax(y_,1)找样本的真实数字类别
# tf.equal判断预测的数字类别是否就是正确的类别，最后返回计算分类是否正确的操作
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  

# 我们统计全样本预测accuracy，这里需要先用tf.cast将之前correct_prediction输出的bool值转换为float32，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 我们将测试数据的特征和label输入评测流程accuracy，计算模型在测试集的准确率，再将结果打印出来
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))


