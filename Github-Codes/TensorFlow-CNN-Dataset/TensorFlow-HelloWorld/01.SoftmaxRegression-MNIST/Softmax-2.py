# -* coding:UTF-8 *-
# TensorFlow实现Softmax Regression识别手写数字
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

########加载数据########
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

########构建模型########
# 创建一个新的IteractiveSession，使用这个命令会将这个Session注册为默认的Session，
# 之后的运算也默认跑在这个Session里，不同Session之间的数据和运算应该是独立的
sess = tf.InteractiveSession()

# 创建placeholder，即输入数据的地方。第一个参数是数据类型，第二个参数代表tensor和shape，也就是数据尺寸。每条输入是784维的向量
x = tf.placeholder(tf.float32, [None, 784])

# 给Softmax Regression模型中的weights和biases创建Variable对象，Variable存储模型参数。
# 不同于存储数据的Tensor一旦使用掉就会消失，Variable在模型训练迭代中是持久化的（比如一直放在显存中），长期存在并且每轮迭代中被更新
# 这里W的shape是[784,10],784是特征的维数，10代表有十类，因为label在one-hot编码之后是10维的向量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 实现Softmax Regression算法
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 定义loss function。多分类问题通常使用cross-entropy作为loss function
# 先定义一个placeholder，输入是真实的label 
y_ = tf.placeholder("float", [None,10])

# 计算交叉熵。tf.reduce.mean是对每个batch数据结果求平均
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices = [1]))


# 定义优化算法。我们采用常见的随机梯度下降。根据反向传播算法进行训练，在每一轮迭代时更新参数来减小loss。
# 设置学习率0.5，优化目标设定为cross-entropy，得到进行训练操作的train_step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

########模型训练########
# 使用TensorFlow的全局参数初始化器
tf.global_variables_initializer().run()

# 每次使用小部分数据，会比全样本数据收敛速度快
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

########模型准确率评测########
# tf.argmax是从一个tensor中寻找最大值的序号，tf.argmax(y,1)预测数字中概率最大那一个，tf.argmax(y_,1)找样本的真实数字类别
# tf.equal判断预测的数字类别是否就是正确的类别，最后返回计算分类是否正确的操作
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 我们统计全样本预测accuracy，这里需要先用tf.cast将之前correct_prediction输出的bool值转换为float32，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 计算所学习到的模型在测试数据集上面的正确率。
print('Softmax Regression识别手写数字Model正确率：')
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


