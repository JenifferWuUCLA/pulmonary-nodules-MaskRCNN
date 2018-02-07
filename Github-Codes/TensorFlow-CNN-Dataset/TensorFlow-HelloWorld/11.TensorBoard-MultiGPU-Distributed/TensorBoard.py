# coding:UTF-8
# 可视化帮助我们理解、调试和优化我们设计的网络
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


max_steps=1000 # 最大的训练步数
learning_rate=0.001 # 学习速率
dropout=0.9 # 节点保留比率
data_dir='/tmp/tensorflow/mnist/input_data' # 设置MNIST数据下载地址
log_dir='/tmp/tensorflow/mnist/logs/mnist_with_summaries' # 汇总数据的日志存放路径-供TensorBoard展示

#----------------------------------------------------------------------------
  # Import data
mnist = input_data.read_data_sets(data_dir,one_hot=True)

sess = tf.InteractiveSession() # 创建tensorflow默认的session
  # Create a multilayer model.


#----------------------------------------------------------------------------
  # Input placeholders
  '''
  为了在tensorboard中展示节点名称，设计网络时使用with tf.name_scope限定命名空间，
  在这个with的所有节点都会被自动命名为input/xxx这样的格式。
  '''
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
# 将输入的一维数据变形为28*28的图片存储到另一个tensor
with tf.name_scope('input_reshape'):
  image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
  tf.summary.image('input', image_shaped_input, 10) # 使用tf.summary.image将图片数据汇总给tensorboard展示

'''
定义神经网络模型参数的初始化方法，权重依然使用我们常用的truncated_normal进行初始化，
偏置则赋值0.1。
'''
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''
定义对variable变量的数据汇总函数，计算出variable的mean、stddev、max和min，再对这些
标量数据使用tf.summary.scalar进行记录和汇总。
'''
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var) # 记录变量var的直方图数据

#-------------------------------------------------------------------------------------
# 设计一个MLP多层神经网络来训练数据，在每一层都会对模型参数进行汇总
# 因此定义创建一层神经网络并进行数据汇总的函数nn_layer
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """
  Args:
  input_tensor：输入数据
  input_dim：输入的维度
  output_dim：输出的维度
  layer_name：层名称
  act=tf.nn.relu：激活函数默认使用relu
  """
  with tf.name_scope(layer_name):
    # 先初始化这层神经网络的权重和偏置
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights) # 进行数据汇总
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases) # 进行数据汇总

    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases # 对输入做矩阵乘法并加偏置
      tf.summary.histogram('pre_activations', preactivate) # 将未激活的结果统计直方图
    activations = act(preactivate, name='activation') # 使用激活函数
    tf.summary.histogram('activations', activations) # 再统计一次直方图
    return activations
# 使用定义好的nn_layer创建一层神经网络。输入是图片尺寸（784=28*28），输出是隐藏节点数
hidden1 = nn_layer(x, 784, 500, 'layer1')
# 创建Dropout层
with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob) # 使用tf.summary.scalar记录 keep_prob
  dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  # 使用nn_layer定义神经网络的输出层，输入是上一层隐含节点数500，输出为类别数10
  # 激活函数为全等映射identity，即暂不使用softmax，在后面会处理
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
  '''
  使用tf.nn.softmax_cross_entropy_with_logits对前面输出层的结果进行softmax处理并计算交叉熵损失
  '''
  diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff) # 计算的平均损失
tf.summary.scalar('cross_entropy', cross_entropy) # 进行统计汇总

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(learning_rate).minimize( # 优化器对损失进行优化
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # 预测正确的样本数
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 计算正确率
tf.summary.scalar('accuracy', accuracy) # 统计汇总

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()  # 获取所有汇总操作   #
# 将session的计算图sess.graph加入训练过程的记录器，展示计算图
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph) # 文件记录器，存放训练的日志数据
test_writer = tf.summary.FileWriter(log_dir + '/test') # 文件记录器，存放测试的日志数据
tf.global_variables_initializer().run() # 初始化全部变量


def feed_dict(train): # 定义feed_dict的损失函数
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs, ys = mnist.train.next_batch(100) # 获取一个batch的样本值
    k = dropout # 设置dropout值
  else:
    xs, ys = mnist.test.images, mnist.test.labels # 获取测试数据
    k = 1.0 # 没有dropout效果
  return {x: xs, y_: ys, keep_prob: k}

#--------------------------------------------------------------------------------
# 实际执行具体的训练、测试及日志记录的操作
saver = tf.train.Saver()  # 创建模型的保存器
for i in range(max_steps):
  if i % 10 == 0:  # 每隔十步执行一次数据汇总和求测试集的准确率
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i) # 将汇总结果和循环步数i写入日志文件
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    if i % 100 == 99:  # Record execution stats # 每隔100步
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # 定义tensorflow运行选项
      run_metadata = tf.RunMetadata() # 定义tensorflow运行的元信息，这样可以记录训练时运行时间和内存占用等信息
      summary, _ = sess.run([merged, train_step], # 执行merged数据汇总操作和train_step训练操作
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata)
      # 将汇总结果summary和训练元信息run_metadata添加到train_writer
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
      saver.save(sess, log_dir+"/model.ckpt", i)
      print('Adding run metadata for', i)
    else:  # 平时只执行merged、train_step操作，并添加summary到train_writer。
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)
# 训练结束时关闭
train_writer.close()
test_writer.close()
'''
进入命令行：
# 执行tensorboard程序，指定tansorflow日志路径
$ tensorboard --logdir = /tmp/tensorflow/mnist/logs/mnist_with_summaries
# 执行上面之后会出现一条提示信息，复制其中网址到浏览器，就可以看到数据的可视化的图标了
'''



