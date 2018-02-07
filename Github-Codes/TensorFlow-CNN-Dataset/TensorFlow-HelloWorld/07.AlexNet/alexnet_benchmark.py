# coding:UTF-8
# 每个batch的forward和backward进行速度测试
# 首先导入接下来会用到的几个系统库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# 共测试100个batch的数据
# batch_size = 32
# num_batches = 100

FLAGS = None

# 展示每一个卷积层或池化层输出的Tensor。
# 函数接受一个tensor作为输入，并显示其名称(t.op.name)和尺寸(t.get_shape().as_list())
print ('AlexNet的网络结构：')
def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


########设计AlexNet网络结构########
def inference(images):

  # images: Images Tensor
  # Reurn:
  # pool5: 返回最后一层（第五个池化层）
  # parameters: 返回AlexNet中所有需要训练的模型参数
  
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope: # 将scope内生成的Variable自动命名为conv1/xxx，区分不同卷积层之间的组件。
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, # 截断的正太分布初始化卷积核
                                             stddev=1e-1), name='weights') # 标准差0.1、尺寸11*11，颜色通道3，数量64
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME') # 对输入images完成卷积操作，strides为4*4
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope) # 非线性处理
    print_activations(conv1) # 将这层输出tensor conv1的结构打印出来
    parameters += [kernel, biases] # 将这层可训练的参数kernel、biases添加到parameters中

  # lrn1
  # 考虑到LRN层效果不明显，而且会让forward和backwood的速度大大下降所以没有添加

  # pool1
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1], # 这里池化尺寸为3*3，即将3*3的像素块降为1*1
                         strides=[1, 2, 2, 1], # 取样步长2*2，步长比池化尺寸小，提高特征的丰富性
                         padding='VALID', # 即取样时不能超过边框（SAME模式填充边界外的点）
                         name='pool1') # 
  print_activations(pool1) # 输出结果pool1的结构打印

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME') # 步长为1，即扫描全图
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2,
                         ksize=[1, 3, 3, 1], # 设置与上一层相同
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  print_activations(pool2)

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,                            
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME') # 步长为1，即扫描全图
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,                                    
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  print_activations(pool5)

  return pool5, parameters # 返回最后一层pool5的结果及所有需要训练的模型参数


########评估AlexNet每轮计算时间########
def time_tensorflow_run(session, target, info_string):
  
  # Args:
  # session:the TensorFlow session to run the computation under.
  # target:需要评测的运算算子。
  # info_string:测试名称。

  num_steps_burn_in = 10 # 先定义预热轮数（头几轮跌代有显存加载、cache命中等问题因此可以跳过，只考量10轮迭代之后的计算时间）
  total_duration = 0.0 # 记录总时间
  total_duration_squared = 0.0 # 总时间平方和  -----用来后面计算方差
  for i in xrange(FLAGS.num_batches + num_steps_burn_in): # 迭代轮数
    start_time = time.time() # 记录时间
    _ = session.run(target) # 每次迭代通过session.run(target)
    duration = time.time() - start_time # 
    if i >= num_steps_burn_in: 
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration  # 累加便于后面计算每轮耗时的均值和标准差
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches # 每轮迭代的平均耗时
  vr = total_duration_squared / FLAGS.num_batches - mn * mn # 
  sd = math.sqrt(vr) # 标准差
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


########定义主函数########
def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default(): # 定义默认的Graph方便后面使用
    image_size = 224 # 图片尺寸
    images = tf.Variable(tf.random_normal([FLAGS.batch_size, # batch大小
                                           image_size,
                                           image_size, 3], # 第四个维度是颜色通道数
                                          dtype=tf.float32,                          
                                          stddev=1e-1)) # 使用tf.random_normal函数构造正态分布（标准差0.1）的随机tensor
    # inference model.
    pool5, parameters = inference(images) # 使用前面定义的inference函数构建整个AlexNet网络（得到pool5和parameters集合）

    # Build an initialization operation.
    init = tf.global_variables_initializer()

    # 创建新的session并且通过上面一条语句初始化参数
    sess = tf.Session()
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, pool5, "Forward")

    # Add a simple objective so we can calculate the backward pass.设置优化目标loss
    objective = tf.nn.l2_loss(pool5)
    # Compute the gradient with respect to all the parameters.根据梯度更新参数
    grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")


def main(_):
  run_benchmark()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.'
      )
  parser.add_argument(
      '--num_batches',            
      type=int,
      default=100,
      help='Number of batches to run.'
      )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
          
