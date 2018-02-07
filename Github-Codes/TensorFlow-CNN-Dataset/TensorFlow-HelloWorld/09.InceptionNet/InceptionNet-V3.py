# coding:UTF-8
import tensorflow as tf
from datetime import datetime
import math
import time

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev) #产生截断的正态分布


########定义函数生成网络中经常用到的函数的默认参数########
# 默认参数：卷积的激活函数、权重初始化方式、标准化器等
def inception_v3_arg_scope(weight_decay=0.00004,  # 设置L2正则的weight_decay
                           stddev=0.1, # 标准差默认值0.1
                           batch_norm_var_collection='moving_vars'):

  batch_norm_params = {  # 定义batch normalization（标准化）的参数字典
      'decay': 0.9997,  # 定义参数衰减系数
      'epsilon': 0.001,  
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  with slim.arg_scope([slim.conv2d, slim.fully_connected], # 给函数的参数自动赋予某些默认值
                      weights_regularizer=slim.l2_regularizer(weight_decay)): # 对[slim.conv2d, slim.fully_connected]自动赋值
  # 使用slim.arg_scope后就不需要每次都重复设置参数了，只需要在有修改时设置
    with slim.arg_scope( # 嵌套一个slim.arg_scope对卷积层生成函数slim.conv2d的几个参数赋予默认值
        [slim.conv2d],
        weights_initializer=trunc_normal(stddev), # 权重初始化器
        activation_fn=tf.nn.relu, # 激活函数
        normalizer_fn=slim.batch_norm, # 标准化器
        normalizer_params=batch_norm_params) as sc: # 标准化器的参数设置为前面定义的batch_norm_params
      return sc # 最后返回定义好的scope


########定义函数可以生成Inception V3网络的卷积部分########
def inception_v3_base(inputs, scope=None):
  '''
  Args:
  inputs：输入的tensor
  scope：包含了函数默认参数的环境
  '''
  end_points = {} # 定义一个字典表保存某些关键节点供之后使用

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], # 对三个参数设置默认值
                        stride=1, padding='VALID'):
      # 正式定义Inception V3的网络结构。首先是前面的非Inception Module的卷积层
      # 299 x 299 x 3
      # 第一个参数为输入的tensor，第二个是输出的通道数，卷积核尺寸，步长stride，padding模式
      net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3') # 直接使用slim.conv2d创建卷积层
      # 149 x 149 x 32
      '''
      因为使用了slim以及slim.arg_scope，我们一行代码就可以定义好一个卷积层
      相比AlexNet使用好几行代码定义一个卷积层，或是VGGNet中专门写一个函数定义卷积层，都更加方便
      '''
      net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
      # 147 x 147 x 32
      net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
      # 147 x 147 x 64
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
      # 73 x 73 x 64
      net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
      # 73 x 73 x 80.
      net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
      # 71 x 71 x 192.
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
      # 35 x 35 x 192.

      # 上面部分代码一共有5个卷积层，2个池化层，实现了对图片数据的尺寸压缩，并对图片特征进行了抽象

    '''
    三个连续的Inception模块组，三个Inception模块组中各自分别有多个Inception Module，这部分是Inception Module V3
    的精华所在。每个Inception模块组内部的几个Inception Mdoule结构非常相似，但是存在一些细节的不同
    '''
    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], # 设置所有模块组的默认参数
                        stride=1, padding='SAME'): # 将所有卷积层、最大池化、平均池化层步长都设置为1
      # mixed: 35 x 35 x 256.
      # 第一个模块组包含了三个结构类似的Inception Module
      with tf.variable_scope('Mixed_5b'): # 第一个Inception Module名称。Inception Module有四个分支
        with tf.variable_scope('Branch_0'): # 第一个分支64通道的1*1卷积
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'): # 第二个分支48通道1*1卷积，链接一个64通道的5*5卷积
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'): # 第四个分支为3*3的平均池化，连接32通道的1*1卷积
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) # 将四个分支的输出合并在一起（第三个维度合并，即输出通道上合并）

      '''
      因为这里所有层步长均为1，并且padding模式为SAME，所以图片尺寸不会缩小，但是通道数增加了。四个分支通道数之和
      64+64+96+32=256，最终输出的tensor的图片尺寸为35*35*256。
      第一个模块组所有Inception Module输出图片尺寸都是35*35，但是后两个输出通道数会发生变化。
      '''
      
      # mixed_1: 35 x 35 x 288.
      with tf.variable_scope('Mixed_5c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_2: 35 x 35 x 288.
      with tf.variable_scope('Mixed_5d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # 第二个Inception模块组。第二个到第五个Inception Module结构相似。
      # mixed_3: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1') # 图片会被压缩
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1') # 图片被压缩
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat([branch_0, branch_1, branch_2], 3) # 输出尺寸定格在17 x 17 x 768

      # mixed4: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7') # 串联1*7卷积和7*1卷积合成7*7卷积，减少了参数，减轻了过拟合
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'): 
          branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1') # 反复将7*7卷积拆分
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1') 
          branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_5: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6c'):
        with tf.variable_scope('Branch_0'):
          '''
          我们的网络每经过一个inception module，即使输出尺寸不变，但是特征都相当于被重新精炼了一遍，
          其中丰富的卷积和非线性化对提升网络性能帮助很大。
          '''
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # mixed_6: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_7: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6e'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points['Mixed_6e'] = net # 将Mixed_6e存储于end_points中，作为Auxiliary Classifier辅助模型的分类

      # 第三个inception模块组包含了三个inception module
      # mixed_8: 8 x 8 x 1280.
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3') # 压缩图片
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'): # 池化层不会对输出通道数产生改变
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat([branch_0, branch_1, branch_2], 3) # 输出图片尺寸被缩小，通道数增加，tensor的总size在持续下降中
      # mixed_9: 8 x 8 x 2048.
      with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) # 输出通道数增加到2048

      # mixed_10: 8 x 8 x 2048.
      with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      return net, end_points
      #Inception V3网络的核心部分，即卷积层部分就完成了
      '''
      设计inception net的重要原则是图片尺寸不断缩小，inception模块组的目的都是将空间结构简化，同时将空间信息转化为
      高阶抽象的特征信息，即将空间维度转为通道的维度。降低了计算量。Inception Module是通过组合比较简单的特征
      抽象（分支1）、比较比较复杂的特征抽象（分支2和分支3）和一个简化结构的池化层（分支4），一共四种不同程度的
      特征抽象和变换来有选择地保留不同层次的高阶特征，这样最大程度地丰富网络的表达能力。
      '''


########全局平均池化、Softmax和Auxiliary Logits########
def inception_v3(inputs,
                 num_classes=1000, # 最后需要分类的数量（比赛数据集的种类数）
                 is_training=True, # 标志是否为训练过程，只有在训练时Batch normalization和Dropout才会启用
                 dropout_keep_prob=0.8, # 节点保留比率
                 prediction_fn=slim.softmax, # 最后用来分类的函数
                 spatial_squeeze=True, # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
                 reuse=None, # 是否对网络和Variable进行重复使用
                 scope='InceptionV3'): # 包含函数默认参数的环境

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], # 定义参数默认值
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], # 定义标志默认值
                        is_training=is_training):
      # 拿到最后一层的输出net和重要节点的字典表end_points
      net, end_points = inception_v3_base(inputs, scope=scope) # 用定义好的函数构筑整个网络的卷积部分

      # Auxiliary Head logits作为辅助分类的节点，对分类结果预测有很大帮助
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'): # 将卷积、最大池化、平均池化步长设置为1
        aux_logits = end_points['Mixed_6e'] # 通过end_points取到Mixed_6e
        with tf.variable_scope('AuxLogits'):
          aux_logits = slim.avg_pool2d(
              aux_logits, [5, 5], stride=3, padding='VALID', # 在Mixed_6e之后接平均池化。压缩图像尺寸
              scope='AvgPool_1a_5x5')
          aux_logits = slim.conv2d(aux_logits, 128, [1, 1], # 卷积。压缩图像尺寸。
                                   scope='Conv2d_1b_1x1')

          # Shape of feature map before the final layer.
          aux_logits = slim.conv2d(
              aux_logits, 768, [5,5],
              weights_initializer=trunc_normal(0.01), # 权重初始化方式重设为标准差为0.01的正态分布
              padding='VALID', scope='Conv2d_2a_5x5')
          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001), # 输出变为1*1*1000
              scope='Conv2d_2b_1x1')
          if spatial_squeeze: # tf.squeeze消除tensor中前两个为1的维度。
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits # 最后将辅助分类节点的输出aux_logits储存到字典表end_points中

      # 处理正常的分类预测逻辑
      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                              scope='AvgPool_1a_8x8')
        # 1 x 1 x 2048
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, # 输出通道数1000
                             normalizer_fn=None, scope='Conv2d_1c_1x1') # 激活函数和规范化函数设为空
        if spatial_squeeze: # tf.squeeze去除输出tensor中维度为1的节点
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions') # Softmax对结果进行分类预测
  return logits, end_points # 最后返回logits和包含辅助节点的end_points






########评估AlexNet每轮计算时间########
def time_tensorflow_run(session, target, info_string):
  
  # Args:
  # session:the TensorFlow session to run the computation under.
  # target:需要评测的运算算子。
  # info_string:测试名称。

  num_steps_burn_in = 10 # 先定义预热轮数（头几轮跌代有显存加载、cache命中等问题因此可以跳过，只考量10轮迭代之后的计算时间）
  total_duration = 0.0 # 记录总时间
  total_duration_squared = 0.0 # 总时间平方和  -----用来后面计算方差
  for i in range(num_batches + num_steps_burn_in): # 迭代轮数
    start_time = time.time() # 记录时间
    _ = session.run(target) # 每次迭代通过session.run(target)
    duration = time.time() - start_time # 
    if i >= num_steps_burn_in: 
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration  # 累加便于后面计算每轮耗时的均值和标准差
      total_duration_squared += duration * duration
  mn = total_duration / num_batches # 每轮迭代的平均耗时
  vr = total_duration_squared / num_batches - mn * mn 
  sd = math.sqrt(vr) # 标准差
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, num_batches, mn, sd))

batch_size = 32 # 因为网络结构较大依然设置为32，以免GPU显存不够
height, width = 299, 299 # 图片尺寸
inputs = tf.random_uniform((batch_size, height, width, 3)) # 随机生成图片数据作为input
with slim.arg_scope(inception_v3_arg_scope()): # scope中包含了batch normalization默认参数，激活函数和参数初始化方式的默认值
  logits, end_points = inception_v3(inputs, is_training=False) # inception_v3中传入inputs获取里logits和end_points
  
init = tf.global_variables_initializer() # 初始化全部模型参数
sess = tf.Session() # 创建session
sess.run(init)  
num_batches=100 # 测试的batch数量
time_tensorflow_run(sess, logits, "Forward")
'''
虽然输入图片比VGGNet的224*224大了78%，但是forward速度却比VGGNet更快。
这主要归功于其较小的参数量，inception V3参数量比inception V1的700万
多了很多，不过仍然不到AlexNet的6000万参数量的一半。相比VGGNet的1.4
亿参数量就更少了。整个网络的浮点计算量为50亿次，比inception V1的15亿
次大了不少，但是相比VGGNet来说不算大。因此较少的计算量让inception V3
网络变得非常实用，可以轻松地移植到普通服务器上提供快速响应服务，甚至
移植到手机上进行实时的图像识别。
'''
'''
Inception V3中设计CNN的思想和Trick：
（1）Factorization into small convolutions很有效，可以降低参数量，减轻
过拟合，增加网络非线性的表达能力。
（2）卷积网络从输入到输出，应该让图片尺寸逐渐减小，输出通道数逐渐增加，
即让空间结构化，将空间信息转化为高阶抽象的特征信息。
（3）Inception Module用多个分支提取不同抽象程度的高阶特征的思路很有效，
可以丰富网络的表达能力。
'''
