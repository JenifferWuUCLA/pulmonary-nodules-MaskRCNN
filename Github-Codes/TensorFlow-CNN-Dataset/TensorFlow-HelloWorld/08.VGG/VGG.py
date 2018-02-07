# --* coding: UTF-8 *--
# VGG通过反复堆叠3*3的小型卷积核和2*2的最大池化层，VGGNet成功的构建了16~19层深的卷积神经网络
# VGGNet评测forward（inference）耗时和backward（training）耗时
from datetime import datetime
import math
import time
import tensorflow as tf


########完成卷积层、全连接层和最大池化层的创建函数########
# conv_op函数用来创建卷积层并且把本层的参数存入参数列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    '''
    Args:
    input_op：输入的tensor
    name：这一层的名称
    kh：kernel height即卷积核的高
    kw：kernel weight即卷积核的宽
    n_out：卷积核数量即输出通道数
    dh：步长的高
    dw：步长的宽
    p：参数列表
    '''
    n_in = input_op.get_shape()[-1].value # 获取input_op的通道数

    with tf.name_scope(name) as scope: # 设置scope，生成的Variable使用默认的命名
        kernel = tf.get_variable(scope+"w",  # kernel（即卷积核参数）使用tf.get_variable创建
                                 shape=[kh, kw, n_in, n_out], # 【卷积核的高，卷积核的宽、输入通道数，输出通道数】
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d()) # 参数初始化
        # 使用tf.nn.conv2d对input_op进行卷积处理，卷积核kernel，步长dh*dw，padding模式为SAME
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME') 
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32) # biases使用tf.constant赋值为0
        biases = tf.Variable(bias_init_val, trainable=True, name='b') # 将bias_init_val转成可训练的参数
        z = tf.nn.bias_add(conv, biases) # 将卷积结果conv和bias相加
        activation = tf.nn.relu(z, name=scope) # 对z进行非线性处理得到activation
        p += [kernel, biases]  # 创建卷积层时用到的参数kernel和bias添加进参数列表
        return activation # 将卷积层的输出activation作为函数结果返回

# 定义全连接层的创建函数
def fc_op(input_op, name, n_out, p):  
    n_in = input_op.get_shape()[-1].value # 获取tensor的通道数

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", # 使用tf.get_variable创建全连接层的参数
                                 shape=[n_in, n_out], # 参数的维度有两个，输入通道数和输出通道数
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        # biases赋值0.1以避免dead neuron
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b') 
        # 对输入变量input_op和kernel做矩阵乘法并加上biases。再做非线性变换activation
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope) 
        p += [kernel, biases]
        return activation

# 定义最大池化层的创建函数
def mpool_op(input_op, name, kh, kw, dh, dw): 
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1], # 池化层尺寸kh*kw
                          strides=[1, dh, dw, 1], # 步长dh*dw
                          padding='SAME',
                          name=name)


########开始创建VGGNet-16的网络结构########
def inference_op(input_op, keep_prob):
    '''
    VGGNet-16的网络结构主要分为6个部分：前五段为卷积网络，最后一段是全连接网络。
    Args:
    input_op：输入Tensor
    keep_prob：控制Dropout的一个placeholder
    '''
    # 初始化参数列表p
    p = []
    # assume input_op shape is 224x224x3（第一个卷积层的输入input_op）

    # 创建第一段卷积网络 -- outputs 112x112x64
    # 两个卷积层的卷积核都是3*3，卷积核数量（输出通道数）均为64，步长1*1，全像素扫描。
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p) # outputs 224x224x64
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p) # outputs 224x224x64
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2) # 标准的2*2的最大池化-outputs 112x112x64

    # 创建第二段卷积网络 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

    # 创建第三段卷积网络 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)    
    pool3 = mpool_op(conv3_3,   name="pool3",   kh=2, kw=2, dh=2, dw=2)

    # 创建第四段卷积网络 -- outputs 14x14x512
    conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)

    # 创建第五段卷积网络 -- outputs 7x7x512
    conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)

    # 备注：VGGNet-16的每一段卷积网络都会将图像的边长缩小一半，但是将卷积输出通道数翻倍。
    # 第五段卷积输出的通道数不再增加。

    # flatten 将第五段卷积网络的输出结果进行扁平化
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
   
    # tf.reshape函数将每个样本化为长度7*7*512 = 25088的向量
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1") 

    # fully connected 隐含节点4096的全连接层
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8) # 得到分类输出概率
    predictions = tf.argmax(softmax, 1) # tf.argmax求输出概率最大类别
    return predictions, softmax, fc8, p
    
    
########评测函数########
def time_tensorflow_run(session, target, feed, info_string): # 与AlexNet非常相似，session参数一点点区别
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed) # 引入feed_dict方便后面传入keep_prob来控制Dropout层的保留比率
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batches, mn, sd))


# 评测主函数
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,   # 生成随机的图片224*224
                                               image_size,
                                               image_size, 3],
                                               dtype=tf.float32,
                                               stddev=1e-1)) # 标准差为0.1的正态分布的随机数

        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob) # 构建网络结构获得参数列表

        init = tf.global_variables_initializer() # 初始化全局参数

        '''
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)
        sess.run(init)
        '''
        sess = tf.Session() # 创建session
        sess.run(init)

        time_tensorflow_run(sess, predictions, {keep_prob:1.0}, "Forward") # 预测时节点保留率

        objective = tf.nn.l2_loss(fc8) # 计算VGGNet-16最后的全连接层的输出fc8的L2 loss
        grad = tf.gradients(objective, p) # 使用tf.gradients求相对于这个loss的所有模型参数的梯度
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, "Forward-backward") # 这里的target为求解梯度的操作grad

batch_size=32 # VGGNet-16模型的体积较大，如果使用较大的batch_size，GPU显存会不够用
num_batches=100
run_benchmark()

# VGGNet-16的计算复杂度相比AlexNet确实高了很多，不过准确率有了很大提升

      