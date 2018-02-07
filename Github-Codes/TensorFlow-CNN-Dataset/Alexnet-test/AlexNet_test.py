from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32  # 每轮迭代的样本数
num_batches = 100


# 用于显示网络每一层网络的尺寸#
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:  # 将scope内生成的Variable自动命名为conv1/xxx，便于区分不同卷积层之间的组件
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1),
                             name='weights')  # 初始化卷积核参数
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')  # 对输入的images完成卷积操作
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True,
                             name='biases')  # biases初始化为0
        bias = tf.nn.bias_add(conv, biases)  # 将参数conv和偏置biases加起来
        conv1 = tf.nn.relu(bias, name=scope)  # 用激活函数relu对结果进行非线性处理
        print_activations(conv1)  # 打印该层结构
        parameters += [kernel, biases]  # 将这一层可训练的参数kernel、biases添加到parameters中


        # pool1
    # lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1') #对前面输出的tensor conv1进行LRN处理 但会降低反馈速度
    pool1 = tf.nn.max_pool(conv1,  # 最大池化处理
                           ksize=[1, 3, 3, 1],  # 池化尺寸
                           strides=[1, 2, 2, 1],  # 取样步长
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1),
                             name='weights')  # 通道数为上一层卷积核数量(所提取的特征数)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    # pool2
    # lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

        # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

        # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
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

    return pool5, parameters


# 评估AlexNet每轮计算时间#
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():  # 定义默认的Graph方便后面使用
        image_size = 224
        # 随机生成图片tensor
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))

        # 用前面定义的inference函数构建整个网络，得到最后一层的输出和整个网络所需要训练的参数集合parameters
        pool5, parameters = inference(images)

        # 初始化所有参数
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # 评测前馈计算Forward
        time_tensorflow_run(sess, pool5, "Forward")

        # 评测反馈计算Backward
        objective = tf.nn.l2_loss(pool5)  # 计算损失loss
        grad = tf.gradients(objective, parameters)  # 求相对loss所有模型参数的梯度
        time_tensorflow_run(sess, grad, "Forward-backward")


run_benchmark()  # 执行主函数
