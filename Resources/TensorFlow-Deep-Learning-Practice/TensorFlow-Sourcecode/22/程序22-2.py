import tensorflow.contrib.slim as slim
import tensorflow as tf

def discriminate(input_data, scope="discriminate",reuse=False,is_training = True):
    batch_norm_params = {  # batch normalization（标准化）的参数
        "is_training" : is_training,
        'decay': 0.9997,  # 参数衰减系数
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': ['moving_vars'],
            'moving_variance': ['moving_vars'],
        }
    }
    with tf.variable_scope(scope, 'discriminate', [input_data]):
        with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                            weights_regularizer=slim.l1_l2_regularizer()
                            ):

            conv1 = slim.conv2d(input_data,32,[3,3],padding="SAME",scope="d_conv1",reuse=reuse)
            conv2 = slim.conv2d(conv1, 64, [5, 5],padding="SAME",scope="d_conv2",reuse=reuse)
            conv3 = slim.conv2d(conv2, 32, [3, 3], padding="SAME", scope="d_conv3",reuse=reuse)
            out = slim.conv2d(conv3, 1, [28, 28], padding="VALID", scope="d_conv4", reuse=reuse)
        return out

def generate(batch_size,trainable = True, scope="generate",reuse=False,is_training = True):

    batch_norm_params = {  # 定义batch normalization（标准化）的参数字典
        "is_training": is_training,
        'decay': 0.9997,  # 定义参数衰减系数
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': ['moving_vars'],
            'moving_variance': ['moving_vars'],
        }
    }
	
    with tf.variable_scope(scope, 'generate', [batch_size]):
        with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                      weights_regularizer=slim.l2_regularizer(0.00005)):
            img_x = tf.random_normal([batch_size,28,28,1])
            conv1 = slim.conv2d(img_x,32,[3,3],padding="SAME",scope="g_conv1",trainable=trainable,reuse=reuse)
            conv2 = slim.conv2d(conv1, 64, [5, 5],padding="SAME",scope="g_conv2",trainable=trainable,reuse=reuse)
            conv3 = slim.conv2d(conv2,32,[3,3],padding="SAME",scope="g_conv3",trainable=trainable,reuse=reuse)
            conv4 = slim.conv2d(conv3, 1, [5, 5],padding="SAME",scope="g_conv4",trainable=trainable,reuse=reuse)
            out = tf.tanh(conv4,name="g_tanh")

    return out
