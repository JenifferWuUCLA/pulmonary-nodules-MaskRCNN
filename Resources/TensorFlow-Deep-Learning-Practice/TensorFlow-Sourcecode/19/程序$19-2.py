import tensorflow.contrib.slim as slim
import tensorflow as tf

weight1 = slim.variable('weight1',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05)
                            )

weight2 = slim.variable('weight2',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05)
                            )

variables = slim.get_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(weight1))
    print("--------------------")
    print(sess.run(variables))
