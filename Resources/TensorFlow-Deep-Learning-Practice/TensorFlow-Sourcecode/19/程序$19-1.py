import tensorflow.contrib.slim as slim
import tensorflow as tf

weight1 = slim.model_variable('weight1',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05))

weight2 = slim.model_variable('weight2',
                            shape=[2, 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05))

model_variables = slim.get_model_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(weight1))
    print("--------------------")
    print(sess.run(model_variables))
    print("--------------------")
print(sess.run(slim.get_variables_by_suffix("weight1")))
