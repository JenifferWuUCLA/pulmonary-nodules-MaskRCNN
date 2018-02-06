import tensorflow.contrib.slim as slim
import tensorflow as tf

weight = tf.Variable(tf.ones([2,3]))
slim.add_model_variable(weight)
model_variables = slim.get_model_variables()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(model_variables))
