import tensorflow as tf

input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.ones([3, 3, 5, 1]))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    conv2d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
    print(sess.run(conv2d))
