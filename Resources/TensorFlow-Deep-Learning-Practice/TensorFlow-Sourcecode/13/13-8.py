import tensorflow as tf
data=tf.constant([
        [[3.0,2.0,3.0,4.0],
        [2.0,6.0,2.0,4.0],
        [1.0,2.0,1.0,5.0],
        [4.0,3.0,2.0,1.0]]
        ])
data = tf.reshape(data,[1,4,4,1])
maxPooling=tf.nn.max_pool(data, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

with tf.Session() as sess:
    print(sess.run(maxPooling))
