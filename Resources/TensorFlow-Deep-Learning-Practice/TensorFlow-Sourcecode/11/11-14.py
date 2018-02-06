import tensorflow as tf
import numpy as np
import Test

def readFile(filename):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    col1, col2, col3, col4, col5 , col6 , col7 = tf.decode_csv(value,record_defaults=record_defaults)
    label = tf.pack([col1,col2])
    features = tf.pack([col3, col4, col5, col6, col7])
    example_batch, label_batch = tf.train.shuffle_batch([features,label],
                                batch_size=3, capacity=100, min_after_dequeue=10)
return example_batch,label_batch

example_batch,label_batch = Test.readFile(["cancer.txt"])

weight = tf.Variable(np.random.rand(5,1).astype(np.float32))
bias = tf.Variable(np.random.rand(2,1).astype(np.float32))
x_ = tf.placeholder(tf.float32, [None, 5])
y_model = tf.matmul(x_, weight) + bias
y = tf.placeholder(tf.float32, [2, 2])

loss = -tf.reduce_sum(y*tf.log(y_model))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    flag = 1
    while(flag):
        e_val, l_val = sess.run([example_batch, label_batch])
        sess.run(train, feed_dict={x_: e_val, y: l_val})
        if sess.run(loss,{x_: e_val, y: l_val}) <= 1:
            flag = 0
    print(sess.run(weight))
