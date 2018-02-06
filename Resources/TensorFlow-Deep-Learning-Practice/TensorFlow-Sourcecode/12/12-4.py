import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_data = tf.placeholder("float32", [None, 784])
x_image = tf.reshape(x_data, [-1,28,28,1])

w_conv = tf.Variable(tf.ones([5,5,1,32]))
b_conv = tf.Variable(tf.ones([32]))
h_conv = tf.nn.relu(tf.nn.conv2d(x_image, w_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)

h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

w_fc = tf.Variable(tf.ones([14*14*32,1024]))
b_fc = tf.Variable(tf.ones([1024]))

h_pool_flat = tf.reshape(h_pool, [-1, 14*14*32])
h_fc = tf.nn.relu(tf.matmul(h_pool_flat, w_fc) + b_fc)

W_fc2 = tf.Variable(tf.ones([1024,10]))
b_fc2 = tf.Variable(tf.ones([10]))

y_model = tf.nn.softmax(tf.matmul(h_fc, W_fc2) + b_fc2)

y_data = tf.placeholder("float32", [None, 10])

loss = -tf.reduce_sum(y_data*tf.log(y_model))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(200)
    sess.run(train_step, feed_dict={x_data:batch_xs, y_data:batch_ys})
    if _ % 50 == 0:
        correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))
