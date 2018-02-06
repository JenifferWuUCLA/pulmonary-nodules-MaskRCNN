import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_data = tf.placeholder("float32", [None, 784])
weight = tf.Variable(tf.ones([784, 10]))
bias = tf.Variable(tf.ones([10]))
y_model = tf.nn.softmax(tf.matmul(x_data, weight) + bias)
y_data = tf.placeholder("float32", [None, 10])

loss = tf.reduce_sum(tf.pow((y_model - y_data), 2))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x_data:batch_xs, y_data:batch_ys})
    if _ % 50 == 0:
        correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))
