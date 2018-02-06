import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

threshold = 1.0e-2
x_data = np.random.randn(100).astype(np.float32)
y_data = x_data * 3 + 1

weight = tf.Variable(1.)
bias = tf.Variable(1.)
x_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y_model = tf.add(tf.mul(x_, weight), bias)

loss = tf.reduce_mean(tf.pow((y_model - y_),2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
flag = 1
while(flag):

    for (x,y) in zip(x_data,y_data):
        sess.run(train_op,feed_dict={x_:x,y_:y})
print(weight.eval(sess), bias.eval(sess))

    if sess.run(loss,feed_dict={x_:x_data,y_:y_data}) <= threshold:
        flag = 0

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, sess.run(weight) * (x_data) + sess.run(bias), label='Fitted line')
plt.legend()
plt.show()
