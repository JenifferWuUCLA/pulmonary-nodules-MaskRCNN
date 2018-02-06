import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.randn(10)
y_data = x_data * 0.3 + 0.15

weight = tf.Variable(0.5)
bias = tf.Variable(0.0)
y_model = weight * x_data + bias

loss = tf.pow((y_model - y_data),2)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for _ in range(200):
    sess.run(train_op)
    print(weight.eval(sess),bias.eval(sess))

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, sess.run(weight) * x_data + sess.run(bias), label='Fitted line')
plt.legend()
plt.show()
