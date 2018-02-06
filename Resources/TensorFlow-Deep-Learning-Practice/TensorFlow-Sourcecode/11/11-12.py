import tensorflow as tf
import numpy as np

houses = 100
features = 2

#设计的模型为 2 * x1 + 3 * x2
x_data = np.zeros([houses,2])
for house in range(houses):
    x_data[house,0] = np.round(np.random.uniform(50., 150.))
    x_data[house,1] = np.round(np.random.uniform(3., 7.))
weights = np.array([[2.],[3.]])
y_data = np.dot(x_data,weights)

x_data_ = tf.placeholder(tf.float32,[None,2])
y_data_ = tf.placeholder(tf.float32,[None,1])
weights_ = tf.Variable(np.ones([2,1]),dtype=tf.float32)
y_model = tf.matmul(x_data_,weights_)

loss = tf.reduce_mean(tf.pow((y_model - y_data_),2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(10):
    for x,y in zip(x_data,y_data):
        z1 = x.reshape(1,2)
        z2 = y.reshape(1,1)
        sess.run(train_op,feed_dict={x_data_:z1,y_data_:z2})
    print(weights_.eval(sess))
