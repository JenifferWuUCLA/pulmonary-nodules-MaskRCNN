import tensorflow as tf
import numpy as np

houses = 100
features = 2

#设计的模型为 2 * x1 + 3 * x2
x_data = np.zeros([100,2])
for house in range(houses):
    x_data[house,0] = np.round(np.random.uniform(50., 150.))
    x_data[house,1] = np.round(np.random.uniform(3., 7.))
weights = np.array([[2.],[3.]])
y_data = np.dot(x_data,weights)
print(y_data.shape)
x_data_ = tf.placeholder(tf.float32,[None,2])
weights_ = tf.Variable(np.ones([2,1]),dtype=tf.float32)
y_model = tf.matmul(x_data_,weights_)

loss = tf.reduce_mean(tf.pow((y_model - y_data),2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for _ in range(20):
    sess.run(train_op,feed_dict={x_data_:x_data})
    print(weights_.eval(sess))
