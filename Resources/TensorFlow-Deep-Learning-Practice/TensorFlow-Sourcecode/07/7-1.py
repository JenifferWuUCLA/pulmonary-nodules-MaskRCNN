import tensorflow as tf
import numpy as np
inputX = np.random.rand(100)
inputY = np.multiply(3,inputX)  + 1
x = tf.placeholder("float32")
weight = tf.Variable(0.25)
bias = tf.Variable(0.25)
y = tf.mul(weight,x) + bias
y_ = tf.placeholder("float32")
loss = tf.reduce_sum(tf.pow((y - y_),2))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for _ in range(1000):
    sess.run(train_step,feed_dict={x:inputX,y_:inputY})
    if _%20 == 0:
        print("W的值为: ",weight.eval(session=sess),";  bias的值为: " ,bias.eval(session=sess))
