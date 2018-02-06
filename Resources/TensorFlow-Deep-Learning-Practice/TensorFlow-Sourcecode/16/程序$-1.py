import tensorflow as tf

class LineRegModel:
    def __init__(self):
        self.a_val = tf.Variable(tf.random_normal([1]))
        self.b_val = tf.Variable(tf.random_normal([1]))
        self.x_input = tf.placeholder(tf.float32)
        self.y_label = tf.placeholder(tf.float32)
        self.y_output = tf.add(tf.mul(self.x_input, self.a_val), self.b_val)
        self.loss = tf.reduce_mean(tf.pow(self.y_output - self.y_label, 2))

    def get_op(self):
        return tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
