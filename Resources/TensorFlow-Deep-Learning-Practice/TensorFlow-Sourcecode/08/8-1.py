import tensorflow as tf

input1 = tf.constant(1)
print(input1)

input2 = tf.Variable(2,tf.int32)
print(input2)

input2 = input1
sess = tf.Session()
print(sess.run(input2))
