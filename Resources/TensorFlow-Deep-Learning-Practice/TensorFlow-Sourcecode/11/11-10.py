import tensorflow as tf

matrix1 = tf.Variable(tf.ones([3,3]))
matrix2 = tf.Variable(tf.zeros([3,3]))
result = tf.matmul(matrix1,matrix2)

init=tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print(sess.run(matrix1))
print("--------------")
print(sess.run(matrix2))
