import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([3., 3.])

sess = tf.Session()
print(sess.run(tf.add(matrix1,matrix2)))
