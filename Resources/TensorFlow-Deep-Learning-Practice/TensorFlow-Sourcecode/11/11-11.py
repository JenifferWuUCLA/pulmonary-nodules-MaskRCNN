import tensorflow as tf
a = tf.constant([[1,2],[3,4]])
matrix2 = tf.placeholder('float32',[2,2])
matrix1 = matrix2
sess = tf.Session()
a = sess.run(a)
print(sess.run(matrix1,feed_dict={matrix2:a}))
