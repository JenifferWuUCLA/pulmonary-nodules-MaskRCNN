import tensorflow as tf

saver = tf.train.import_meta_graph('..\\model\\save_model.ckpt.meta')

graph = tf.get_default_graph()
a_val = graph.get_tensor_by_name('var/a_val:0')

y_output=graph.get_tensor_by_name('output:0')

with tf.Session() as sess:
    saver.restore(sess, '..\\model\\save_model.ckpt')
    print(sess.run(a_val))
