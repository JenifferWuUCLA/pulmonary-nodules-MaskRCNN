import tensorflow as tf

saver = tf.train.import_meta_graph('..\\model\\save_model.ckpt.meta')

#读取placeholder和最终的输出结果
graph = tf.get_default_graph()
a_val = graph.get_tensor_by_name('var/a_val:0')

input_placeholder=graph.get_tensor_by_name('input_placeholder:0')
labels_placeholder=graph.get_tensor_by_name('result_placeholder:0')
y_output=graph.get_tensor_by_name('output:0')#最终输出结果的tensor

with tf.Session() as sess:
    saver.restore(sess, '..\\model\\save_model.ckpt')#恢复权值
    result = sess.run(y_result, feed_dict={input_placeholder: [1]})
    print(result)
    print(sess.run(a_val))
