import tensorflow as tf

q = tf.FIFOQueue(1000,"float32")
counter = tf.Variable(0.0)
add_op = tf.assign_add(counter, tf.constant(1.0))
enqueueData_op = q.enqueue(counter)

sess = tf.Session()
qr = tf.train.QueueRunner(q, enqueue_ops=[add_op, enqueueData_op] * 2)
sess.run(tf.initialize_all_variables())
enqueue_threads = qr.create_threads(sess, start=True)  

coord = tf.train.Coordinator()  
enqueue_threads = qr.create_threads(sess, coord = coord,start=True)  

for i in range(0, 10):  
    print(sess.run(q.dequeue()))  
  
coord.request_stop()  
coord.join(enqueue_threads)  
