import tensorflow as tf

with tf.Session() as sess:
    q = tf.FIFOQueue(3,"float")
    init = q.enqueue_many(([0.1, 0.2, 0.3],))
    init2 = q.dequeue()
    init3 = q.enqueue(1.)

    sess.run(init)
    sess.run(init2)
    sess.run(init3)

    quelen =  sess.run(q.size())
    for i in range(quelen):
        print(sess.run(q.dequeue()))
