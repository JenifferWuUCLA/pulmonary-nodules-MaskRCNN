import tensorflow as tf
import cv2
import Test2

filename = "train.tfrecords"
img,label = Test2.read_and_decode(filename)

img_batch,label_batch = tf.train.shuffle_batch([img,label],batch_size=1,
                                               capacity=10,
                                               min_after_dequeue=1)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
threads = tf.train.start_queue_runners(sess=sess)

for _ in range(10):
    val = sess.run(img_batch)
    label = sess.run(label_batch)
    val.resize((300,300,3))
    cv2.imshow("cool",val)
    cv2.waitKey()
    print(label)
