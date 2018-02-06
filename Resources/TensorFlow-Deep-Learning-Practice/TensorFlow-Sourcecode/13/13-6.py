import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread("img01.jpg")
img = np.array(img,dtype=np.float32)
x_image=tf.reshape(img,[1,1280,960,3])

filter = tf.Variable(tf.ones([7, 7, 3, 1]))

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    res = tf.nn.conv2d(x_image, filter, strides=[1, 2, 2, 1], padding='SAME')
    res_image = sess.run(tf.reshape(res,[640,480]))/128 + 1

cv2.imshow("lover",res_image.astype('uint8'))
cv2.waitKey()
