import tensorflow as tf
import cv2

image_add_list = []
image_label_list = []
with open("train_list.csv") as fid:
    for image in fid.readlines():
        image_add_list.append(image.strip().split(",")[0])
        image_label_list.append(image.strip().split(",")[1])


def get_image(image_path):
    return tf.image.convert_image_dtype(
        tf.image.decode_jpeg(
            tf.read_file(image_path), channels=1),
        dtype=tf.uint8)

img = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file('jpg\\020.jpg'), channels=1),dtype=tf.float32)

with tf.Session() as sess:
    cv2Img = sess.run(img)
    img2 = cv2.resize(cv2Img, (200,200))
    cv2.imshow('image', img2)
    cv2.waitKey()
