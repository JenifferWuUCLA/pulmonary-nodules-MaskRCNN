import tensorflow as tf
import cv2

image_add_list = []
image_label_list = []
with open("train_list.csv") as fid:
    for image in fid.readlines():
        image_add_list.append(image.strip().split(",")[0])
        image_label_list.append(image.strip().split(",")[1])
img=tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file('jpg\\image_0000.jpg'),channels=1)
,dtype=tf.float32)
print(img)
