import os
import tensorflow as tf
from PIL import Image

path = "jpg"
filenames=os.listdir(path)
writer = tf.python_io.TFRecordWriter("train.tfrecords")

for name in os.listdir(path):
    class_path = path + os.sep + name
    for img_name in os.listdir(class_path):
        img_path = class_path+os.sep+img_name
        img = Image.open(img_path)
        img = img.resize((500,500))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[name])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
