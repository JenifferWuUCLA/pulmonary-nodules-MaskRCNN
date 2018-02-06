#encoding=utf-8
import os
import tensorflow as tf
from PIL import Image

def create_record(cwd = "C:/cat_and_dog_r",classes = {'cat',"dog"},img_heigh=224,img_width=224):
    """
    :param cwd: 主文件夹 位置 ，所有分类的数据存储在这里
    :param classes:子文件夹 名称 ，每个文件夹的名称作为一个分类，由[1,2,3......]继续分下去
    :return:最终在当前位置生成一个tfrecords文件
    """
    writer = tf.python_io.TFRecordWriter("train.tfrecords") #最终生成的文件名
    for index, name in enumerate(classes):
        class_path = cwd +"/"+ name+"/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((img_heigh, img_width))
            img_raw = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
               }))
            writer.write(example.SerializeToString())
    writer.close()
