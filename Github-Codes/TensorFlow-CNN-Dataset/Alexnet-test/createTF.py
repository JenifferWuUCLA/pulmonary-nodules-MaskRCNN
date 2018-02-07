import os
import tensorflow as tf
from PIL import Image
import numpy as np
import random as rd

path = '/Users/qpj/Desktop/深度学习/AlexNet/data/'
i = 0

for file in os.listdir(path):
    if file != ".DS_Store":#该文件为OS系统自带的用于储存文件夹属性的隐藏性文件，若不排除则也会被重命名
        newname = '%d'%i
        os.rename(os.path.join(path,file),os.path.join(path,newname))
        i += 1
print("Rename finished!")

writer_train = tf.python_io.TFRecordWriter('data_train.tfrecords')#要生成的文件
writer_test = tf.python_io.TFRecordWriter('data_test.tfrecords')#要生成的文件

for j in range(i):
	img_path = path + str(j)        
	for per_img in os.listdir(img_path): 
		image = img_path+'/'+per_img#每一个图片的地址
		print(image)
		img = Image.open(image)#打开图片
		img = img.resize((224, 224))
		img_raw = img.tobytes()#将图片转化为二进制格式
		print(j)
		flag = rd.randint(1, 100)#随机产生1-100的整数，若小于等于75则做成训练集，大于75做成测试集（当然也可以每四张取一张做测试集，j%4==0即可）
		if flag <= 80:
			example_train = tf.train.Example(features=tf.train.Features(feature={
			  'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
			'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
			}))#example对象对label和image数据进行封装
			writer_train.write(example_train.SerializeToString())#序列化为字符串
		else:
			example_test = tf.train.Example(features=tf.train.Features(feature={
			  'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
			'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
			}))#example对象对label和image数据进行封装
			writer_test.write(example_test.SerializeToString())#序列化为字符串

        
writer_train.close()                
writer_test.close()
print("tfRecord finished！")
