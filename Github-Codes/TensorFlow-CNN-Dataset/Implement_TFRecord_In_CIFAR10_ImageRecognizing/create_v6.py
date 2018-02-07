import os
import tensorflow as tf
from PIL import Image
from TraceFileName import trace_filename as trace

'''
n01440764 --> 1:tench, Tinca tinca 
n01443537 --> 2:goldfish, Carassius auratus 
n01484850 --> 3:great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias 
n01491361 --> 4:tiger shark, Galeocerdo cuvieri 
n01494475 --> 5:hammerhead, hammerhead shark 
n01496331 --> 6:electric ray, crampfish, numbfish, torpedo 
n01498041 --> 7:stingray 
n01514668 --> 8:cock 
n01514859 --> 9:hen 
n01518878 --> 0:ostrich, Struthio camelus 

Each class has 500 images
'''

TRAIN_TFRECORD = "train.tfrecords_v6.2"
pace = 'train_v6/bigger_v2/'

def build_train_tfrecords():
    
    
    writer = tf.python_io.TFRecordWriter(TRAIN_TFRECORD)    
    file_list_with_path, file_list = trace(pace, '.JPEG')
    for index, file_with_path in enumerate(file_list_with_path):
 
        i = int(file_list[index][2:9])
		
		# decide class depending on ID
        if i == 1518878:
            Class = 0
        if i == 1440764:
            Class = 1
        if i == 1443537:
            Class = 2
        if i == 1484850:
            Class = 3
        if i == 1491361:
            Class = 4
        if i == 1494475:
            Class = 5
        if i == 1496331:
            Class = 6
        if i == 1498041:
            Class = 7
        if i == 1514668:
            Class = 8
        if i == 1514859:
            Class = 9
		
        print('index = %d, Class is %d'%(index, Class) )
		
        img = Image.open(file_with_path)
        img = img.resize((128, 128))	
        img_raw = img.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={ \
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[Class])), \
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])), \
                'name': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
            }))
        writer.write(example.SerializeToString())
        
    writer.close()
    print("Build train dataset success!")
	

if __name__ == '__main__':
    build_train_tfrecords()