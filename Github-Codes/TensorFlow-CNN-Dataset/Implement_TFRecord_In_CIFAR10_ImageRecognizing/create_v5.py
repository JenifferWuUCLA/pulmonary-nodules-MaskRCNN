import os
import tensorflow as tf
from PIL import Image
from TraceFileName import trace_filename as trace


#using n01514859 1000 images
#and   n01518878 1000 images


classes = ['0']
cwd = os.getcwd()
print(cwd)

train_data_path = cwd + '/train_v5'
TRAIN_TFRECORD = "train.tfrecords_v5"

def create_record():
    writer = tf.python_io.TFRecordWriter(TRAIN_TFRECORD)
	
    text_file = open("Output.txt", "w")
	
    i = 0
    for index, name in enumerate(classes):
        class_path = train_data_path + "/" + name 
        #print(index, ' ',  name)
        #print(class_path)
        for img_name in os.listdir(class_path):
            img_path = class_path + '/' + img_name
            img = Image.open(img_path)
            #img = img.resize((227, 227))
            img = img.resize((228, 228))
            img_raw = img.tobytes() 
            img_class = int(img_name[1:9])
			
            index_new = img_class % 2
            print('img_name is %s, label is %d' %(img_name, index_new))
           
            text_file.write("img_name is %s, id is %s, index is %s\n" % (img_name, i, index_new))
			
            example = tf.train.Example(features=tf.train.Features(feature={ \
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index_new])), \
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])), \
                "img_class": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_class])),   
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))				
            }))
            writer.write(example.SerializeToString())
            i += 1
        print('finish part ', index, ' TFReocrd writing')
    writer.close()	
    text_file.close()
	
def build_train_tfrecords():
    
    pace = '0/'
    writer = tf.python_io.TFRecordWriter("train_jason.tfrecords")    
    file_list_with_path, file_list = trace(pace, '.jpg')
    for index, file_with_path in enumerate(file_list_with_path):
        
        i = int(file_list[index][1:9])
        if (index%10) == 0:
            print(index)
        img = Image.open(file_with_path)
        img = img.resize((128, 128))	
        img_raw = img.tobytes()
        #print('label is ', i%2)
        example = tf.train.Example(features=tf.train.Features(feature={ \
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i%2])), \
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])), \
                'name': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
            }))
        writer.write(example.SerializeToString())
        
    writer.close()
    print("Build train dataset success!")
	

if __name__ == '__main__':
    build_train_tfrecords()