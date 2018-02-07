import tensorflow as tf
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

##简单读取
'''
for _example in tf.python_io.tf_record_iterator('pointer.tfrecords'):
    example=tf.train.Example()
    example.ParseFromString(_example)
    image=example.features.feature['image'].bytes_list.value
    label=example.features.feature['label'].int64_list.value
    #print type(image)
    print label



'''

##一旦生成了TFRecords文件，为了高效地读取数据，TF中使用队列读取数据
def read_and_decode(filename):#读入制作好的TFRecords文件
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回 文件名 和 文件
    features = tf.parse_single_example(serialized_example,
                                       features = {
                                        'label':tf.FixedLenFeature([],tf.int64),
                                        'img_raw':tf.FixedLenFeature([],tf.string)
                                       })#将image数据和label取出来
    img = tf.decode_raw(features['img_raw'], tf.uint8)#将被转化为二进制数据的image解码还原为图片
    img = tf.reshape(img, [100, 100, 3])#reshape为100x100的3通道图片
    img = tf.cast(img, tf.float32) * (1./255)#在流中抛出img张量
    label = tf.cast(features['label'],tf.int32)#在流中抛出label张量
    return img, label



img, label = read_and_decode('data.tfrecords')
img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size = 50, capacity = 2000, min_after_dequeue = 1000)

init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
init.run()
threads = tf.train.start_queue_runners(sess = sess)

lab = label_batch.eval()
print(lab)
print("done")



'''

with tf.Session() as sess:
    sess.run(init)
    threads=tf.train.start_queue_runners(sess=sess)
    for i in range(1):
        val,l=sess.run([img_batch,label_batch])        
        print l
        print val.shape
        print 'done'
    
'''   
    


