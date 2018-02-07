
# coding: utf-8

# In[5]:


import os
import ops as op
import tensorflow as tf
import kaggle_mnist_alexnet_model as model
import numpy
from PIL import Image


#classes = ['0', '1']
cwd = os.getcwd()
print(cwd)

image_size = 227
image_channel = 3
label_cnt = 10




# In[6]:


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
train_data_path = cwd + '/train_v3'
TRAIN_TFRECORD = "train.tfrecords_v3"


# In[ ]:


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'img_class': tf.FixedLenFeature([], tf.int64),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)

    img = tf.reshape(img, [227, 227, 3])
    #img.set_shape([227 * 227 * 3])
 
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    img_class = tf.cast(features['img_class'], tf.int32)
    return img, label, img_class



# In[ ]:


if __name__ == '__main__':
    # create TFRecord from /train
    #create_record()
    img, label, img_class = read_and_decode(TRAIN_TFRECORD)
    print(img.shape)
    print(label.shape)

    img_batch, label_batch, img_class_batch = tf.train.shuffle_batch([img, label, img_class],
                                                   batch_size = 1, capacity= 100,
                                                   min_after_dequeue = 80)
        
    #img_batch, label_batch, img_class_batch = tf.train.shuffle_batch([img, label, img_class],
    #                                                batch_size=40, capacity=1000 + 3 * 40,
    #                                                min_after_dequeue=1000)
    
    #img_batch, label_batch, img_class_batch = tf.train.shuffle_batch([img, label, img_class],
    #                                                batch_size=4, capacity= 70,
    #                                                min_after_dequeue=10)
    
    ### OK with v2
    #img_batch, label_batch, img_class_batch = tf.train.shuffle_batch([img, label, img_class],
    #                                                batch_size=5, capacity=40,
    #                                                min_after_dequeue=30)
    
    inputs, labels, dropout_keep_prob, learning_rate = model.input_placeholder(image_size, image_channel, label_cnt)
    logits = model.inference(inputs, dropout_keep_prob, label_cnt)
    accuracy = model.accuracy(logits, labels)
    loss = model.loss(logits, labels)
    train = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss)
    
    
    #初始化所有的op
    init = tf.global_variables_initializer()
    
    
    
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
	#启动队列
        try:
            while not coord.should_stop():
                
				# train state
                print('train start')
                total_loss = 0
                total_acc = 0
                count = 0
                for i in range(300):
                    
                    count += 1
                    
                    val, l, img_Class= sess.run([img_batch, label_batch, img_class_batch])
                  
			
                    labels_onehot = sess.run(tf.one_hot(l,10))
					
                    acc, _, loss_result = sess.run([accuracy, train, loss], feed_dict={inputs: val, labels:labels_onehot, dropout_keep_prob:0.9, learning_rate:0.001})
                    total_loss += loss_result
                    total_acc += acc
					
                    if i % 50 == 0:
                        print('epoch ', i) 
                        print('loss = %f , accuracy = %f' %(total_loss / count , total_acc / count))
				
                print('train end')
				
				################################################
				
                print('test start')
                test_loss = 0
                test_acc = 0
                test_count = 0
                for i in range(50):
                    test_count += 1
                    val, l, img_Class= sess.run([img_batch, label_batch, img_class_batch])
					
                    labels_onehot = sess.run(tf.one_hot(l,10))
                    loss_result, Logits  = sess.run([loss, logits], feed_dict={inputs: val, labels:labels_onehot, dropout_keep_prob:1.0, learning_rate:0.001})
                    acc = accuracy.eval(session=sess, feed_dict={inputs: val, labels: labels_onehot , dropout_keep_prob: 1.0, learning_rate:0.001})
                    test_acc += acc
					
                    #print('acc = ', acc)
                    print('label_batch = ', l)
                    if i % 10 == 0:
                        #print('img = ', val)
                        #print('')
                        print('logits = ', Logits)
                        #print('label = ', l)
                        print('acc = ', acc)
                        #print('')
					
                print('accuracy = %f' % (test_acc / test_count))
                print('test end')
                
                break
        except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        
        
        coord.join(threads)
        sess.close()


# In[ ]:




