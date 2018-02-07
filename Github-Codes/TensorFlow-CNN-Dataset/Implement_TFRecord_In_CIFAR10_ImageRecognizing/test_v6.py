

# In[5]:


import os
import ops as op
import tensorflow as tf
import kaggle_mnist_alexnet_model as model
import numpy
from PIL import Image

image_channel = 3

#Indicates how many classes we identify
label_cnt = 10


TRAIN_TFRECORD = "train.tfrecords_v6.2"
image_size = 128

#Hyperparameter
BATCH_SIZE = 1
BATCH_CAPACITY = 20
MIN_AFTER_DEQU = 5	

#Depends on amount of image data 
train_step = 3000
test_step = 200


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string)
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [image_size, image_size, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    
    return img, label



if __name__ == '__main__':
    img, label = read_and_decode(TRAIN_TFRECORD)
   
    ## shuffle is useless right now
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                   batch_size = BATCH_SIZE, capacity= BATCH_CAPACITY,
                                                   min_after_dequeue = MIN_AFTER_DEQU) 
         
    
    inputs, labels, dropout_keep_prob, learning_rate = model.input_placeholder(image_size, image_channel, label_cnt)
    logits = model.inference2(inputs, dropout_keep_prob, label_cnt)
    accuracy = model.accuracy(logits, labels)
    loss = model.loss(logits, labels)
    train = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
				# train state
                print('train start')
                total_loss = 0
                total_acc = 0
                count = 0
				
                for i in range(train_step):
                    
                    count += 1
                    val, l= sess.run([img_batch, label_batch])          			
                    labels_onehot = sess.run(tf.one_hot(l,label_cnt))
					                  	
                    acc, _, loss_result = sess.run([accuracy, train, loss], feed_dict={inputs: val, labels:labels_onehot, dropout_keep_prob:0.9, learning_rate:0.002})
                    total_loss += loss_result
                    total_acc += acc
		                              
                    if i % 100 == 0:
                        print('epoch ', i) 
                        print('loss = %f , accuracy = %f' %(total_loss / count , total_acc / count))
				
                print('train end')
				
				################################################
				
                print('test start')
                test_loss = 0
                test_acc = 0
                test_count = 0
                for i in range(test_step):
				
                    test_count += 1
                    val, l= sess.run([img_batch, label_batch])
                    labels_onehot = sess.run(tf.one_hot(l,label_cnt))
                    loss_result, Logits  = sess.run([loss, logits], feed_dict={inputs: val, labels:labels_onehot, dropout_keep_prob:0.9, learning_rate:0.001})
                    acc = accuracy.eval(session=sess, feed_dict={inputs: val, labels: labels_onehot , dropout_keep_prob: 0.9, learning_rate:0.001})
                    test_acc += acc	
                  
                    #print('label_batch = ', l)
                    if i % 20 == 0:                        
                        print('logits = ', Logits)
                        print('label = ', l)
                        print('')
					
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




