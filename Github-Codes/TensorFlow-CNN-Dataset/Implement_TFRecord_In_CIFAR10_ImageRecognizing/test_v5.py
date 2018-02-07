

# In[5]:


import os
import ops as op
import tensorflow as tf
import kaggle_mnist_alexnet_model as model
import numpy
from PIL import Image



cwd = os.getcwd()
print(cwd)


image_channel = 3
label_cnt = 2



# In[6]:

classes = ['0', '1']
train_data_path = cwd + '/train_v4'

#image_size = 227



# This seems to be correct
#TRAIN_TFRECORD = "test_p0.tfrecords"
#image_size = 128
BATCH_SIZE = 15
BATCH_CAPACITY = 20
MIN_AFTER_DEQU = 5	



# Testing now
TRAIN_TFRECORD = "train_jason.tfrecords"
image_size = 128
BATCH_SIZE = 15
BATCH_CAPACITY = 20
MIN_AFTER_DEQU = 5	


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
    #img.set_shape([227 * 227 * 3])
 
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    
    return img, label



# In[ ]:


if __name__ == '__main__':
    # create TFRecord from /train
    #create_record()
    img, label = read_and_decode(TRAIN_TFRECORD)
    print(img.shape)
    print(label.shape)

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                   batch_size = BATCH_SIZE, capacity= BATCH_CAPACITY,
                                                   min_after_dequeue = MIN_AFTER_DEQU) 
         
    
    inputs, labels, dropout_keep_prob, learning_rate = model.input_placeholder(image_size, image_channel, label_cnt)
    logits = model.inference2(inputs, dropout_keep_prob, label_cnt)
	#logits = model.inference(inputs, dropout_keep_prob, label_cnt)
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
                for i in range(100):
                    
                    count += 1
                    
                    val, l= sess.run([img_batch, label_batch])
                  
			
                    labels_onehot = sess.run(tf.one_hot(l,label_cnt))
					
                    acc, _, loss_result = sess.run([accuracy, train, loss], feed_dict={inputs: val, labels:labels_onehot, dropout_keep_prob:0.8, learning_rate:0.001})
                    total_loss += loss_result
                    total_acc += acc
		             
                    #print('epoch ', i)
                  
                    #print('loss = %f , accuracy = %f' %(total_loss / count , total_acc / count))
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
                    val, l= sess.run([img_batch, label_batch])
                    labels_onehot = sess.run(tf.one_hot(l,label_cnt))
					
                    loss_result, Logits  = sess.run([loss, logits], feed_dict={inputs: val, labels:labels_onehot, dropout_keep_prob:0.9, learning_rate:0.001})
                    acc = accuracy.eval(session=sess, feed_dict={inputs: val, labels: labels_onehot , dropout_keep_prob: 0.9, learning_rate:0.001})
                    test_acc += acc	
                    print('logits = ', Logits)
                    #print('acc = ', acc)
                    print('label_batch = ', l)
                    if i % 10 == 0:                        
                        #print('logits = ', Logits)
                        #print('label = ', l)
                        #print('label_batch = ', l)
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


# In[ ]:




