# -*- coding: utf-8 -*-
# written by lqy


from skimage import io,transform
import glob
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"      #指定服务器上2号显卡进行计算
import tensorflow as tf
import numpy as np
import time

################需要自己设定的部分#####################
##  定义一些路径及训练参数
reload_net = 0              #是否预加载网络参数   1是0否
reload_net_path = 'save_model'        #预加载网络参数的存放位置
savenet = 1               #是否保存网络        1是0否
save_net_path = '/Users/qpj/Desktop/深度学习/AlexNet/save_model/'         #本次保存网络的位置

#path = '/Users/qpj/Desktop/深度学习/AlexNet/train_data/'      #数据的存放位置
model_dir = "save_model"        
model_name = "ckp"

#将所有的图片resize成100*100
w = 224             #图片尺寸
h = 224
c = 3               #图片通道数

n_epoch = 20000            #训练次数
batch_size = 50          #池大小

################需要自己设定的部分#####################

#用于显示网络每一层网络的尺寸#
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#读取图片
def read_img(path):  
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    print(cate)#打印分类图片文件夹路径
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h,c))
            print(img.shape)
            imgs.append(img)#向列表中添加元素
            cl_label=im.split('/')[-2]#将路径按‘/’分解成一个list，取list的倒数第二项，即图片分类文件夹的名字作为这一类图片的标签
            labels.append(cl_label)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data, label=read_img(path)
#print(data.shape, label.shape)


#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

#定义一个函数，按批次取数据 ，有点像mnist中的batch函数。
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
    img = tf.reshape(img, [224, 224, 3])#reshape为224x224的3通道图片
    img = tf.cast(img, tf.float32) * (1./255)#在流中抛出img张量
    label = tf.cast(features['label'],tf.int32)#在流中抛出label张量
#    print(type(label))
    return img, label

data_train, label_train = read_and_decode('data_train.tfrecords')
data_test, label_test = read_and_decode('data_test.tfrecords')


#-----------------构建网络----------------------
#占位符
x = tf.placeholder(tf.float32,shape = [None,w,h,c],name='x')
y_ = tf.placeholder(tf.int32,shape = [None,],name='y_')

#第一个卷积层（224——>56)
conv1 = tf.layers.conv2d(
        inputs = x,
        filters = 64,
        kernel_size = [11, 11],
        #步长默认为1，又因为模式为same，所以尺寸不变
        padding = "same",
        activation = tf.nn.relu,
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
print_activations(conv1)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)#图片缩小实际是跟步长有关，与size无关
print_activations(pool1)

#第二个卷积层(56->28)
conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 192,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu,
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
print_activations(conv2)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
print_activations(pool2)

#第三个卷积层(28->14)
conv3 = tf.layers.conv2d(
        inputs = pool2,
        filters = 384,
        kernel_size = [3, 3],
        padding = "same",
        activation = tf.nn.relu,
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
print_activations(conv3)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
print_activations(pool3)

#第四个卷积层(14->7)
conv4 = tf.layers.conv2d(
        inputs = pool3,
        filters = 256,
        kernel_size = [3, 3],
        padding = "same",
        activation = tf.nn.relu,
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
print_activations(conv4)
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
print_activations(pool4)

#第五个卷积层(7->3)
conv5 = tf.layers.conv2d(
        inputs = pool4,
        filters = 256,
        kernel_size = [3, 3],
        padding = "same",
        activation = tf.nn.relu,
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
print_activations(conv5)
pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
print_activations(pool5)

re1 = tf.reshape(pool5, [-1, 3 * 3 * 256])#6x6是因为100x100的图片经过了四次池化，尺寸100->50->25->12->6`

#全连接层
dense1 = tf.layers.dense(
         inputs = re1, 
         units = 4096, 
         activation = tf.nn.relu,
         kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
         kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))
print_activations(dense1)

dense2 = tf.layers.dense(
         inputs = dense1, 
         units = 4096, 
         activation = tf.nn.relu,
         kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
         kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))
print_activations(dense2)

logits = tf.layers.dense(
         inputs = dense2, 
         units = 1000, 
         activation = None,
         kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
         kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))
print_activations(logits)
#---------------------------网络结束---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





#训练和测试数据，可将n_epoch设置更大一些


sess = tf.InteractiveSession()

if reload_net == 1:
    saver = tf.train.Saver()
    saver.restore(sess,model_dir+'/'+model_name)
    print('恢复数据成功')
else:
    sess.run(tf.global_variables_initializer())
    print ('未恢复数据，初始化了数据')


for epoch in range(n_epoch):#1个epoch等于使用训练集中的全部样本训练一次
    start_time = time.time()
    print("-epoch %d-" % (epoch))

    data_tr_batch, label_tr_batch = tf.train.shuffle_batch([data_train, label_train], batch_size = batch_size, capacity = 2000, min_after_dequeue = 1000)
    threads = tf.train.start_queue_runners(sess = sess)
    data_tr_batch = data_tr_batch.eval()
    label_tr_batch = label_tr_batch.eval()#将tensor类型转化为numpy.ndarray类型，不然没法feed给placeholder

    data_te_batch, label_te_batch = tf.train.shuffle_batch([data_test, label_test], batch_size = batch_size, capacity = 2000, min_after_dequeue = 1000)
    threads = tf.train.start_queue_runners(sess = sess)
    data_te_batch = data_te_batch.eval()
    label_te_batch = label_te_batch.eval()#将tensor类型转化为numpy.ndarray类型，不然没法feed给placeholder
    
    '''####################训练方法：1#########################

    train_op.run(feed_dict = {x: data_batch, y_: label_batch})

    if epoch%10 == 0:
    	train_accuracy = accuracy.eval(feed_dict = {x: data_batch, y_: label_batch})
    	print("step %d, training accuracy %g"%(epoch, train_accuracy))
    
    '''######################################################

    
    '''####################训练方法：2#########################'''
    #training
    _, err, ac=sess.run([train_op, loss, accuracy], feed_dict = {x: data_tr_batch, y_: label_tr_batch})#sess.run返回的是在里面填的东西，一一对应
    print("   train loss: %f" % (err))
    print("   train acc: %f" % (ac))

    #testing
    err, ac=sess.run([loss, accuracy], feed_dict = {x: data_te_batch, y_: label_te_batch})#sess.run返回的是在里面填的东西，一一对应
    print("   test loss: %f" % (err))
    print("   test acc: %f" % (ac))
    '''######################################################'''
    
    
  
print("训练完成！")

# 创建模型保存目录
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# 保存模型
saver = tf.train.Saver()
if (savenet == 1):
    saver.save(sess, os.path.join(model_dir, model_name))
    print("保存模型成功！")
sess.close()