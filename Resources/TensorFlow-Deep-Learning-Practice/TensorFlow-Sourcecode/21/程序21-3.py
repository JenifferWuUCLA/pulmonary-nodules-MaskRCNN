from numpy import ogrid, repeat, newaxis
from skimage import io
import  numpy as np
size = 3
x, y = ogrid[:size, :size]
img = repeat((x + y)[..., newaxis], 3, 2) / 12.

import tensorflow as tf

img = tf.cast(img,tf.float32)	 #对数据格式进行转化uint8->float32
img = tf.expand_dims(img,0)	#扩大一个维度

kernel = tf.random_normal([5,5,3,3],dtype=tf.float32)		#随机生成一个卷积核
res = tf.nn.conv2d_transpose(img,kernel,[1,9,9,3],[1,1,1,1],padding="VALID")	#使用反卷积进行处理

with tf.Session() as sess:
    img = sess.run(tf.squeeze(res))		#使用图进行计算，并压缩结果

io.imshow(img/np.argmax(img), interpolation='none')		#显示压缩后图像
io.show()
