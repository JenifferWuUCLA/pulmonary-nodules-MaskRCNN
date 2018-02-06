#确定卷积核大小
def get_kernel_size(factor):
    return 2 * factor - factor % 2

#创建相关矩阵
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

#进行upsampling卷积核
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = get_kernel_size(factor)
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights


from numpy import ogrid, repeat, newaxis
from skimage import io
import  numpy as np
size = 3
x, y = ogrid[:size, :size]
img = repeat((x + y)[..., newaxis], 3, 2) / 12.

import tensorflow as tf
img = tf.cast(img,tf.float32)	 #对数据格式进行转化uint8->float32
img = tf.expand_dims(img,0)	#扩大一个维度

kernel = bilinear_upsample_weights(3,3)		#随机生成一个卷积核
res = tf.nn.conv2d_transpose(img,kernel,[1,9,9,3],[1,3,3,1],padding="SAME")	#使用反卷积进行处理

with tf.Session() as sess:
    img = sess.run(tf.squeeze(res))		#使用图进行计算，并压缩结果

io.imshow(img, interpolation='none')		#显示压缩后图像
io.show()
