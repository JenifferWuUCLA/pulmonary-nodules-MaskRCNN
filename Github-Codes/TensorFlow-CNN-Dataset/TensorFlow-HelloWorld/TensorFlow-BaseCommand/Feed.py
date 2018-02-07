# coding: UTF-8
'''该机制 可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.feed 使用一个 tensor 值临时替换一个操作的输出结果. 你可以提供 feed 数据作为 run() 调用的参数. feed 只在调用它的方法内有效, 方法结束, feed 就会消失. 最常见的用例是将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符. ""'''

import tensorflow as tf

# 如果没有正确提供 feed, placeholder() 操作将会产生错误. 
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
    # 输出:
    # [array([ 14.], dtype=float32)]
