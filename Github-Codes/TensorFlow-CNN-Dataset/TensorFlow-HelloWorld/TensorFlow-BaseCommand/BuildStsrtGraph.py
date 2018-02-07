# coding: UTF-8
'''
使用 TensorFlow, 你必须明白 TensorFlow:
    使用图 (graph) 来表示计算任务.
    在被称之为 会话 (Session) 的上下文 (context) 中执行图.
    使用 tensor 表示数据.
    通过 变量 (Variable) 维护状态.
    使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.
'''

# -*- coding: UTF-8 -*- 
# 构建图并且在一个图中启动一个会话
import tensorflow as tf

# 默认图现在有三个节点, 两个 constant() op, 和一个matmul() op. 为了真正进行矩阵相乘运算, 并得到矩阵乘法的 结果, 你必须在会话里启动这个图. 
# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3.,3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.''''''
product = tf.matmul(matrix1,matrix2)

# 启动默认图.
# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数. 
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
# 返回值 'result' 是一个 numpy `ndarray` 对象.''''''''''
# 任务完成, 关闭会话.

# Session 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作.""

with tf.Session() as sess:
    result = sess.run([product])
    print result
    print ('Hello World!')

'''
如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的. 为了让 TensorFlow 使用这些 GPU, 你必须将 op 明确指派给它们执行. with...Device 语句用来指派特定的 CPU 或 GPU 执行操作:
with tf.Session() as sess:
  with tf.device("/gpu:1"):
      matrix1 = tf.constant([[3., 3.]])
      matrix2 = tf.constant([[2.],[2.]])
      product = tf.matmul(matrix1, matrix2)
      ..."")

设备用字符串进行标识. 目前支持的设备包括:
   "/cpu:0": 机器的 CPU.
   "/gpu:0": 机器的第一个 GPU, 如果有的话.
   "/gpu:1": 机器的第二个 GPU, 以此类推.
   """"""
'''             
