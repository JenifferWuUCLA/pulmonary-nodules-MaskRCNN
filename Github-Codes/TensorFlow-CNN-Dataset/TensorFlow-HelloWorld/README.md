TensorFlow代码实战
--------------------------

#### 备注：
自动下载安装数据集脚本：input.py

数据文件：MNIST_data

#### 1.TensorFlow实现Softmax Regression识别手写数字

Softmax-2.py

#### 2.极客学院翻译文章代码基于TensorFlow1.0以下的老版本实现（有注释和无注释两版）：

TensorFlow实现Softmax Regression识别手写数字

Sofrmax-1.py，Softmax-1-NoComment.py

#### 3.TensorFlow实现Softmax Regression多层感知机
SostmaxOptization.py

#### 4.卷积神经网络测试MNIST数据
cnn_tf_mnist.py

#### 5.TensorFlow实现卷积神经网络（进阶）

（1）TensorFlow官方Model：
[官方Model链接](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10)


文件 |		作用
---|---
cifar10_input.py | 读取本地CIFAR-10的二进制文件格式的内容。
cifar10.py | 建立CIFAR-10的模型。
cifar10_train.py | 在CPU或GPU上训练CIFAR-10的模型。
cifar10_multi_gpu_train.py | 在多GPU上训练CIFAR-10的模型。
cifar10_eval.py | 评估CIFAR-10模型的预测性能。



（2）TensorFlow实战代码：

三个代码文件：cifar10.py cifar10_input.py cnn_tf_CIFAR-10.py

卷积神经网络结构：

conv1 卷积层和激活函数

pool1 最大池化

norm1 LRN

conv2 卷积层和激活函数

norm2 LRN

pool2 最大池化层

local3 全连接层和激活函数

local4 全连接层和激活函数

logits 模型Inference的输出结果

[TensorFlow实现卷积神经网络（进阶）--- 极客学院相似案例解析](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/deep_cnn.html)

	
#### 6.TensorFlow运作方式 （更新中）
展示如何利用TensorFlow使用（经典）MNIST数据集训练并评估一个用于识别手写数字的简易前馈神经网络（feed-forward neural network）。我们的目标读者，是有兴趣使用TensorFlow的资深机器学习人士。

[官方代码](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist)

[极客学院教程](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_tf.html)

两个代码文件：

代码文件 | 目的
---|---
mnist.py | 构建一个完全连接（fully connected）的MINST模型所需的代码。
fully_connected_feed.py | 利用下载的数据集训练构建好的MNIST模型的主要代码，以数据反馈字典（feed dictionary）的形式作为输入模型。
#### 7.TensorFlow实现AlexNet

alexnet_benchmark.py（forward和backward耗时计算）

alexnet.py（预测与训练）

#### 8.TensorFlow实现VGGNet-16

VGG.py（forward和backward耗时计算）

#### 9.InceptionNet

InceptionNet-V3.py（forward耗时计算）

#### 10.ResNet


ResNet.py（ResNet 152的forward耗时计算）

#### 11.TensorBoard、多GPU并行及分布式并行

Distributed.py（分布式并行）

MultiGPU.py（多GPU并行）

TensorBoard.py（可视化）