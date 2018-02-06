import tensorflow as tf
import MLP_model as model
import tensorflow.contrib.slim as slim
#定义的存储地址，读者可以自由定义sava_path作为模型的存储地址
from global_variable import save_path as save_path 

import shutil
shutil.rmtree(save_path)

g = tf.Graph()
with g.as_default():
    #在控制台打印log信息
    tf.logging.set_verbosity(tf.logging.INFO)

    #创建数据集
    xs,ys = model.produce_batch(200)
    #将数据转化为Tensor，使用这种格式能够是的Tensorflow队列自动调用
    inputs, outputs = model.convert_data_to_tensors(xs,ys)

    #计算模型值
    prediction, _ = model.mlp_model(inputs)

    #损失函数定义
    loss = slim.losses.mean_squared_error(prediction,outputs,scope="loss") #均方误差

    #使用梯度下降算法训练模型
    optimizer = slim.train.GradientDescentOptimizer(0.005)    
    train_op = slim.learning.create_train_op(loss,optimizer)

    #使用Tensorflow高级执行框架“图”去执行模型训练任务。
    final_loss = slim.learning.train(
        train_op,
        logdir=save_path,
        number_of_steps=1000,
        log_every_n_steps=200,
    )

    print("Finished training. Last batch loss:", final_loss)
    print("Checkpoint saved in %s" % save_path)
