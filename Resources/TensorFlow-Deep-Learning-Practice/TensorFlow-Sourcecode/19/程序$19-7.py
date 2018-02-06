import tensorflow as tf
import MLP_model as model
import tensorflow.contrib.slim as slim
from global_variable import save_path as save_path

#import shutil
#shutil.rmtree(save_path)

g = tf.Graph()
with g.as_default():
    #在控制台打印log信息
    tf.logging.set_verbosity(tf.logging.INFO)

    #创建数据集
    xs,ys = model.produce_batch(200)
    #将数据转化为Tensor，使用这种格式能够是的Tensorflow队列自动调用
    inputs, outputs = model.convert_data_to_tensors(xs,ys)

    #计算模型值
    prediction, end_point = model.mlp_model(inputs)

#损失函数定义
#均方误差
    mean_squared_error = slim.losses.mean_squared_error(prediction,outputs,scope="mean_squared_error") 
    #绝对误差
absolute_difference_loss =slim.losses.absolute_difference(prediction,outputs,scope="absolute_difference_loss")


    #定义全部的损失函数
    total_loss = mean_squared_error + absolute_difference_loss



    # 使用梯度下降算法训练模型
    optimizer = slim.train.GradientDescentOptimizer(0.005)  # 可以改成后面加mini....
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    saver = slim.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(1000):
            sess.run(train_op)
        saver.save(sess,save_path+"MLP_train_multiple_loss.ckpt")
        print(sess.run(end_point["fc1"]))
