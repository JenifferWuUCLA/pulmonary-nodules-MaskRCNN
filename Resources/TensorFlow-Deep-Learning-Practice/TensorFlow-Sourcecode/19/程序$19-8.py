import tensorflow as tf
import MLP_model as model
import tensorflow.contrib.slim as slim
from global_variable import save_path as save_path

with tf.Graph().as_default():
    #在控制台打印log信息
    tf.logging.set_verbosity(tf.logging.INFO)

    #创建数据集
    xs,ys = model.produce_batch(200)
    #将数据转化为Tensor，使用这种格式能够是的Tensorflow队列自动调用
    inputs, outputs = model.convert_data_to_tensors(xs,ys)

    #计算模型值
prediction, _ = model.mlp_model(inputs,is_training=False)

    saver = tf.train.Saver()
    save_path =  tf.train.latest_checkpoint(save_path)
    with tf.Session() as sess:
        saver.restore(sess,save_path)
        inputs, prediction, outputs = sess.run([inputs,prediction,outputs])
