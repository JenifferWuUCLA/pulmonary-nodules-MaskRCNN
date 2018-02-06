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

    #制定的度量值-相对误差和绝对误差:
    names_to_value_nodes, names_to_update_nodes = slim.metrics.aggregate_metric_map({
      'Mean Squared Error': slim.metrics.streaming_mean_squared_error(prediction, outputs),
      'Mean Absolute Error': slim.metrics.streaming_mean_absolute_error(prediction, outputs)
    })

    sv = tf.train.Supervisor(logdir=save_path)
    with sv.managed_session() as sess:
        names_to_value = sess.run(names_to_value_nodes)
        names_to_update = sess.run(names_to_update_nodes)

    for key, value in names_to_value.items():
      print( (key, value))

    print("\n")
    for key, value in names_to_update.items():
        print((key, value))
