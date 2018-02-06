from model_and_pretrain_ckpt import vgg16 as model
from preprocessing import vgg_preprocessing
import global_variable
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
image_size = 224
checkpoints_path  = global_variable.pretrain_ckpt_path
up_path = os.path.abspath('..')  #获取上层路径,如果不需要可以删除

with tf.Graph().as_default():
#根据路径组合图片地址，可以根据需要自定义图片地址
image_path = os.path.join(up_path, global_variable.bus)    
image = tf.image.decode_jpeg(tf.read_file(image_path),channels=3)
    image = vgg_preprocessing.preprocess_image(image,image_size,image_size,is_training=False)
    image = tf.expand_dims(image, 0)

	#进行图片预测
    with slim.arg_scope(model.vgg_arg_scope()):
        logits, _ = model.vgg_16(image, is_training=False)
    probabilities = slim.softmax(logits)

    #按地址获取权重存档路径，
    model_path = os.path.join(up_path,checkpoints_path, 'vgg_16.ckpt')
    #按写法回复存档权重至模型中
    init_fn = slim.assign_from_checkpoint_fn(
        model_path, slim.get_model_variables('vgg_16'))

    with tf.Session() as sess:
        init_fn(sess)
        res = sess.run([probabilities])
        probabilities = res[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),key=lambda x: x[1])]

        from datasets import classes_names#class_names是imagenet物品分类，可在代码库中下载
        names = classes_names.names
        for i in range(5):
            #下面是排序并输出
            index = sorted_inds[i]
            print('Probability %0.2f => [%s]' % (probabilities[index], names[index]))
