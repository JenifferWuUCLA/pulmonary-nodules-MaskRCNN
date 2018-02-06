from matplotlib import pyplot as plt
import numpy as np
import global_variable
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
import os
checkpoints_path  = global_variable.pretrain_ckpt_path
import upsampling as utile
import tensorflow.contrib.slim as slim

from preprocessing.vgg_preprocessing import (_mean_image_subtraction,_R_MEAN, _G_MEAN, _B_MEAN)

def discrete_matshow(data, labels_names=[], title=""):
    fig_size = [7, 6]
    plt.rcParams["figure.figsize"] = fig_size
    cmap = plt.get_cmap('Paired', np.max(data)-np.min(data)+1)
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin = np.min(data)-.5,
                      vmax = np.max(data)+.5)

    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data),np.max(data)+1))
    if labels_names:
        cax.ax.set_yticklabels(labels_names)
    if title:
        plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.show()

def upsample_tf(factor, input_img):
    number_of_classes = input_img.shape[2]
    new_height = input_img.shape[0] * factor
    new_width = input_img.shape[1] * factor

    expanded_img = np.expand_dims(input_img, axis=0).astype(np.float32)

    with tf.Graph().as_default():
        with tf.Session() as sess:
                upsample_filter_np = utile.bilinear_upsample_weights(factor,
                                        number_of_classes)
                res = tf.nn.conv2d_transpose(expanded_img, upsample_filter_np,
                        output_shape=[1, new_height, new_width, number_of_classes],
                        strides=[1, factor, factor, 1])
                res = sess.run(res)
    return np.squeeze(res)

with tf.Graph().as_default():
    image = tf.image.decode_jpeg(tf.read_file("apple.jpg"), channels=3)
    image = tf.image.resize_images(image,[224,224])
    # 减去均值之前，将像素值转为32位浮点
    image_float = tf.to_float(image, name='ToFloat')
    # 每个像素减去像素的均值
    processed_image = _mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])
    input_image = tf.expand_dims(processed_image, 0)
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits,endpoints  = vgg.vgg_16(input_image,
                               num_classes=1000,
                               is_training=False,
                               spatial_squeeze=False)

    pred = tf.argmax(logits, dimension=3) #对输出层进行逐个比较，取得不同层同一位置中最大的概率所对应的值
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_path, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    with tf.Session() as sess:
        init_fn(sess)
        fcn8s,fcn16s,fcn32s = sess.run([endpoints["vgg_16/pool3"],endpoints["vgg_16/pool4"],endpoints["vgg_16/pool5"]])

    upsampled_logits = upsample_tf(factor=16, input_img=fcn8s.squeeze())
    upsample_predictions32 = upsampled_logits.squeeze().argmax(2)

    unique_classes, relabeled_image = np.unique(upsample_predictions32,return_inverse=True)
    relabeled_image = relabeled_image.reshape(upsample_predictions32.shape)

    labels_names = []
    import classes_names
    names = classes_names.names

    for index, current_class_number in enumerate(unique_classes):
        labels_names.append(str(index) + ' ' + names[current_class_number+1])

    discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")
