import numpy as np

def get_kernel_size(factor):
    return 2 * factor - factor % 2

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

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

import tensorflow as tf
def upsample_tf(factor, input_img):
    number_of_classes = input_img.shape[2]
    new_height = input_img.shape[0] * factor
    new_width = input_img.shape[1] * factor
    expanded_img = np.expand_dims(input_img, axis=0)

    with tf.Graph().as_default():
        with tf.Session() as sess:
                upsample_filter_np = bilinear_upsample_weights(factor,number_of_classes)
                res = tf.nn.conv2d_transpose(expanded_img, upsample_filter_np,
                        output_shape=[1, new_height, new_width, number_of_classes],
                        strides=[1, factor, factor, 1])
                final_result = sess.run(res,
                                feed_dict={upsample_filt_pl: upsample_filter_np,
                                           logits_pl: expanded_img})
    return final_result.squeeze()
