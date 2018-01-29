import tensorflow as tf
import tflearn


def generator(input_image):
    conv2d = tflearn.conv_2d
    batch_norm = tflearn.batch_normalization
    relu = tf.nn.relu

    ratios = [16, 8, 4, 2, 1]
    n_filter = 8
    net = []

    for i in range(len(ratios)):
        net.append(tflearn.max_pool_2d(input_image, ratios[i], ratios[i]))
        # block_i_0, block_i_1, block_i_2
        for block in range(3):
            ksize = 1 if (block + 1) % 3 == 0 else 3
            net[i] = relu(batch_norm(conv2d(net[i], n_filter, ksize)))
        if i != 0:
            # concat with net[i-1]
            upnet = batch_norm(net[i - 1])
            downnet = batch_norm(net[i])
            net[i] = tf.concat(3, [upnet, downnet])
            # block_i_3, block_i_4, block_i_5
            for block in range(3, 6):
                ksize = 1 if (block + 1) % 3 == 0 else 3
                net[i] = conv2d(net[i], n_filter * (i + 1), ksize)
                net[i] = relu(batch_norm(net[i]))

        if i != len(ratios) - 1:
            # upsample for concat
            net[i] = tflearn.upsample_2d(net[i], 2)

    nn = len(ratios) - 1
    output = conv2d(net[nn], 3, 1)
    return output
