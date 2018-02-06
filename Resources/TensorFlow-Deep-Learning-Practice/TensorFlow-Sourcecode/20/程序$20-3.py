import tensorflow.contrib.slim as slim

class Slim_cnn:
    def __init__(self,images,num_classes):
        self.images = images
        self.num_classes = num_classes
        self.net = self.model()

    def model(self):
        with slim.arg_scope([slim.max_pool2d],kernel_size = [2,2],stride = 2):
            net = slim.conv2d(self.images, 32, [3, 3])
            net = slim.max_pool2d(net)
            net = slim.conv2d(net, 64, [3, 3])
            net = slim.max_pool2d(net)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 128)
            net = slim.fully_connected(net, self.num_classes, activation_fn=None)
            return net
