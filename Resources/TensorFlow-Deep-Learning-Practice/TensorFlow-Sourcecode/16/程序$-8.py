import numpy as np
import tensorflow as tf
import global_variable
import vgg16_weights_and_classe

class vgg16:

    def __init__(self, imgs):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc8)

    def saver(self):
        return tf.train.Saver()

    def maxpool(self,name,input_data):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name, input_data, out_channel):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return out

    def fc(self,name,input_data,out_channel):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32)
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32)
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters += [weights, biases]
        return out

    def convlayers(self):
        # zero-mean input
        #conv1
        self.conv1_1 = self.conv("conv1re_1",self.imgs,64)
        self.conv1_2 = self.conv("conv1_2",self.conv1_1,64)
        self.pool1 = self.maxpool("poolre1",self.conv1_2)

        #conv2
        self.conv2_1 = self.conv("conv2_1",self.pool1,128)
        self.conv2_2 = self.conv("convwe2_2",self.conv2_1,128)
        self.pool2 = self.maxpool("pool2",self.conv2_2)

        #conv3
        self.conv3_1 = self.conv("conv3_1",self.pool2,256)
        self.conv3_2 = self.conv("convrwe3_2",self.conv3_1,256)
        self.conv3_3 = self.conv("convrew3_3",self.conv3_2,256)
        self.pool3 = self.maxpool("poolre3",self.conv3_3)

        #conv4
        self.conv4_1 = self.conv("conv4_1",self.pool3,512)
        self.conv4_2 = self.conv("convrwe4_2",self.conv4_1,512)
        self.conv4_3 = self.conv("conv4rwe_3",self.conv4_2,512)
        self.pool4 = self.maxpool("pool4",self.conv4_3)

        #conv5
        self.conv5_1 = self.conv("conv5_1",self.pool4,512)
        self.conv5_2 = self.conv("convrwe5_2",self.conv5_1,512)
        self.conv5_3 = self.conv("conv5_3",self.conv5_2,512)
        self.pool5 = self.maxpool("poorwel5",self.conv5_3)

    def fc_layers(self):

        self.fc6 = self.fc("fc1", self.pool5, 4096)
        self.fc7 = self.fc("fc2", self.fc6, 4096)
        self.fc8 = self.fc("fc3", self.fc7, 1000)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")
