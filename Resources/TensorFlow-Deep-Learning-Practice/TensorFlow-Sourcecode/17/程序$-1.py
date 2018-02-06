import tensorflow as tf

class NIN_model:

    def __init__(self, imgs):
        self.imgs = imgs
        self.convlayers()
        self.NIN_result = self.result

    def saver(self):
        return tf.train.Saver()

    def maxpool(self,name,input_data,kernel_h,kernel_w,stride_h,stride_w):
        print(input_data.get_shape())
        out = tf.nn.max_pool(input_data,[1,kernel_h,kernel_w,1],
[1,stride_h,stride_w,1],padding="SAME",name=name)
        return out


    def avg_pool(self,name,input_data,kernel_h,kernel_w,stride_h,stride_w):
        print(input_data.get_shape())
        return tf.nn.avg_pool(input_data,[1,kernel_h,kernel_w,1],
[1,stride_h,stride_w,1],padding="VALID",name=name)

    def conv(self,name, input_data, out_channel,kernel_h,kernel_w,stride_h,stride_w,padding="SAME"):
        print(input_data.get_shape())
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [kernel_h, kernel_w, in_channel, out_channel], dtype=tf.float32)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, stride_h, stride_w, 1], padding=padding)
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        return out

    def relu(self,name,input_data):
        out = tf.nn.relu(input_data,name)
        return out

    def convlayers(self):

        #conv1
        self.out_data = self.conv("conv1",self.imgs,48,11,11,4,4)#(?, 224, 224, 3)
        self.out_data = self.batch_norm(self.out_data)
        self.out_data = self.relu("relu",self.out_data)
        self.out_data = self.conv("cccp1",self.out_data,48,1,1,1,1)#(?, 56, 56, 96)
        self.out_data = self.relu("relu",self.out_data)
        self.out_data = self.conv("cccp2",self.out_data,48,1,1,1,1)#(?, 56, 56, 96)
        self.out_data = self.maxpool("pool1",self.out_data,2,2,2,2)#(?, 56, 56, 96)

        #conv2
        self.out_data = self.conv("conv2",self.out_data,72,5,5,1,1) #(?, 28, 28, 96)
        self.out_data = self.batch_norm(self.out_data)
        self.out_data = self.relu("relu",self.out_data)
        self.out_data = self.conv("cccp3",self.out_data,72,1,1,1,1)#(?, 28, 28, 256)
        self.out_data = self.relu("relu",self.out_data)
        self.out_data = self.conv("cccp4",self.out_data,72,1,1,1,1)#(?, 28, 28, 256)
        self.out_data = self.maxpool("pool2",self.out_data,2,2,2,2)#(?, 28, 28, 256)

        #conv3
        self.out_data = self.conv("conv3",self.out_data,32,3,3,1,1)#(?, 14, 14, 256)
        self.out_data = self.batch_norm(self.out_data)
        self.out_data = self.relu("relu",self.out_data)
        self.out_data = self.conv("cccp5",self.out_data,32,1,1,1,1)#(?, 14, 14, 128)
        self.out_data = self.relu("relu",self.out_data)
        self.out_data = self.conv("cccp6",self.out_data,32,1,1,1,1)#(?, 14, 14, 128)
        self.out_data = self.maxpool("pool3",self.out_data,2,2,2,2)#(?, 14, 14, 128)

        #conv3
        self.out_data = self.conv("conv4",self.out_data,1024,3,3,1,1)#(?, 7, 7, 128)
        self.out_data = self.batch_norm(self.out_data)
        self.out_data = self.relu("relu",self.out_data)
        self.out_data = self.conv("cccp7",self.out_data,1000,1,1,1,1)#(?, 7, 7, 2)
        self.out_data = self.relu("relu",self.out_data)
        self.out_data = self.conv("cccp8",self.out_data,1000,1,1,1,1)#(?, 7, 7, 2)

        print("here shape is :", self.out_data.get_shape())
        self.out_data = self.avg_pool("avgpool",self.out_data,7,7,2,2)
        print("here shape is :" , self.out_data.get_shape())
        self.result = tf.reshape(self.out_data,[-1,2])
        print("here is model_2 and result shape is :" , self.result.get_shape())
