#!/usr/bin/env python
# _*_ coding: utf-8 _*_

from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = "../Pulmonary_nodules_data/train/"
val_data_dir = "../Pulmonary_nodules_data/val/"

# 训练的batch_size
batch_size = 16
# 训练的epoch
epochs = 100

# 图像Generator，用来构建输入数据
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True)

# 从文件中读取数据，目录结构应为train下面是各个类别的子目录，每个子目录中为对应类别的图像
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(224, 224), batch_size=batch_size)

# 训练图像的数量
image_numbers = train_generator.samples

# 输出类别信息
print train_generator.class_indices

# 生成测试数据
test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory(val_data_dir, target_size=(224, 224), batch_size=batch_size)

# 使用ResNet的结构，不包括最后一层，且加载ImageNet的预训练参数
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 构建网络的最后一层，3是自己的数据的类别
predictions = Dense(3, activation='softmax')(base_model.output)

# 定义整个模型
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型，loss为交叉熵损失
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 训练模型
model.fit_generator(train_generator, steps_per_epoch=image_numbers, batch_size=batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=batch_size)

# 保存训练得到的模型
model.save_weights('weights.h5')
