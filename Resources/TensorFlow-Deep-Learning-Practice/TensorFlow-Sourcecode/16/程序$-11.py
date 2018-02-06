import numpy as np
import tensorflow as tf
import global_variable
import VGG16_model as model
from vgg16_weights_and_classe.imagenet_classes import class_names
from scipy.misc import imread, imresize

imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = model.vgg16(imgs)
saver = vgg.saver()

sess = tf.Session()
saver.restore(sess, global_variable.save_path)
img1 = imread('001.jpg', mode='RGB')
img1 = imresize(img1, (224, 224))
prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
