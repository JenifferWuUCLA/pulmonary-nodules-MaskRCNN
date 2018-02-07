"""
Written by Matteo Dunnhofer - 2017

Classify an input image
"""
import sys
import os.path
import tensorflow as tf
import train_util as tu
from models import alexnet
import numpy as np

def classify(
		image, 
		top_k, 
		k_patches, 
		ckpt_path, 
		imagenet_path):
	"""	Procedure to classify the image given through the command line

		Args:
			image:	path to the image to classify
			top_k: 	integer representing the number of predictions with highest probability
					to retrieve
			k_patches:	number of crops taken from an image and to input to the model
			ckpt_path:	path to model's tensorflow checkpoint
			imagenet_path:	path to ILSRVC12 ImageNet folder containing train images, 
						validation images, annotations and metadata file

	"""
	wnids, words = tu.load_imagenet_meta(os.path.join(imagenet_path, 'data/meta.mat'))

	# taking a few crops from an image
	image_patches = tu.read_k_patches(image, k_patches)

	x = tf.placeholder(tf.float32, [None, 224, 224, 3])

	_, pred = alexnet.classifier(x, dropout=1.0)

	# calculate the average precision through the crops
	avg_prediction = tf.div(tf.reduce_sum(pred, 0), k_patches)

	# retrieve top 5 scores
	scores, indexes = tf.nn.top_k(avg_prediction, k=top_k)

	saver = tf.train.Saver()

	with tf.Session(config=tf.ConfigProto()) as sess:
		saver.restore(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))

		s, i = sess.run([scores, indexes], feed_dict={x: image_patches})
		s, i = np.squeeze(s), np.squeeze(i)

		print('AlexNet saw:')
		for idx in range(top_k):
			print ('{} - score: {}'.format(words[i[idx]], s[idx]))


if __name__ == '__main__':
	TOP_K = 5
	K_CROPS = 5
	IMAGENET_PATH = 'ILSVRC2012'
	CKPT_PATH = 'ckpt-alexnet'

	image_path = sys.argv[1]

	classify(
		image_path, 
		TOP_K, 
		K_CROPS, 
		CKPT_PATH, 
		IMAGENET_PATH)

