"""
Written by Matteo Dunnhofer - 2017

Class that defines the testing procedure
"""
import argparse
import re
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from data import ImageNetDataset
from config import Configuration
from models.alexnet import AlexNet

tfe.enable_eager_execution()

class Tester(object):

	def __init__(self, cfg, net, testset):

		self.cfg = cfg
		self.net = net
		self.testset = testset

		# dummy input to create the tf variables
		_ = self.net(tf.random_uniform([1, self.cfg.IMG_SHAPE[0], self.cfg.IMG_SHAPE[1], self.cfg.IMG_SHAPE[2]]))
		
		tfe.Saver(self.net.variables).restore(tf.train.latest_checkpoint(self.cfg.CKPT_PATH))

	def predict(self, x):
		"""
		Predicts and averages tha probabilities for a input image

		Args:
			x, a tf tensor representing a batch of images

		Returns:
			the averaged predictions
		"""
		return tf.reduce_mean(tf.nn.softmax(self.net(x)), axis=0)

	def top_1_accuracy(self, x, y):
		"""
		Computes the top-1 accuracy for k patches

		Args:
			mode, string 'train' or 'val'
			x, tf tensor representing a batch of k patches
			y, tf tensor representing a label

		Returns:
			the top-k accuracy between the predictions on the patches and the groundtruth
		"""
		pred = self.predict(x)

		top_1_accuracy_value = tf.reduce_mean(
					tf.cast(
						tf.equal(
							tf.argmax(pred, output_type=tf.int64),
							tf.argmax(y, output_type=tf.int64)
						),
						dtype=tf.float32
					) 
				)

		#tf.contrib.summary.scalar('Accuracy', accuracy_value)

		return top_1_accuracy_value

	def top_k_accuracy(self, x, y):
		"""
		Computes the top-k accuracy for k patches

		Args:
			mode, string 'train' or 'val'
			x, tf tensor representing a batch of k patches
			y, tf tensor representing a label

		Returns:
			the top-k accuracy between the predictions on the patches and the groundtruth
		"""
		pred = self.predict(x)

		top_k_accuracy_value = tf.reduce_mean(
					tf.cast(
						tf.nn.in_top_k(
							tf.stack([pred]), 
							tf.stack([tf.argmax(y)]), 
							k=self.cfg.TOP_K),
						dtype=tf.float32
					) 
				)

		#tf.contrib.summary.scalar('Accuracy', accuracy_value)

		return top_k_accuracy_value

	
	def test(self, mode):
		"""
		Testing procedure

		Args:
			mode: string, 'validation' or 'test',
				choose which set to test
		"""
		test_examples = self.testset.dataset_size

		total_top1_accuracy = 0.
		total_topk_accuracy = 0.

		for (ex_i, (images, label)) in enumerate(tfe.Iterator(self.testset.dataset)):

			top_1_a = self.top_1_accuracy(images, label)
			top_k_a = self.top_k_accuracy(images, label)
			
			total_top1_accuracy += top_1_a
			total_topk_accuracy += top_k_a

			if (ex_i % self.cfg.DISPLAY_STEP) == 0:
				print ('Examples done: {:5d}/{} ---- Top-1: {:.4f} -- Top-{}: {:.4f}'.format(ex_i + 1, test_examples, total_top1_accuracy / (ex_i + 1), self.cfg.TOP_K, total_topk_accuracy / (ex_i + 1)))
		
		print ('---- Final accuracy ----')
		print ('Top-1: {:.4f} -- Top-{}: {:.4f}'.format(total_top1_accuracy / test_examples, self.cfg.TOP_K, total_topk_accuracy / test_examples))
		print ('Top-1 error rate: {:.4f} -- Top-{} error rate: {:.4f}'.format(1 - (total_top1_accuracy / test_examples), self.cfg.TOP_K, 1 - (total_topk_accuracy / test_examples)))


	def classify_image(self, img_path):
		"""
		Predict the classes and the probabilities of an input image

		Args:
			img_path: the path of the image
		"""

		image, _ = self.testset.input_parser(img_path, [])

		pred = self.predict(image)

		# retrieve top k scores
		scores, indexes = tf.nn.top_k(pred, k=self.cfg.TOP_K)
		scores, indexes = scores.numpy(), indexes.numpy()

		print('AlexNet saw:')
		for idx in range(self.cfg.TOP_K):
			print ('{} - score: {}'.format(self.testset.words[indexes[idx]], scores[idx]))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--classify', help='Predict the class of an input image', type=str)
	parser.add_argument('--test', help='Evaluate accuracy on the test set', action='store_true')
	parser.add_argument('--validation', help='Evaluate accuracy on the validation set', action='store_true')
	args = parser.parse_args()

	cfg = Configuration()
	net = AlexNet(cfg, training=False)

	testset = ImageNetDataset(cfg, 'test')

	if tfe.num_gpus() > 2:
		# set 2 to 0 if you want to run on the gpu
		# but currently running on gpu is impossible 
		# because tf.in_top_k does not have a cuda implementation
		with tf.device('/gpu:0'):
			tester = Tester(cfg, net, testset)
			
			if args.classify:
				tester.classify_image(args.classify)
			elif args.validation:
				tester.test('validation')
			else:
				tester.test('test')
	else:
		tester = Tester(cfg, net, testset)
		
		if args.classify:
			tester.classify_image(args.classify)
		elif args.validation:
			tester.test('validation')
		else:
			tester.test('test')



