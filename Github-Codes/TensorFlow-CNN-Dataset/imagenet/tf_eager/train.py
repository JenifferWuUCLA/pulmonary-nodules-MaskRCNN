"""
Written by Matteo Dunnhofer - 2017

models training on ImageNet
"""
import argparse
import os.path
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from models.alexnet import AlexNet
from data import ImageNetDataset
from config import Configuration
import utils as ut

tfe.enable_eager_execution()

class Trainer(object):

	def __init__(self, cfg, net, trainingset, valset, resume):
		self.cfg = cfg
		self.net = net
		self.trainingset = trainingset
		self.valset = valset

		#self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.LEARNING_RATE)
		self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.cfg.LEARNING_RATE, momentum=self.cfg.MOMENTUM)

		self.global_step = tf.train.get_or_create_global_step()

		self.epoch = tfe.Variable(0, name='epoch', dtype=tf.float32, trainable=False)

		self.writer = tf.contrib.summary.create_summary_file_writer(self.cfg.SUMMARY_PATH)

		self.all_variables = (self.net.variables
						+ self.optimizer.variables()
						+ [self.global_step]
						+ [self.epoch])

		if resume:
			tfe.Saver(self.all_variables).restore(tf.train.latest_checkpoint(self.cfg.CKPT_PATH))

	def loss(self, mode, x, y):
		"""
		Computes the loss for a given batch of examples

		Args:
			mode, string 'train' or 'val'
			x, tf tensor representing a batch of images
			y, tf tensor representing a batch of labels

		Returns:
			the loss between the predictions on the images and the groundtruths
		"""
		pred = self.net(x)

		loss_value = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred)
		weight_decay = tf.reduce_sum(self.cfg.LAMBDA * tf.stack([tf.nn.l2_loss(v) for v in self.net.variables]))

		total_loss = loss_value + weight_decay

		tf.contrib.summary.scalar(mode + '/loss', total_loss)

		return total_loss

	def accuracy(self, mode, x, y):
		"""
		Computes the accuracy for a given batch of examples

		Args:
			mode, string 'train' or 'val'
			x, tf tensor representing a batch of images
			y, tf tensor representing a batch of labels

		Returns:
			the accuracy of the predictions on the images and the groundtruths
		"""
		pred = tf.nn.softmax(self.net(x))

		accuracy_value = tf.reduce_sum(
					tf.cast(
						tf.equal(
							tf.argmax(pred, axis=1, output_type=tf.int64),
							tf.argmax(y, axis=1, output_type=tf.int64)
						),
						dtype=tf.float32
					) 
				) / float(pred.shape[0].value)

		tf.contrib.summary.scalar(mode +'/accuracy', accuracy_value)

		return accuracy_value


	def train(self):
		"""
		Training procedure
		"""
		start_time = time.time()
		step_time = 0.0

		with self.writer.as_default():
			with tf.contrib.summary.record_summaries_every_n_global_steps(self.cfg.DISPLAY_STEP):
				
				for e in range(self.epoch.numpy(), self.cfg.EPOCHS):
					tf.assign(self.epoch, e)
					for (batch_i, (images, labels)) in enumerate(tfe.Iterator(self.trainingset.dataset)):
						self.global_step = tf.train.get_global_step()
						step = self.global_step.numpy() + 1
						
						step_start_time = int(round(time.time() * 1000))

						self.optimizer.minimize(lambda: self.loss('train', images, labels), global_step=self.global_step)

						step_end_time = int(round(time.time() * 1000))
						step_time += step_end_time - step_start_time

						if (step % self.cfg.DISPLAY_STEP) == 0:
							l = self.loss('train', images, labels)
							a = self.accuracy('train', images, labels).numpy()
							print ('Epoch: {:03d} Step/Batch: {:09d} Step mean time: {:04d}ms \nLoss: {:.7f} Training accuracy: {:.4f}'.format(e, step, int(step_time / step), l, a))
						
						if (step % self.cfg.VALIDATION_STEP) == 0:
							val_images, val_labels = tfe.Iterator(self.valset.dataset).next()
							l = self.loss('val', val_images, val_labels)
							a = self.accuracy('val', val_images, val_labels).numpy()
							int_time = time.time() - start_time
							print ('Elapsed time: {} --- Loss: {:.7f} Validation accuracy: {:.4f}'.format(ut.format_time(int_time), l, a))
						
						if (step % self.cfg.SAVE_STEP) == 0:
							tfe.Saver(self.all_variables).save(os.path.join(self.cfg.CKPT_PATH, 'net.ckpt'), global_step=self.global_step)
							print('Variables saved')
							
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--resume', help='Resume the training from the last checkpoint', action='store_true')
	args = parser.parse_args()

	cfg = Configuration()
	net = AlexNet(cfg, training=True)

	trainingset = ImageNetDataset(cfg, 'train')
	valset = ImageNetDataset(cfg, 'val')

	if not os.path.exists(cfg.CKPT_PATH):
		os.makedirs(cfg.CKPT_PATH)

	if tfe.num_gpus() > 0:
		with tf.device('/gpu:0'):
			trainer = Trainer(cfg, net, trainingset, valset, args.resume)
			trainer.train()
	else:
		trainer = Trainer(cfg, net, args.resume)
		trainer.train()