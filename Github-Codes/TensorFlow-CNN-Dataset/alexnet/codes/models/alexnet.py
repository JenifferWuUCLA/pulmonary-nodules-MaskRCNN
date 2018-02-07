

import tensorflow as tf
import util as tu

def alexnet(x):
	"""
	AlexNet convolutional layers definition

	Args:
		x: tensor of shape [batch_size, width, height, channels]

	Returns:
		pool5: tensor with all convolutions, pooling and lrn operations applied

	"""
	with tf.name_scope('alexnetwork') as scope:
		with tf.name_scope('conv1') as inner_scope:
			wcnn1 = tu.weight([11, 11, 3, 96], name='wcnn1')
			bcnn1 = tu.bias(0.0, [96], name='bcnn1')
			conv1 = tf.add(tu.conv2d(x, wcnn1, stride=(4, 4), padding='SAME'), bcnn1)
			#conv1 = tu.batch_norm(conv1)
			conv1 = tu.relu(conv1)
			norm1 = tu.lrn(conv1, depth_radius=5, bias=1.0, alpha=1e-04, beta=0.75)
			pool1 = tu.max_pool2d(norm1, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

		with tf.name_scope('conv2') as inner_scope:
			wcnn2 = tu.weight([5, 5, 96, 256], name='wcnn2')
			bcnn2 = tu.bias(1.0, [256], name='bcnn2')
			conv2 = tf.add(tu.conv2d(pool1, wcnn2, stride=(1, 1), padding='SAME'), bcnn2)
			#conv2 = tu.batch_norm(conv2)
			conv2 = tu.relu(conv2)
			norm2 = tu.lrn(conv2, depth_radius=5, bias=1.0, alpha=1e-04, beta=0.75)
			pool2 = tu.max_pool2d(norm2, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

		with tf.name_scope('conv3') as inner_scope:
			wcnn3 = tu.weight([3, 3, 256, 384], name='wcnn3')
			bcnn3 = tu.bias(0.0, [384], name='bcnn3')
			conv3 = tf.add(tu.conv2d(pool2, wcnn3, stride=(1, 1), padding='SAME'), bcnn3)
			#conv3 = tu.batch_norm(conv3)
			conv3 = tu.relu(conv3)

		with tf.name_scope('conv4') as inner_scope:
			wcnn4 = tu.weight([3, 3, 384, 384], name='wcnn4')
			bcnn4 = tu.bias(1.0, [384], name='bcnn4')
			conv4 = tf.add(tu.conv2d(conv3, wcnn4, stride=(1, 1), padding='SAME'), bcnn4)
			#conv4 = tu.batch_norm(conv4)
			conv4 = tu.relu(conv4)

		with tf.name_scope('conv5') as inner_scope:
			wcnn5 = tu.weight([3, 3, 384, 256], name='wcnn5')
			bcnn5 = tu.bias(1.0, [256], name='bcnn5')
			conv5 = tf.add(tu.conv2d(conv4, wcnn5, stride=(1, 1), padding='SAME'), bcnn5)
			#conv5 = tu.batch_norm(conv5)
			conv5 = tu.relu(conv5)
			pool5 = tu.max_pool2d(conv5, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

		return pool5

def classifier(x, dropout):
	"""
	AlexNet fully connected layers definition

	Args:
		x: tensor of shape [batch_size, width, height, channels]
		dropout: probability of non dropping out units

	Returns:
		fc3: 1000 linear tensor taken just before applying the softmax operation
			it is needed to feed it to tf.softmax_cross_entropy_with_logits()
		softmax: 1000 linear tensor representing the output probabilities of the image to classify

	"""
	pool5 = alexnet(x)

	dim = pool5.get_shape().as_list()
	flat_dim = dim[1] * dim[2] * dim[3] # 6 * 6 * 256
	flat = tf.reshape(pool5, [-1, flat_dim])

	with tf.name_scope('classifier') as scope:
		with tf.name_scope('fullyconected1') as inner_scope:
			wfc1 = tu.weight([flat_dim, 4096], name='wfc1')
			bfc1 = tu.bias(0.0, [4096], name='bfc1')
			fc1 = tf.add(tf.matmul(flat, wfc1), bfc1)
			#fc1 = tu.batch_norm(fc1)
			fc1 = tu.relu(fc1)
			fc1 = tf.nn.dropout(fc1, dropout)

		with tf.name_scope('fullyconected2') as inner_scope:
			wfc2 = tu.weight([4096, 4096], name='wfc2')
			bfc2 = tu.bias(0.0, [4096], name='bfc2')
			fc2 = tf.add(tf.matmul(fc1, wfc2), bfc2)
			#fc2 = tu.batch_norm(fc2)
			fc2 = tu.relu(fc2)
			fc2 = tf.nn.dropout(fc2, dropout)

		with tf.name_scope('classifier_output') as inner_scope:
			wfc3 = tu.weight([4096, 1000], name='wfc3')
			bfc3 = tu.bias(0.0, [1000], name='bfc3')
			fc3 = tf.add(tf.matmul(fc2, wfc3), bfc3)
			softmax = tf.nn.softmax(fc3)

	return fc3, softmax

