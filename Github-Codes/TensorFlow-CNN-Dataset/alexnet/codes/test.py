import os.path
import tensorflow as tf
import util as tu
from models import alexnet
import numpy as np


def test(
        top_k,
        k_patches,
        display_step,
        imagenet_path,
        ckpt_path):
    """
    Procedure to evaluate top-1 and top-k accuracy (and error-rate) on the
    ILSVRC2012 validation (test) set.

    Args:
        top_k: 	integer representing the number of predictions with highest probability
                to retrieve
        k_patches:	number of crops taken from an image and to input to the model
        display_step: number representing how often printing the current testing accuracy
        imagenet_path:	path to ILSRVC12 ImageNet folder containing train images,
                        validation images, annotations and metadata file
        ckpt_path:	path to model's tensorflow checkpoint
    """

    test_images = sorted(os.listdir(os.path.join(imagenet_path, 'ILSVRC2012_img_val')))
    test_labels = tu.read_test_labels(os.path.join(imagenet_path, 'data/ILSVRC2012_validation_ground_truth.txt'))

    test_examples = len(test_images)

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, 1000])

    _, pred = alexnet.classifier(x, 1.0)

    # calculate the average precision of the crops of the image
    avg_prediction = tf.div(tf.reduce_sum(pred, 0), k_patches)

    # accuracy
    top1_correct = tf.equal(tf.argmax(avg_prediction, 0), tf.argmax(y, 1))
    top1_accuracy = tf.reduce_mean(tf.cast(top1_correct, tf.float32))

    topk_correct = tf.nn.in_top_k(tf.stack([avg_prediction]), tf.argmax(y, 1), k=top_k)
    topk_accuracy = tf.reduce_mean(tf.cast(topk_correct, tf.float32))

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto()) as sess:
        saver.restore(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))

        total_top1_accuracy = 0.
        total_topk_accuracy = 0.

        for i in range(test_examples):
            # taking a few patches from an image
            image_patches = tu.read_k_patches(os.path.join(imagenet_path, 'ILSVRC2012_img_val', test_images[i]),
                                              k_patches)
            label = test_labels[i]

            top1_a, topk_a = sess.run([top1_accuracy, topk_accuracy], feed_dict={x: image_patches, y: [label]})
            total_top1_accuracy += top1_a
            total_topk_accuracy += topk_a

            if i % display_step == 0:
                print ('Examples done: {:5d}/{} ---- Top-1: {:.4f} -- Top-{}: {:.4f}'.format(i + 1, test_examples,
                                                                                             total_top1_accuracy / (
                                                                                             i + 1), top_k,
                                                                                             total_topk_accuracy / (
                                                                                             i + 1)))

        print ('---- Final accuracy ----')
        print ('Top-1: {:.4f} -- Top-{}: {:.4f}'.format(total_top1_accuracy / test_examples, top_k,
                                                        total_topk_accuracy / test_examples))
        print (
        'Top-1 error rate: {:.4f} -- Top-{} error rate: {:.4f}'.format(1 - (total_top1_accuracy / test_examples), top_k,
                                                                       1 - (total_topk_accuracy / test_examples)))


if __name__ == '__main__':
    TOP_K = 5
    K_PATCHES = 5
    DISPLAY_STEP = 10
    IMAGENET_PATH = '/media/desktop/F64E50644E502023/ILSVRC2012'
    CKPT_PATH = 'ckpt-alexnet'

    test(
        TOP_K,
        K_PATCHES,
        DISPLAY_STEP,
        IMAGENET_PATH,
        CKPT_PATH)
