"""
Written by Matteo Dunnhofer - 2017

Data provider class
"""
import os
import random
from PIL import Image
import scipy.ndimage, scipy.misc
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset


class ImageNetDataset(object):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode

        self.wnids, self.words = self.load_imagenet_meta()
        data_imgs, data_labels = self.read_image_paths_labels()

        if self.mode == 'train':
            self.dataset = tf.data.Dataset.from_tensor_slices((data_imgs, data_labels))
            self.dataset = self.dataset.map(self.input_parser)
            self.dataset = self.dataset.shuffle(10000).batch(self.cfg.BATCH_SIZE)
        elif self.mode == 'val':
            self.dataset = tf.data.Dataset.from_tensor_slices((data_imgs, data_labels))
            self.dataset = self.dataset.map(self.input_parser).batch(self.cfg.BATCH_SIZE)
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices((data_imgs, data_labels))
            self.dataset = self.dataset.map(self.input_parser)

    def load_imagenet_meta(self):
        """
        It reads ImageNet metadata from ILSVRC 2012 dev tool file

        Returns:
            wnids: list of ImageNet wnids labels (as strings)
            words: list of words (as strings) referring to wnids labels and describing the classes

        """
        meta_path = os.path.join(self.cfg.DATA_PATH, 'data', 'meta.mat')
        metadata = loadmat(meta_path, struct_as_record=False)

        # ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
        synsets = np.squeeze(metadata['synsets'])
        ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
        wnids = np.squeeze(np.array([s.WNID for s in synsets]))
        words = np.squeeze(np.array([s.words for s in synsets]))
        return wnids, words

    def read_image_paths_labels(self):
        """
        Reads the paths of the images (from the folders structure)
        and the indexes of the labels (using an annotation file)
        """
        paths = []
        labels = []

        if self.mode == 'train':
            for i, wnid in enumerate(self.wnids):
                img_names = os.listdir(os.path.join(self.cfg.DATA_PATH, self.mode, wnid))
                for img_name in img_names:
                    paths.append(os.path.join(self.cfg.DATA_PATH, self.mode, wnid, img_name))
                    labels.append(i)

            # shuffling the images names and relative labels
            d = zip(paths, labels)
            random.shuffle(d)
            paths, labels = zip(*d)

        else:
            with open(os.path.join(self.cfg.DATA_PATH, 'data', 'ILSVRC2012_validation_ground_truth.txt')) as f:
                groundtruths = f.readlines()
            groundtruths = [int(x.strip()) for x in groundtruths]

            images_names = sorted(os.listdir(os.path.join(self.cfg.DATA_PATH, 'ILSVRC2012_img_val')))

            for image_name, gt in zip(images_names, groundtruths):
                paths.append(os.path.join(self.cfg.DATA_PATH, 'ILSVRC2012_img_val', image_name))
                labels.append(gt)

        self.dataset_size = len(paths)

        return tf.constant(paths), tf.constant(labels)

    def input_parser(self, img_path, label):
        """
        Parse a single example
        Reads the image tensor (and preprocess it) given its path and produce a one-hot label given an integer index

        Args:
            img_path: a TF string tensor representing the path of the image
            label: a TF int tensor representing an index in the one-hot vector

        Returns:
            a preprocessed tf.float32 tensor of shape (heigth, width, channels)
            a tf.int one-hot tensor
        """
        one_hot = tf.one_hot(label, self.cfg.NUM_CLASSES)

        # image reading
        image = self.read_image(img_path)
        image_shape = tf.shape(image)

        # resize of the image (setting largest border to 256px)
        new_h = tf.cond(image_shape[0] < image_shape[1],
                        lambda: tf.div(tf.multiply(256, image_shape[1]), image_shape[0]),
                        lambda: 256)
        new_w = tf.cond(image_shape[0] < image_shape[1],
                        lambda: 256,
                        lambda: tf.div(tf.multiply(256, image_shape[0]), image_shape[1]))

        image = tf.image.resize_images(image, size=[new_h, new_w])

        if self.mode == 'test':
            # take random crops for testing
            patches = []
            for k in range(self.cfg.K_PATCHES):
                patches.append(
                    tf.random_crop(image, size=[self.cfg.IMG_SHAPE[0], self.cfg.IMG_SHAPE[1], self.cfg.IMG_SHAPE[2]]))

            image = patches
        else:
            image = tf.random_crop(image, size=[self.cfg.IMG_SHAPE[0], self.cfg.IMG_SHAPE[1], self.cfg.IMG_SHAPE[2]])

            if self.mode == 'train':
                # some easy data augmentation
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # normalization
        image = tf.to_float(image)
        image = tf.subtract(image, self.cfg.IMAGENET_MEAN)

        return image, one_hot

    def read_image(self, img_path):
        """
        Given a path of image it reads its content
        into a tf tensor

        Args:
            img_path, a tf string tensor representing the path of the image

        Returns:
            the tf image tensor
        """
        img_file = tf.read_file(img_path)
        return tf.image.decode_jpeg(img_file, channels=self.cfg.IMG_SHAPE[2])
