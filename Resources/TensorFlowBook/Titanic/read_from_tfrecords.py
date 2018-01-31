#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf


def read_and_decode(train_files, num_threads=2, num_epochs=100,
                    batch_size=10, min_after_dequeue=10):
    # read data from trainFile with TFRecord format
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(
        train_files,
        num_epochs=num_epochs)
    _, serialized_example = reader.read(filename_queue)
    featuresdict = tf.parse_single_example(
        serialized_example,
        features={
            'Survived': tf.FixedLenFeature([], tf.int64),
            'Pclass': tf.FixedLenFeature([], tf.int64),
            'Parch': tf.FixedLenFeature([], tf.int64),
            'SibSp': tf.FixedLenFeature([], tf.int64),
            'Sex': tf.FixedLenFeature([], tf.int64),
            'Age': tf.FixedLenFeature([], tf.float32),
            'Fare': tf.FixedLenFeature([], tf.float32)})

    # decode features to same format of float32
    labels = featuresdict.pop('Survived')
    features = [tf.cast(value, tf.float32)
                for value in featuresdict.values()]

    # get data with shuffle batch and return
    features, labels = tf.train.shuffle_batch(
        [features, labels],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_after_dequeue + 3 * batch_size,
        min_after_dequeue=min_after_dequeue)
    return features, labels


def train_with_queuerunner():
    x, y = read_and_decode(['train.tfrecords'])

    with tf.Session() as sess:
        tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()).run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                # Run training steps or whatever
                features, lables = sess.run([x, y])
                if step % 100 == 0:
                    print('step %d:' % step, lables)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)


if __name__ == '__main__':
    train_with_queuerunner()
