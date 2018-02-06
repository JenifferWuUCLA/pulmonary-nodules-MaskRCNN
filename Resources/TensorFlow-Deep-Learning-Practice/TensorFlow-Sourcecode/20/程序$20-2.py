from datasets import flowers
import tensorflow as tf
import matplotlib.pyplot as plt
import global_variable
import tensorflow.contrib.slim.python.slim as slim
flowers_data_dir = global_variable.flowers_data_dir

with tf.Graph().as_default():
    dataset = flowers.get_split('train', flowers_data_dir)
    import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as providerr
    data_provider = providerr.DatasetDataProvider(
        dataset, common_queue_capacity=32, common_queue_min=1)
    image, label = data_provider.get(['image', 'label'])

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for i in range(5):
                np_image, np_label = sess.run([image, label])
                height, width, _ = np_image.shape
                class_name = name = dataset.labels_to_names[np_label]

                plt.figure()
                plt.imshow(np_image)
                plt.title('%s, %d x %d' % (name, height, width))
                plt.axis('off')
                plt.show()
