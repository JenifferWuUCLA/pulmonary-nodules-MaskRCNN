import tensorflow as tf
import tensorflow.contrib.slim as slim
import global_variable
import inception_resnet_v2 as model

def load_batch(dataset, batch_size=4, height=299, width=299, is_training=True):

    import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as providerr
    data_provider = providerr.DatasetDataProvider(
        dataset, common_queue_capacity=8, common_queue_min=1)
    image_raw, label = data_provider.get(['image', 'label'])
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.image.convert_image_dtype(image_raw,tf.float32)

    images_raw, labels = tf.train.batch(
        [image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)
    return images_raw, labels

g = tf.Graph()
with g.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    from datasets import flowers
    dataset = flowers.get_split('train', global_variable.flowers_data_dir)
    images, labels = load_batch(dataset)

    with slim.arg_scope(model.inception_resnet_v2_arg_scope()):
        pre,_ = model.inception_resnet_v2(images,num_classes=5)
    probabilities = tf.nn.softmax(pre)

    one_hot_labels = slim.one_hot_encoding(labels, num_classes=5)

    print(one_hot_labels.shape)
    print(probabilities.shape)

    slim.losses.softmax_cross_entropy(probabilities, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    final_loss = slim.learning.train(
                    train_op,
                    logdir=None,
                    number_of_steps=10
    )

    print("training done and finished")
