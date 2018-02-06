import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import global_variable
import inception_resnet_v2 as model
checkpoints_dir = global_variable.pre_ckpt_save_model

def load_batch(dataset, batch_size=8, height=299, width=299, is_training=True):

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

fintuning_newFC = tf.Graph()
with fintuning_newFC.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    from datasets import flowers
    dataset = flowers.get_split('train', global_variable.flowers_data_dir)
    images, labels = load_batch(dataset)

    with slim.arg_scope(model.inception_resnet_v2_arg_scope()):
        pre,_ = model.inception_resnet_v2(images,num_classes=1001)
        pre = slim.fully_connected(pre,5)
    probabilities = tf.nn.softmax(pre)

    one_hot_labels = slim.one_hot_encoding(labels, 5)

    slim.losses.softmax_cross_entropy(probabilities, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    model_path = os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt')
    variables = slim.get_model_variables('InceptionResnetV2')
    init_fn = slim.assign_from_checkpoint_fn(model_path,variables)

    with tf.Session() as sess:
        init_fn(sess)
        print("done")
        final_loss = slim.learning.train(
                train_op,
                logdir=global_variable.save_model,
                number_of_steps=100
        )
