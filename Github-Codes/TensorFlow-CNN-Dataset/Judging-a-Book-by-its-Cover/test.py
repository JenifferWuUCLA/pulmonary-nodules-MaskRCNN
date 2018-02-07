import tensorflow as tf
from train import get_data,features_from_data, PretrainedAlexNet

flags = tf.app.flags
FLAGS = flags.FLAGS


def classified_data():
    total_data = get_data(is_test=True)
    # ['ASIN', 'FILENAME', 'IMAGE_URL', 'TITLE', 'AUTHOR', 'CATEGORY_ID', 'CATEGORY']
    classified_data_list = []
    class_names = []
    for i in range(30):
        classified_data_list.append(total_data[total_data.CATEGORY_ID == i])
        class_names.append(total_data[total_data.CATEGORY_ID == i].CATEGORY.values[0])
    return classified_data_list, class_names


def test():
    test_data_list, class_names = classified_data()
    #TEST_SIZE = 57
    CLASS_SIZE = 30
    data_list, class_names = classified_data()
    total_accuracy = 0
    total_top3_accuracy = 0
    total_top2_accuracy = 0
    class_top1_accuracy = []
    class_top2_accuracy = []
    class_top3_accuracy = []

    with tf.Session() as sess:
        x_ = tf.placeholder(tf.float32, [None, 227, 227, 3])
        x_image = tf.reshape(x_, [-1, 227, 227, 3])
        y_ = tf.placeholder(tf.float32, [None, 30])
        keep_prob = tf.placeholder(tf.float32)
        model = PretrainedAlexNet(images=x_image, _keep_prob=keep_prob)

        correct_prediction = tf.equal(tf.argmax(model.fc8, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        is_top1 = tf.equal(tf.nn.top_k(model.fc8, k=1)[1][:, 0], tf.cast(tf.argmax(y_, 1), "int32"))
        is_top2 = tf.equal(tf.nn.top_k(model.fc8, k=2)[1][:, 1], tf.cast(tf.argmax(y_, 1), "int32"))
        is_top3 = tf.equal(tf.nn.top_k(model.fc8, k=3)[1][:, 2], tf.cast(tf.argmax(y_, 1), "int32"))
        is_in_top1 = is_top1
        is_in_top2 = tf.logical_or(is_in_top1, is_top2)
        is_in_top3 = tf.logical_or(is_in_top2, is_top3)

        accuracy1 = tf.reduce_mean(tf.cast(is_in_top1, "float"))
        accuracy2 = tf.reduce_mean(tf.cast(is_in_top2, "float"))
        accuracy3 = tf.reduce_mean(tf.cast(is_in_top3, "float"))

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_models)
        saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(CLASS_SIZE):
            test_x, test_labels = features_from_data(test_data_list[i], is_test=True)
            print(class_names[i])
            #top1 = sess.run(accuracy1, feed_dict={x_: test_x, y_: test_labels, keep_prob: 1.0})
            #top2 = sess.run(accuracy2, feed_dict={x_: test_x, y_: test_labels, keep_prob: 1.0})
            #top3 = sess.run(accuracy3, feed_dict={x_: test_x, y_: test_labels, keep_prob: 1.0})

            print("test_accuracy %f" % (accuracy.eval(feed_dict={x_: test_x, y_: test_labels, keep_prob: 1.0})))
            print("top1_accuracy %f" % (accuracy1.eval(feed_dict={x_: test_x, y_: test_labels, keep_prob: 1.0})))
            print("top2_accuracy %f" % (accuracy2.eval(feed_dict={x_: test_x, y_: test_labels, keep_prob: 1.0})))
            print("top3_accuracy %f\n" % (accuracy3.eval(feed_dict={x_: test_x, y_: test_labels, keep_prob: 1.0})))

            """
            total_accuracy = total_accuracy + top1
            total_top2_accuracy = total_top2_accuracy + top2
            total_top3_accuracy = total_top3_accuracy + top3

            class_top1_accuracy.append(top1)
            class_top2_accuracy.append(top2)
            class_top3_accuracy.append(top3)
            """

    #print('top1 avg=' + str(total_accuracy / CLASS_SIZE) + '\ntop2 avg=' + str(
    #    total_top2_accuracy / CLASS_SIZE) + '\ntop3 avg=' + str(total_top3_accuracy / CLASS_SIZE))


def main(argv=None):
    #if not tf.gfile.Exists(FLAGS.test_models):
    #    tf.gfile.makeDirs(FLAGS.test_models)
    test()


if __name__ == "__main__":
    tf.app.run()
