import tensorflow as tf
from alexnet import AlexNet
import matplotlib.pyplot as plt

class_name = ['cat', 'dog']


def test_image(path_image, num_class, weights_path='Default'):
    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized = tf.reshape(img_resized, shape=[1, 227, 227, 3])
    model = AlexNet(img_resized, 0.5, 2, skip_layer='', weights_path=weights_path)
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./tmp/checkpoints/model_epoch100.ckpt")
        # score = model.fc8
        print(sess.run(model.fc8))
        prob = sess.run(max)[0]
        plt.imshow(img_decoded.eval())
        plt.title("Class:" + class_name[prob])
        plt.show()


test_image('./validate/10.jpeg', num_class=2)
