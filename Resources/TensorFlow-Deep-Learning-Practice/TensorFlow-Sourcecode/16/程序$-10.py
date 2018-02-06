import tensorflow as tf
import VGG16_model as model
import global_variable

if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = model.vgg16(imgs)
    prob = vgg.probs
    saver = vgg.saver()
    sess = tf.Session()
    vgg.load_weights(".\\vgg16_weights_and_classe\\vgg16_weights.npz",sess)
    saver.save(sess,global_variable.save_path)
