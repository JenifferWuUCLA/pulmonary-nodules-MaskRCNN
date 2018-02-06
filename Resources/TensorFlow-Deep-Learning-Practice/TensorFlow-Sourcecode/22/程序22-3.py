import tensorflow as tf
from MNIST_data import input_data
import tensorflow.contrib.slim as slim
import model as model
batch_size = 300
import global_var

logdir_path = global_var.logdir_path #存储地址，读者自行设定

with tf.Graph().as_default():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x_input = tf.placeholder(tf.float32,[None,28,28,1])

    Gz = model.generate(batch_size)    #生成图片
    Dx = model.discriminate(x_input)    #判断真实的图片
    Dg = model.discriminate(Gz,reuse=True)   #判断图片的真假

    #对生成的图像是真为判定
    d_loss_real = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(Dx),logits=Dx ))
    #对生成的图像是假为判定
    d_loss_fake = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(Dg),logits=Dg))
    d_loss = d_loss_real + d_loss_fake
    #对生成器的结果进行判定
    g_loss = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(Dg),logits=Dg))

    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if "d_" in var.name]
    g_vars = [var for var in tvars if "g_" in var.name]

    d_trainer = tf.train.AdamOptimizer(0.005).minimize(loss=d_loss,var_list=d_vars)
    g_trainer = tf.train.GradientDescentOptimizer(0.001).minimize(loss=g_loss,var_list=g_vars)

    tf.summary.scalar('Generator_loss', g_loss)
    tf.summary.scalar('Discriminator_loss', d_loss)

    images_for_tensorboard = model.generate(5,is_training=False,reuse=True)
    tf.summary.image('Generated_images', images_for_tensorboard, 5)

    merged_summary_op = tf.summary.merge_all()  # 修改到sess上方
    saver = tf.train.Saver()
	
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir_path, sess.graph)  # 紧接在tf.Session() as sess下
        sess.run(tf.global_variables_initializer())

        for i in range(120):
            batch_xs = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            _,d_loss_var = sess.run([d_trainer,d_loss],feed_dict={x_input:batch_xs})
            print("d_loss_var: ",d_loss_var)
            print("-------pre train epoch %d end---------"%i)

        i = 0
        while True:
            batch_xs = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            *total_op, g_loss_var = sess.run([d_trainer,g_trainer,g_loss],feed_dict={x_input:batch_xs})
            print("g_loss_var: ",g_loss_var)
            i += 1
            if (i + 1)% 3 == 0:
                merged_summary = sess.run(merged_summary_op,feed_dict={x_input:batch_xs})
                writer.add_summary(merged_summary, global_step=i)
                saver.save(sess,"./discriminate_ckpt/GAN.ckpt")
            print("-------model train epoch %d end---------"%i)
