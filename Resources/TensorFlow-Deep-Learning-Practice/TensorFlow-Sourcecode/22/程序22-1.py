import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf

#产生的数据[size,length]
size = 500
length = 1000
logdir_path = "./simple_norm_gan_ckpt/"

with tf.Graph().as_default():

    def sample_data(size=size, length=length):
        data = []
        for _ in range(size):
            data.append(sorted(np.random.normal(4, 1.5, length)))
        return np.array(data).astype(np.float32)

    def random_data(size=size, length=length):
        data = []
        for _ in range(size):
            data.append(np.random.random(length))
        return np.array(data).astype(np.float32)

    def generate(input_data, reuse=False):
        with tf.variable_scope("generate"):
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                weights_regularizer=slim.l1_l2_regularizer(),activation_fn=None
                                ):
                fc1 = slim.fully_connected(inputs=input_data, num_outputs=length, scope="g_fc1", reuse=reuse)
                fc1 = tf.nn.softplus(fc1,name="g_softplus")
                fc2 = slim.fully_connected(inputs=fc1, num_outputs=length, scope="g_fc2", reuse=reuse)
        return fc2

    def discriminate(input_data, reuse=False):
        with tf.variable_scope("discriminate"):
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                weights_regularizer=slim.l1_l2_regularizer(),activation_fn=None
                                ):
                fc1 = slim.fully_connected(inputs=input_data, num_outputs=length, scope="d_fc1", reuse=reuse)
                fc1 = tf.tanh(fc1)
                fc2 = slim.fully_connected(inputs=fc1, num_outputs=length, scope="d_fc2", reuse=reuse)
                fc2 = tf.tanh(fc2)
                fc3 = slim.fully_connected(inputs=fc2, num_outputs=1, scope="d_fc3", reuse=reuse)
                fc3 = tf.tanh(fc3)
                fc3 = tf.sigmoid(fc3)
        return fc3

    fake_input = tf.placeholder(tf.float32, shape=[size, length], name="fake_input")
    real_input = tf.placeholder(tf.float32, shape=[size, length], name="real_input")

    Gz = generate(fake_input)
    Dz_r = discriminate(real_input)
    Dz_f = discriminate(Gz,reuse=True)

    d_loss = tf.reduce_mean(-tf.log(Dz_r) - tf.log(1 - Dz_f))
    g_loss = tf.reduce_mean(-tf.log(Dz_f))

    tf.summary.scalar('Generator_loss', g_loss) #加入
    tf.summary.scalar('Discriminator_loss', d_loss) #加入

    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if "d_" in var.name]
    g_vars = [var for var in tvars if "g_" in var.name]

    d_optimizator = tf.train.AdamOptimizer(0.0005).minimize(loss=d_loss,var_list=d_vars)
    g_optimizator = tf.train.AdamOptimizer(0.0003).minimize(loss=g_loss, var_list=g_vars)

    merged_summary_op = tf.summary.merge_all()  # 修改到sess上方
    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir_path, sess.graph) #紧接在tf.Session() as sess下
        sess.run(tf.global_variables_initializer())

        for i in range(300):
            sess.run(d_optimizator,feed_dict={real_input:sample_data(),fake_input:random_data()})
            print("--------pre_train %d epoch end---------"%i)

            if i % 50 == 0:
                merged_summary = sess.run(merged_summary_op,feed_dict={real_input:sample_data(),fake_input:random_data()})
                writer.add_summary(merged_summary, global_step=i)
                saver.save(sess, save_path=logdir_path, global_step=i)

        for i in range(1500):
            sess.run([d_optimizator],feed_dict={real_input:sample_data(),fake_input:random_data()})
            sess.run([g_optimizator], feed_dict={fake_input: random_data()})

            print("--------model_train %d epoch end---------"%i)

            if i % 50 == 0:
merged_summary=sess.run(merged_summary_op,feed_dict={real_input:sample_data(),fake_input:random_data()})
                writer.add_summary(merged_summary, global_step=i)

                saver.save(sess, save_path=logdir_path, global_step=i)
