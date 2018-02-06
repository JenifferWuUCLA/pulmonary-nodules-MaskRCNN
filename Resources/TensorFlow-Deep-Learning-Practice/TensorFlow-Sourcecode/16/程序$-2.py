import tensorflow as tf
import numpy as np
import global_variable
from save_and_restore import lineRegulation_model as model

train_x = np.random.rand(5)
train_y = 5 * train_x + 3.2   # y = 5 * x + 3
model = model.LineRegModel()

a_val = model.a_val
b_val = model.b_val

x_input = model.x_input
y_label = model.y_label

y_output = model.y_output

loss = model.loss
optimize = model.get_op()
saver = tf.train.Saver()
if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    flag = True
    epoch = 0
    while flag:
        epoch += 1
        _ , loss_val = sess.run([optimize,loss],feed_dict={x_input:train_x,y_label:train_y})
        if loss_val < 1e-6:
            flag = False
    print(a_val.eval(sess) , "   ", b_val.eval(sess))
    print("-----------%d-----------"%epoch)

    saver.save(sess,global_variable.save_path)
    print("model save finished")
    sess.close()
