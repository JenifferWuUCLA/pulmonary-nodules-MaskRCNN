import tensorflow as tf
import global_variable
from save_and_restore import lineRegulation_model as model

model = model.LineRegModel()

x_input = model.x_input
y_output = model.y_output

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,global_variable.save_path)

result = sess.run(y_output,feed_dict={x_input:[1]})
print(result)
