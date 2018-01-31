import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn

from data_processing import get_test_data
from data_processing import get_train_data

train_data = get_train_data()
X = train_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Child',
                'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
Y = train_data[['Deceased', 'Survived']].as_matrix()

# arguments that can be set in command line
tf.app.flags.DEFINE_integer('epochs', 10, 'Training epochs')
FLAGS = tf.app.flags.FLAGS

ckpt_dir = './ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# defind model
n_features = X.shape[1]
input = tflearn.input_data([None, n_features])
network = tflearn.layers.fully_connected(input, 100, activation='relu')
network = tflearn.layers.fully_connected(network, 100, activation='relu')
y_pred = tflearn.layers.fully_connected(network, 2, activation='softmax')
net = tflearn.regression(y_pred)
model = tflearn.DNN(net)

# restore model if there is a checkpoint
if os.path.isfile(os.path.join(ckpt_dir, 'model.ckpt')):
    model.load(os.path.join(ckpt_dir, 'model.ckpt'))

# train model
model.fit(X, Y, validation_set=0.1, n_epoch=FLAGS.epochs)

# save the trained model
model.save(os.path.join(ckpt_dir, 'model.ckpt'))

metric = model.evaluate(X, Y)
print('Accuracy on train set: %.9f' % metric[0])

# predict on test dataset
test_data = get_test_data()
X = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Child',
               'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
predictions = np.argmax(model.predict(X), 1)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

submission.to_csv("titanic-submission.csv", index=False)
