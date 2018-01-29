import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

################################
# Preparing Data
################################

# read data from file
data = pd.read_csv('data/train.csv')

# fill nan values with 0
data = data.fillna(0)
# convert ['male', 'female'] values of Sex to [1, 0]
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
# 'Survived' is the label of one class,
# add 'Deceased' as the other class
data['Deceased'] = data['Survived'].apply(lambda s: 1 - s)

# select features and labels for training
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()
dataset_Y = data[['Deceased', 'Survived']].as_matrix()

# split training data and validation set data
X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)

################################
# Constructing Dataflow Graph
################################

# create symbolic variables
X = tf.placeholder(tf.float32, shape=[None, 6])
y = tf.placeholder(tf.float32, shape=[None, 2])

# weights and bias are the variables to be trained
weights = tf.Variable(tf.random_normal([6, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')
y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)

# Minimise cost using cross entropy
# NOTE: add a epsilon(1e-10) when calculate log(y_pred),
# otherwise the result will be -inf
cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10),
                                reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

# use gradient descent optimizer to minimize cost
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# calculate accuracy
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

################################
# Training and Evaluating the model
################################

# use session to run the calculation
with tf.Session() as sess:
    # variables have to be initialized at the first place
    tf.global_variables_initializer().run()

    # training loop
    for epoch in range(10):
        total_loss = 0.
        for i in range(len(X_train)):
            # prepare feed data and run
            feed_dict = {X: [X_train[i]], y: [y_train[i]]}
            _, loss = sess.run([train_op, cost], feed_dict=feed_dict)
            total_loss += loss
        # display loss per epoch
        print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))

    # Accuracy calculated by TensorFlow
    accuracy = sess.run(acc_op, feed_dict={X: X_val, y: y_val})
    print("Accuracy on validation set: %.9f" % accuracy)

    # Accuracy calculated by NumPy
    pred = sess.run(y_pred, feed_dict={X: X_val})
    correct = np.equal(np.argmax(pred, 1), np.argmax(y_val, 1))
    numpy_accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set (numpy): %.9f" % numpy_accuracy)

    # predict on test data
    testdata = pd.read_csv('data/test.csv')
    testdata = testdata.fillna(0)
    # convert ['male', 'female'] values of Sex to [1, 0]
    testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    predictions = np.argmax(sess.run(y_pred, feed_dict={X: X_test}), 1)
    submission = pd.DataFrame({
        "PassengerId": testdata["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv("titanic-submission.csv", index=False)
