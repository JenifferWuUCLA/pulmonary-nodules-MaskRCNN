import pandas as pd
import tensorflow.contrib.learn as skflow
from sklearn import metrics
from sklearn.model_selection import train_test_split

from data_processing import get_test_data, get_train_data

train_data = get_train_data()
X = train_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Child',
                'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
Y = train_data['Survived']

# split training data and validation set data
X_train, X_val, Y_train, Y_val = (
    train_test_split(X, Y, test_size=0.1, random_state=42))

# skflow classifier
feature_cols = skflow.infer_real_valued_columns_from_input(X_train)
classifier = skflow.LinearClassifier(
    feature_columns=feature_cols, n_classes=2)
classifier.fit(X_train, Y_train, steps=200)
score = metrics.accuracy_score(Y_val, classifier.predict(X_val))
print("Accuracy: %f" % score)

# predict on test dataset
test_data = get_test_data()
X = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Child',
               'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
predictions = classifier.predict(X)
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})
submission.to_csv("titanic-submission.csv", index=False)
