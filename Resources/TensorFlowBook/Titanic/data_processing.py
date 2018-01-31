import os
import re

import pandas as pd
import tensorflow as tf

pjoin = os.path.join
DATA_DIR = pjoin(os.path.dirname(__file__), 'data')

train_data = pd.read_csv(pjoin(DATA_DIR, 'train.csv'))
test_data = pd.read_csv(pjoin(DATA_DIR, 'test.csv'))

# Translation:
#  Don: an honorific title used in Spain, Portugal, Italy
#  Dona: Feminine form for don
#  Mme: Madame, Mrs
#  Mlle: Mademoiselle, Miss
#  Jonkheer (female equivalent: Jonkvrouw) is a Dutch honorific of nobility
HONORABLE_TITLES = ['sir', 'lady', 'don', 'dona', 'countess', 'jonkheer',
                    'major', 'col', 'dr', 'master', 'capt']
NORMAL_TITLES = ['mr', 'ms', 'mrs', 'miss', 'mme', 'mlle', 'rev']
TITLES = HONORABLE_TITLES + NORMAL_TITLES


def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    return title_search.group(1).lower()


def get_family(row):
    last_name = row['Name'].split(",")[0]
    if last_name:
        family_size = 1 + row['Parch'] + row['SibSp']
        if family_size > 3:
            return "{0}_{1}".format(last_name.lower(), family_size)
        else:
            return "nofamily"
    else:
        return "unknown"


def get_deck(cabin):
    if pd.isnull(cabin):
        return 'U'
    return cabin[:1]


class TitanicDigest(object):
    def __init__(self, dataset):
        self.count_by_sex = dataset.groupby('Sex')['PassengerId'].count()
        self.mean_age = dataset['Age'].mean()
        self.mean_age_by_sex = dataset.groupby("Sex")["Age"].mean()
        self.mean_fare_by_class = dataset.groupby("Pclass")["Fare"].mean()
        self.titles = TITLES
        self.families = dataset.apply(get_family, axis=1).unique().tolist()
        self.decks = dataset["Cabin"].apply(get_deck).unique().tolist()
        self.embarkments = dataset.Embarked.unique().tolist()
        self.embark_mode = dataset.Embarked.dropna().mode().values


def preprocess(data, digest):
    # convert ['male', 'female'] values of Sex to [1, 0]
    data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    # fill empty age field with mean age
    data['Age'] = data['Age'].apply(
        lambda age: digest.mean_age if pd.isnull(age) else age)

    # is child flag
    data['Child'] = data['Age'].apply(lambda age: 1 if age <= 15 else 0)

    # fill fare with mean fare of the class
    def get_fare_value(row):
        if pd.isnull(row['Fare']):
            return digest.mean_fare_by_class[row['Pclass']]
        else:
            return row['Fare']

    data['Fare'] = data.apply(get_fare_value, axis=1)

    # fill Embarked with mode
    data['Embarked'] = data['Embarked'].apply(
        lambda e: digest.embark_mode if pd.isnull(e) else e)
    data["EmbarkedF"] = data["Embarked"].apply(digest.embarkments.index)

    #
    data['Cabin'] = data['Cabin'].apply(lambda c: 'U0' if pd.isnull(c) else c)

    # Deck
    data["Deck"] = data["Cabin"].apply(lambda cabin: cabin[0])
    data["DeckF"] = data['Deck'].apply(digest.decks.index)

    data['Title'] = data['Name'].apply(get_title)
    data['TitleF'] = data['Title'].apply(digest.titles.index)

    data['Honor'] = data['Title'].apply(
        lambda title: int(title in HONORABLE_TITLES))

    data['Family'] = data.apply(get_family, axis=1)

    if 'Survived' in data.keys():
        data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
    return data


digest = TitanicDigest(train_data)


def get_train_data():
    return preprocess(train_data, digest)


def get_test_data():
    return preprocess(test_data, digest)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def transform_to_tfrecord():
    data = pd.read_csv(pjoin(DATA_DIR, 'train.csv'))
    filepath = pjoin(DATA_DIR, 'data.tfrecords')
    writer = tf.python_io.TFRecordWriter(filepath)
    for i in range(len(data)):
        feature = {}
        for key in data.keys():
            value = data[key][i]
            if isinstance(value, int):
                value = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[value]))
            elif isinstance(value, float):
                value = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[value])
                )
            elif isinstance(value, str):
                value = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[value.encode(encoding="utf-8")])
                )
            feature[key] = value
        example = tf.train.Example(
            features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    transform_to_tfrecord()
