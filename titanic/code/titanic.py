import tensorflow as tf
import pandas as pd
import argparse
import os

from clusterone import get_data_path, get_logs_path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

FEATURE_CLASSES = ['pclass','age', 'sex', 'sibsp', 'parch', 'embarked', 'survived']
network = [20, 20, 20]

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default="../data/titanic_train.csv", type=str, help='Path to training data file')
parser.add_argument('--test_path', default="../data/titanic_test.csv", type=str, help='Path to test data file')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    log_path = get_logs_path(root=os.path.abspath(os.path.expanduser('~/Documents/tf_logs/logs')))

    train, test = load_data(args.train_path, args.test_path)
    (train_x, train_y), new_feature_classes = preprocess_data(train, ['sex', 'embarked'])
    (test_x, test_y), new_feature_classes = preprocess_data(test, ['sex', 'embarked'])

    passenger_features = []
    passenger_features.append(tf.feature_column.numeric_column(key='pclass'))
    passenger_features.append(tf.feature_column.numeric_column(key='age'))
    passenger_features.append(tf.feature_column.numeric_column(key='sibsp'))
    passenger_features.append(tf.feature_column.numeric_column(key='parch'))
    passenger_features.append(tf.feature_column.numeric_column(key='sex_male'))
    passenger_features.append(tf.feature_column.numeric_column(key='sex_female'))
    passenger_features.append(tf.feature_column.numeric_column(key='embarked_C'))
    passenger_features.append(tf.feature_column.numeric_column(key='embarked_Q'))
    passenger_features.append(tf.feature_column.numeric_column(key='embarked_S'))

    classifier = tf.estimator.DNNClassifier(
                                            hidden_units=network, 
                                            feature_columns=passenger_features, 
                                            model_dir=log_path,
                                            n_classes=2)

    classifier.train(input_fn=lambda:train_input_fn(train_x, train_y), steps=1000)

    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y))

    print('\nNetwork layout: %s' % network)
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def load_data(train_path, test_path):
    tr_path, tr_filename = os.path.split(train_path)
    train_dir = get_data_path(
                            dataset_name = "svenchmie/titanic_data/titanic_train.csv",
                            local_root = tr_path,
                            local_repo = tr_filename,
                            path = ''
                            )

    train = pd.read_csv(train_dir[:-1], engine="python")
    train = train[FEATURE_CLASSES].dropna(axis=0, how='any')

    te_path, te_filename = os.path.split(test_path)
    test_dir = get_data_path(
                            dataset_name = "svenchmie/titanic_data/titanic_test.csv",
                            local_root = te_path,
                            local_repo = te_filename,
                            path = ''
                            )

    test = pd.read_csv(test_dir[:-1], engine="python")
    test = test[FEATURE_CLASSES].dropna(axis=0, how='any')
    return train, test


def preprocess_data(data, features):
    new_features =  []
    for feature in features:
        data, new_features_batch = label_to_onehot(data, feature)
        new_features.extend(new_features_batch)
    x, y = data, data.pop('survived')
    return (x, y), new_features


def label_to_onehot(data, feature_name):
    # Use sklearn's encoders to turn the Sex feature into a one-hot vector
    label_encoder = LabelEncoder()
    numerical = label_encoder.fit_transform(data[feature_name])
    numerical_classes = label_encoder.classes_
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot = onehot_encoder.fit_transform(numerical.reshape(-1,1))

    # Add new one-hot features to the dataset and drop the old feature
    new_features = {}
    for count, label in enumerate(numerical_classes):
        new_features[feature_name + '_' + label] = onehot[:,count]
    data = data.assign(**new_features)
    data = data.drop(feature_name,axis=1)

    return data, new_features.keys()


def train_input_fn(features, labels, batch_size=100):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(2000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size=100):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":
    tf.app.run(main)
