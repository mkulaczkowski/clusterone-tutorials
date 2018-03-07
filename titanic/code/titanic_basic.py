import tensorflow as tf
import pandas as pd
import argparse
import os
import numpy as np

from clusterone import get_data_path, get_logs_path
from pprint import pprint

FEATURE_CLASSES = ['pclass','age', 'survived']

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default="../data/titanic_train.csv", type=str, help='Path to training data file')
parser.add_argument('--test_path', default="../data/titanic_test.csv", type=str, help='Path to test data file')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    log_path = get_logs_path(root=os.path.abspath(os.path.expanduser('~/Documents/tf_logs/logs/titanic_basic')))

    (train_x, train_y), (test_x, test_y) = load_data(args.train_path, args.test_path)

    passenger_features = []
    passenger_features.append(tf.feature_column.numeric_column(key='pclass'))
    passenger_features.append(tf.feature_column.numeric_column(key='age'))

    classifier = tf.estimator.DNNClassifier(
                                            hidden_units=[20, 20, 20], 
                                            feature_columns=passenger_features, 
                                            model_dir=log_path,
                                            n_classes=2)

    classifier.train(input_fn=lambda:train_input_fn(train_x, train_y), steps=1000)

    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y))

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
    train_x, train_y = train, train.pop('survived')

    te_path, te_filename = os.path.split(test_path)
    test_dir = get_data_path(
                            dataset_name = "svenchmie/titanic_data/titanic_test.csv",
                            local_root = te_path,
                            local_repo = te_filename,
                            path = ''
                            )

    test = pd.read_csv(test_dir[:-1], engine="python")
    test = test[FEATURE_CLASSES].dropna(axis=0, how='any')
    test_x, test_y = test, test.pop('survived')
    return (train_x, train_y), (test_x, test_y)

    return data

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
