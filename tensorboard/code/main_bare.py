import tensorflow as tf
import os
import math
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from clusterone import get_data_path

LOCAL_DATA_PATH = os.path.abspath(os.path.expanduser('~/Documents/data/mnist'))

data_dir = get_data_path(
                        dataset_name = "malo/mnist",
                        local_root = LOCAL_DATA_PATH,
                        local_repo = "mnist",
                        path = ''
                        )

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

hidden1_units = 128
hidden2_units = 128

learning_rate = 0.1
batch_size = 100


def bare_inference(images):
    # Building model based on TF mnist 
    # Hidden layer 1
    weights1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))))
    biases1 = tf.Variable(tf.zeros([hidden1_units]))
    hidden1 = tf.nn.relu(tf.matmul(images, weights1) + biases1)

    # Hidden layer 2
    weights2 = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))))
    biases2 = tf.Variable(tf.zeros([hidden2_units]))
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

    # Linear
    weights_linear = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2_units))))
    biases_linear = tf.Variable(tf.zeros([NUM_CLASSES]))
    logits = tf.matmul(hidden2, weights_linear) + biases_linear

    return logits

def loss(logits, labels):
    labels = tf.to_int64(labels) 
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def bare_training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def main(argv):
    data = read_data_sets(data_dir,
            one_hot=False,
            fake_data=False)

    print(data.train)

    images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    labels = tf.placeholder(tf.float32, [None])

    logits = bare_inference(images)
    loss_ = loss(logits, labels)
    train = bare_training(loss_, learning_rate)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(55000):
        xs, ys = data.train.next_batch(batch_size)
        sess.run(train, feed_dict={images: xs, labels: ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={images: data.test.images,
                                        labels: data.test.labels}))


if __name__ == "__main__":
    tf.app.run()