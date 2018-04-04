import tensorflow as tf
import argparse
import os
import math
import shutil
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from clusterone import get_data_path, get_logs_path

LOCAL_DATA_PATH = os.path.abspath(os.path.expanduser('../data/'))
LOCAL_LOGS_PATH = os.path.abspath(os.path.expanduser('logs/'))

# Storage directory for the MNIST dataset. 
# Returns LOCAL_DATA_PATH when running locally, '/data/malo/mnist' when running on Clusterone.
data_dir = get_data_path(
                        dataset_name = "malo/mnist",
                        local_root = LOCAL_DATA_PATH,
                        local_repo = "mnist",
                        path = ''
                        )

# Storage dictory for the log files produced by this script.
logs_dir = get_logs_path(LOCAL_LOGS_PATH)

# The MNIST dataset has 10 classes, representing the digits 0 through 9
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Each hidden layer gets 128 neurons
hidden1_units = 128
hidden2_units = 128

# Further hyperparameters
learning_rate = 0.1
batch_size = 100
train_steps = 10000

# Configure command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--keeplogs' '-k', action='store_true', help="Don't delete log files from previous runs")


def variable_summaries(var):
    """ Calculate a variety of TensorBoard summary values for each tensor.
        This code is taken from the TensorFlow tutorial for TensorBoard,
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main(argv):
    ### Parse command line arguments
    args = parser.parse_args(argv[1:])

    print('Storing data in: %s' % data_dir)
    print('Storing logs in: %s' % logs_dir)

    ### Prepare logs directory
    if not args.keeplogs_k and os.path.isdir(logs_dir):
        print('Found previous log files. Deleting...')
        shutil.rmtree(logs_dir)

    print('When runnning locally, start TensorBoard with: tensorboard --logdir %s' % logs_dir)

    ### Load the data using TensorFlow's MNIST tutorial function read_data_sets()
    data = read_data_sets(data_dir,
            one_hot=False,
            fake_data=False)

    ### Create the input variables for images and their labels
    with tf.name_scope('input'):
        images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS], name='images')
        labels = tf.placeholder(tf.float32, [None], name='labels')

    ### Build the neural network. It consists of two hidden layers with ReLu activation functions and a linear output layer.
    # Hidden layer 1
    with tf.name_scope('hidden1'):
        with tf.name_scope('weights'):
            weights1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights1')
            variable_summaries(weights1)
        with tf.name_scope('biases'):
            biases1 = tf.Variable(tf.zeros([hidden1_units]), name='biases1')
            variable_summaries(biases1)
        with tf.name_scope('activation_relu'):
            hidden1 = tf.nn.relu(tf.matmul(images, weights1) + biases1)
            tf.summary.histogram('activations', hidden1)

    # Hidden layer 2
    with tf.name_scope('hidden2'):
        with tf.name_scope('weights'):
            weights2 = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights2')
            variable_summaries(weights2)
        with tf.name_scope('biases'):
            biases2 = tf.Variable(tf.zeros([hidden2_units]), name='biases2')
            variable_summaries(biases2)
        with tf.name_scope('activation_rel'):
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)
            tf.summary.histogram('activations', hidden2)

    # Linear
    with tf.name_scope('linear'):
        with tf.name_scope('weights'):
            weights_linear = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights_linear')
            variable_summaries(weights_linear)
        with tf.name_scope('biases'):
            biases_linear = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_linear')
            variable_summaries(biases_linear)
        with tf.name_scope('activation_linear'):
            logits = tf.matmul(hidden2, weights_linear) + biases_linear
            tf.summary.histogram('activations', logits)

    ### Define the loss calculation based on the labels
    with tf.name_scope('cross_entropy'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.to_int64(labels), logits=logits)
    tf.summary.scalar('Loss', loss)

    ### Define the training operation
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    ### Define the accuracy calculation
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    ### Create the session object and initialize the variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    ### Create summary writers for test and train operations
    train_writer = tf.summary.FileWriter(logs_dir + '/train', graph=sess.graph)
    test_writer = tf.summary.FileWriter(logs_dir + '/test')
    summary_op = tf.summary.merge_all()

    print('Start training...')

    ### Train the model
    for i in range(train_steps):

        # Every 10th iteration, calculate accuracy and write to summary for TensorBoard
        if i % 10 == 0:
            test_summary, acc = sess.run([summary_op, accuracy], feed_dict={images: data.test.images, labels: data.test.labels})
            test_writer.add_summary(test_summary, i)

            # Every 100th iteration, print accuracy to console
            if i % 100 == 0:
                print('Accuracy at step %s: %s' % (i, acc))
        
        # Training step is executed here
        else:
            xs, ys = data.train.next_batch(batch_size)
            train_summary, _ = sess.run([summary_op, train_op], feed_dict={images: xs, labels: ys})
            train_writer.add_summary(train_summary, i)

    print('Training complete.')


if __name__ == "__main__":
    tf.app.run()