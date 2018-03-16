import tensorflow as tf
import os
import math
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from clusterone import get_data_path

LOCAL_DATA_PATH = os.path.abspath(os.path.expanduser('~/Documents/data/mnist'))

# Storage directory for the MNIST dataset. 
# Returns LOCAL_DATA_PATH when running locally, '/data/malo/mnist' when running on Clusterone.
data_dir = get_data_path(
                        dataset_name = "malo/mnist",
                        local_root = LOCAL_DATA_PATH,
                        local_repo = "mnist",
                        path = ''
                        )

# The MNIST dataset has 10 classes, representing the digits 0 through 9
NUM_CLASSES = 10

# The MNIST images are 28x28 pixels in size
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Each hidden layer gets 128 neurons
hidden1_units = 128
hidden2_units = 128

# Further hyperparameters
learning_rate = 0.1
batch_size = 100
train_steps = 10000


def main(argv):
    ### Load the data using TensorFlow's MNIST tutorial function read_data_sets()
    data = read_data_sets(data_dir,
            one_hot=False,
            fake_data=False)

    ### Create the input variables for images and their labels
    images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    labels = tf.placeholder(tf.float32, [None])

    ### Build the neural network. It consists of two hidden layers with ReLu activation functions and a linear output layer.
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

    ### Define the loss calculation based on the labels
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.to_int64(labels), logits=logits)

    ### Define the training operation
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    ### Create the session object and initialize the variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    ### Train the network
    print('Start training...')
    for _ in range(train_steps):
        xs, ys = data.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={images: xs, labels: ys})

    print('Training complete.')

    ### Evaulate the trained model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={images: data.test.images,
                                        labels: data.test.labels}))


if __name__ == "__main__":
    tf.app.run()