import tensorflow as tf
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.tensorboard.plugins import projector
from clusterone import get_data_path, get_logs_path

LOCAL_DATA_PATH = os.path.abspath(os.path.expanduser('~/Documents/data/mnist'))
LOCAL_LOGS_PATH = os.path.abspath(os.path.expanduser('~/Documents/tf_logs/mnist-tb/'))

data_dir = get_data_path(
                        dataset_name = "malo/mnist",
                        local_root = LOCAL_DATA_PATH,
                        local_repo = "mnist",
                        path = ''
                        )

logs_dir = get_logs_path(LOCAL_LOGS_PATH)

# constants for embeddings
PATH_TO_EMBEDDING_SPRITES = os.path.join(logs_dir,'mnistdigits.png')
PATH_TO_EMBEDDING_METADATA = os.path.join(logs_dir,'metadata.tsv')
NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
TO_EMBED_COUNT = 10000

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

hidden1_units = 128
hidden2_units = 128

learning_rate = 0.1
batch_size = 100


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization). Stolen from TF tutorial. """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def bare_inference(images):
    # Building model based on TF mnist 
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

    return logits

def loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        return tf.losses.sparse_softmax_cross_entropy(labels=tf.to_int64(labels), logits=logits)

def bare_training(loss, learning_rate):
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def prep_embeddings(data, writer):
    # Let's build the embeddings:
    batch_xs, batch_ys = data.train.next_batch(TO_EMBED_COUNT)
    embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALISE_VARIABLE)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    embedding.metadata_path = PATH_TO_EMBEDDING_METADATA

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = PATH_TO_EMBEDDING_SPRITES
    embedding.sprite.single_image_dim.extend([28,28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(writer, config)

    return batch_xs, batch_ys

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits

def main(argv):
    data = read_data_sets(data_dir,
            one_hot=False,
            fake_data=False)

    with tf.name_scope('input'):
        images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS], name='images')
        labels = tf.placeholder(tf.float32, [None], name='labels')

    logits = bare_inference(images)
    loss_ = loss(logits, labels)
    tf.summary.scalar('Loss', loss_)
    train_op = bare_training(loss_, learning_rate)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    sess = tf.InteractiveSession()

    # writer = tf.summary.FileWriter(logs_dir, graph=sess.graph)
    train_writer = tf.summary.FileWriter(logs_dir + '/train', graph=sess.graph)
    test_writer = tf.summary.FileWriter(logs_dir + '/test')
    summary_op = tf.summary.merge_all()

    batch_xs, batch_ys = prep_embeddings(data, train_writer)

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(logs_dir, "model.ckpt"), 1)
    to_visualize = vector_to_matrix_mnist(batch_xs)
    to_visualize = invert_grayscale(to_visualize)

    sprite_image = create_sprite_image(to_visualize)

    plt.imsave(PATH_TO_EMBEDDING_SPRITES, sprite_image, cmap='gray')
    plt.imshow(sprite_image,cmap='gray')

    with open(PATH_TO_EMBEDDING_METADATA, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(batch_ys):
            f.write("%d\t%d\n" % (index, label))


    for i in range(10000):
        if i % 10 == 0:  # execute every 10th iteration
            test_summary, acc = sess.run([summary_op, accuracy], feed_dict={images: data.test.images, labels: data.test.labels})
            test_writer.add_summary(test_summary, i)
            if i % 100 == 0:
                print('Accuracy at step %s: %s' % (i, acc))
        else:
            xs, ys = data.train.next_batch(batch_size)
            train_summary, _ = sess.run([summary_op, train_op], feed_dict={images: xs, labels: ys})
            train_writer.add_summary(train_summary, i)


if __name__ == "__main__":
    tf.app.run()