import tensorflow as tf
import numpy as np
import os
import math
import cv2
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from clusterone import get_data_path, get_logs_path

LOCAL_DATA_PATH = os.path.abspath(os.path.expanduser('~/Documents/data/mnist'))
LOCAL_LOGS_PATH = os.path.abspath(os.path.expanduser('~/Documents/tf_logs/mnist-tb-img/'))

data_dir = get_data_path(
                        dataset_name = "malo/mnist",
                        local_root = LOCAL_DATA_PATH,
                        local_repo = "mnist",
                        path = ''
                        )

logs_dir = get_logs_path(LOCAL_LOGS_PATH)
print('When runnning locally, start TensorBoard with: tensorboard --logdir %s' % logs_dir)

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
TB_IMAGE_PIXELS = IMAGE_PIXELS * 2

hidden1_units = 128
hidden2_units = 128

learning_rate = 0.5
batch_size = 100
train_steps = 5500


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


# def get_wrong_predictions(predictions):
#     # Todo: Make a generatir 
#     wrong_predictions = [count for count, pred in enumarate(predictions) if not pred] 
#     wrong_predictions = []
#     for count, pred in enumarate(predictions):
#         if not pred:
#             wrong_predictions.append(count)
#     return wrong_predictions


def get_wrong_images(predictions_bool, predictions, data):
    wrong_predictions = [count for count, p in enumerate(predictions_bool) if not p]
    wrong_images = np.zeros((len(wrong_predictions), TB_IMAGE_PIXELS))
    for count, index in enumerate(wrong_predictions):
        img = data.test.images[index]
        img = np.append(img, np.zeros(IMAGE_PIXELS))
        cv2.putText(img, str(predictions[index]),(0,0), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2, cv2.LINE_AA)
        # print(np.shape(img))
        # print(np.shape(wrong_images))
        wrong_images[count] = 1-img
        # print("Images with index %s was classified as %s, but is labeled as %s." % (index, predictions[index], data.test.labels[index]))
    # image_summary_op = tf.summary.image('images', tf.reshape(wrong_images, [-1, 28, 28, 1]), min(len(wrong_predictions), 25))
    print("%s images have been incorrectly classified." % len(wrong_predictions))
    image_summary_op = tf.summary.image('images', tf.reshape(wrong_images, [-1, IMAGE_SIZE*2, IMAGE_SIZE, 1]), 10)
    return image_summary_op, wrong_images


def main(argv):
    data = read_data_sets(data_dir,
            one_hot=False,
            fake_data=False)

    with tf.name_scope('input'):
        images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS], name='images')
        labels = tf.placeholder(tf.float32, [None], name='labels')

    with tf.name_scope('tensorboard_input'):
        tb_images = tf.placeholder(tf.float32, [None, TB_IMAGE_PIXELS], name='tb_images')

    logits = bare_inference(images)
    loss_ = loss(logits, labels)
    tf.summary.scalar('Loss', loss_)
    train_op = bare_training(loss_, learning_rate)

    with tf.name_scope('Accuracy'):
        predictions_op = tf.argmax(logits, 1)
        predictions_bool_op = tf.equal(predictions_op, tf.to_int64(labels))
        accuracy_op = tf.reduce_mean(tf.cast(predictions_bool_op, tf.float32))
        tf.summary.scalar('Accuracy', accuracy_op)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # writer = tf.summary.FileWriter(logs_dir, graph=sess.graph)
    train_writer = tf.summary.FileWriter(logs_dir + '/train', graph=sess.graph)
    test_writer = tf.summary.FileWriter(logs_dir + '/test')
    summary_op = tf.summary.merge_all()

    for i in range(train_steps):
        if i % 10 == 0:  # execute every 10th iteration
            # TODO: remove preditions_bool_op here and test
            test_summary, pred_bool, acc = sess.run([summary_op, predictions_bool_op, accuracy_op], feed_dict={images: data.test.images, labels: data.test.labels})
            test_writer.add_summary(test_summary, i)
            if i % 100 == 0:
                print('Accuracy at step %s: %s' % (i, acc))
        elif i == train_steps-1:
            print("The end is here.")
            pred, pred_bool = sess.run([predictions_op, predictions_bool_op], feed_dict={images: data.test.images, labels: data.test.labels}) # do I really need this?
            print(pred_bool)
            image_summary_op, wrong_images = get_wrong_images(pred_bool, pred, data)
            image_sum = sess.run(image_summary_op, feed_dict={tb_images: wrong_images})
            test_writer.add_summary(image_sum, i)
                
        else:
            xs, ys = data.train.next_batch(batch_size)
            train_summary, _ = sess.run([summary_op, train_op], feed_dict={images: xs, labels: ys})
            train_writer.add_summary(train_summary, i)


if __name__ == "__main__":
    tf.app.run()