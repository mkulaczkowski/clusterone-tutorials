import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.logging.set_verbosity(tf.logging.INFO)


def get_args():
    """Parse arguments"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='''Train a convolution neural network with MNIST dataset.
                            For distributed mode, the script will use few environment variables as defaults:
                            JOB_NAME, TASK_INDEX, PS_HOSTS, and WORKER_HOSTS. These environment variables will be
                            available on distributed Tensorflow jobs on Clusterone platform by default.
                            If running this locally, you will need to set these environment variables
                            or pass them in as arguments (i.e. python mnist.py --job_name worker --task_index 0
                            --worker_hosts "localhost:2222,localhost:2223" --ps_hosts "localhost:2224").
                            If these are not set, the script will run in non-distributed (single instance) mode.''')

    # Configuration for distributed task
    parser.add_argument('--job_name', type=str, default=os.environ.get('JOB_NAME', None), choices=['worker', 'ps'],
                        help='Task type for the node in the distributed cluster. Worker-0 will be set as master.')
    parser.add_argument('--task_index', type=int, default=os.environ.get('TASK_INDEX', 0),
                        help='Worker task index, should be >= 0. task_index=0 is the chief worker.')
    parser.add_argument('--ps_hosts', type=str, default=os.environ.get('PS_HOSTS', ''),
                        help='Comma-separated list of hostname:port pairs.')
    parser.add_argument('--worker_hosts', type=str, default=os.environ.get('WORKER_HOSTS', ''),
                        help='Comma-separated list of hostname:port pairs.')

    # Experiment related parameters
    parser.add_argument('--local_data_dir', type=str, default='data/',
                        help='Path to local data directory')
    parser.add_argument('--local_log_dir', type=str, default='logs/',
                        help='Path to local log directory')

    # Training params
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate used in Adam optimizer.')
    parser.add_argument('--learning_decay', type=float, default=0.001,
                        help='Exponential decay rate of the learning rate per step.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size to use during training and evaluation.')
    opts = parser.parse_args()

    # Clusterone snippet: Grabs the correct paths, depending on if the job is running local or on Clusterone
    opts.data_dir = get_data_path(dataset_name='',
                                  local_root=opts.local_data_dir,
                                  local_repo='',
                                  path='')
    opts.log_dir = get_logs_path(root=opts.local_log_dir)

    return opts


def make_tf_config(opts):
    """
    Returns TF_CONFIG that can be used to set the environment variable necessary for distributed training
    """
    if all([opts.job_name is None, not opts.ps_hosts, not opts.worker_hosts]):
        return {}
    elif any([opts.job_name is None, not opts.ps_hosts, not opts.worker_hosts]):
        tf.logging.warn('Distributed setting is incomplete. You must pass job_name, ps_hosts, and worker_hosts.')
        if opts.job_name is None:
            tf.logging.warn('Expected job_name of worker or ps. Received {}.'.format(opts.job_name))
        if not opts.ps_hosts:
            tf.logging.warn('Expected ps_hosts, list of hostname:port pairs. Got {}. '.format(opts.ps_hosts) +
                            'Example: --ps_hosts "localhost:2224" or --ps_hosts "localhost:2224,localhost:2225')
        if not opts.worker_hosts:
            tf.logging.warn('Expected worker_hosts, list of hostname:port pairs. Got {}. '.format(opts.worker_hosts) +
                            'Example: --worker_hosts "localhost:2222,localhost:2223"')
        tf.logging.warn('Ignoring distributed arguments. Running single mode.')
        return {}

    tf_config = {
        'task': {
            'type': opts.job_name,
            'index': opts.task_index
        },
        'cluster': {
            'master': [opts.worker_hosts[0]],
            'worker': opts.worker_hosts,
            'ps': opts.ps_hosts
        },
        'environment': 'cloud'
    }

    # Nodes may need to refer to itself as localhost
    local_ip = 'localhost:' + tf_config['cluster'][opts.job_name][opts.task_index].split(':')[1]
    tf_config['cluster'][opts.job_name][opts.task_index] = local_ip
    if opts.job_name == 'worker' and opts.task_index == 0:
        tf_config['task']['type'] = 'master'
        tf_config['cluster']['master'][0] = local_ip
    return tf_config


def keras_model(opts):
    """Return a CNN Keras model"""
    input_tensor = tf.keras.layers.Input(shape=(784,), name='input')

    temp = tf.keras.layers.Reshape([28, 28, 1], name='input_image')(input_tensor)
    for i, n_units in enumerate([32, 64]):
        temp = tf.keras.layers.Conv2D(n_units, kernel_size=3, strides=(2, 2),
                                      activation='relu', name='cnn'+str(i))(temp)
        temp = tf.keras.layers.Dropout(0.5, name='dropout'+str(i))(temp)
    temp = tf.keras.layers.GlobalAvgPool2D(name='average')(temp)
    output = tf.keras.layers.Dense(10, activation='softmax', name='output')(temp)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
    optimizer = tf.keras.optimizers.Adam(lr=opts.learning_rate, decay=opts.learning_decay)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def main(opts):
    """Main function"""
    data = read_data_sets(opts.data_dir,
                          one_hot=False,
                          fake_data=False)

    model = keras_model(opts)
    config = tf.estimator.RunConfig(
                model_dir=opts.log_dir,
                save_summary_steps=1,
                save_checkpoints_steps=100,
                keep_checkpoint_max=3,
                log_step_count_steps=10)
    estimator = tf.keras.estimator.model_to_estimator(model, model_dir=opts.log_dir, config=config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.train.images},
                         y=data.train.labels,
                         num_epochs=None,
                         batch_size=opts.batch_size,
                         shuffle=True,
                         queue_capacity=10*opts.batch_size,
                         num_threads=4)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.test.images},
                         y=data.test.labels,
                         num_epochs=1,
                         shuffle=False,
                         queue_capacity=10*opts.batch_size,
                         num_threads=4)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1e6)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=60)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    opts = get_args()

    # Clusterone snippet: Set environment variable TF_CONFIG for distributed training
    TF_CONFIG = make_tf_config(opts)
    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)

    main(opts)
