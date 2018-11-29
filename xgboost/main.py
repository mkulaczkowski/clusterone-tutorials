import glob
import logging
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import xgboost as xgb
from clusterone import get_logs_path

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Parse args"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='''Train a logistic regressor using XGBoost.
                            For distributed mode, use dmlc-core submit.
                            ''')

    # Experiment related parameters
    parser.add_argument('--data_dir', type=str, default=os.path.join(FILE_DIR, 'data'),
                        help='Directory where your data files are.')
    parser.add_argument('--local_log_root', type=str, default=os.path.join(FILE_DIR, 'logs'),
                        help='Path to store logs and checkpoints. This path will be /logs on Clusterone.')
    parser.add_argument('--train_file_pattern', type=str, default='*.train',
                        help='Use * as wildcard. Describe sub-directory/filename pattern for train data.')
    parser.add_argument('--test_file_pattern', type=str, default='*.test',
                        help='Use * as wildcard. Describe sub-directory/filename pattern for test data.')
    parser.add_argument('--model_name', type=str, default='saved.model',
                        help='Filename to use for saved model.')

    # General params
    parser.add_argument('--silent', type=int, default=0, choices=[0, 1],
                        help='0 means printing running messages, 1 means silent mode')

    # Booster params
    parser.add_argument('--eta', type=float, default=0.3,
                        help='Step size shrinkage used in update to prevents overfitting. '
                             'After each boosting step, we can directly get the weights of new features, '
                             'and eta shrinks the feature weights to make the boosting process more conservative. '
                             'Range: [0,1]')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Minimum loss reduction required to make a further partition on a leaf node of the tree. '
                             'The larger gamma is, the more conservative the algorithm will be. Range: [0,inf]')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Maximum depth of a tree. Increasing this value will make the model more complex and more '
                             'likely to overfit. 0 indicates no limit. Note that limit is required when grow_policy is '
                             'set of depthwise.')
    parser.add_argument('--min_child_weight', type=float, default=1.0,
                        help='Minimum sum of instance weight (hessian) needed in a child. If the tree partition step '
                             'results in a leaf node with the sum of instance weight less than min_child_weight, then '
                             'the building process will give up further partitioning. In linear regression task, this '
                             'simply corresponds to minimum number of instances needed to be in each node. The larger '
                             'min_child_weight is, the more conservative the algorithm will be.')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would '
                             'randomly sample half of the training data prior to growing trees. and this will prevent '
                             'overfitting. Subsampling will occur once in every boosting iteration.')
    parser.add_argument('--colsample_bytree', type=float, default=1.0,
                        help='Subsample ratio of columns when constructing each tree. Subsampling will occur once in '
                             'every boosting iteration.')
    parser.add_argument('--l2', type=float, default=1.0,
                        help='L2 regularization term on weights. Increasing this value will make model more '
                             'conservative.')
    parser.add_argument('--l1', type=float, default=0.0,
                        help='L1 regularization term on weights. Increasing this value will make model more '
                             'conservative.')
    parser.add_argument('--tree_method', type=str, default='auto',
                        choices=['auto', 'exact', 'approx', 'hist', 'gpu_exact', 'gpu_hist'],
                        help='The tree construction algorithm used in XGBoost. '
                             'Distributed and external memory version only support tree_method=approx.')
    parser.add_argument('--scale_pos_weight', type=float, default=1.,
                        help='Control the balance of positive and negative weights, useful for unbalanced classes. '
                             'A typical value to consider: sum(negative instances) / sum(positive instances).')

    # Learning task parameters
    parser.add_argument('--objective', type=str, default='binary:logistic',
                        choices=['binary:logistic', 'binary:logitraw'],
                        help='See https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters')
    parser.add_argument('--eval_metric', type=str, nargs='*', default=['error'],
                        choices=['logloss', 'auc', 'error'],
                        help='Evaluation metrics for validation data. '
                             'See https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random number seed')

    # Command line parameters
    parser.add_argument('--num_round', type=int, default=10,
                        help='The number of rounds for boosting')

    # Train params
    parser.add_argument('--cache_data', action='store_true',
                        help='Use external memory version')

    # Testing/Debugging
    parser.add_argument('--set_verbosity', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging verbosity level')

    # Parse args
    opts = parser.parse_args()

    opts.train_file_pattern = os.path.join(opts.data_dir, opts.train_file_pattern)
    opts.test_file_pattern = os.path.join(opts.data_dir, opts.test_file_pattern)
    train_files = glob.glob(opts.train_file_pattern)
    test_files = glob.glob(opts.test_file_pattern)

    if train_files:
        opts.train_data = train_files[0]
        if len(train_files) > 1:
            logging.warning('Detected multiple files. Using {}.'.format(opts.train_data))
    else:
        raise IOError('Did not detect any files with train_file_pattern "{}"'.format(opts.train_file_pattern))

    if test_files:
        opts.test_data = test_files[0]
        if len(test_files) > 1:
            logging.warning('Detected multiple files. Using {}.'.format(opts.test_data))
    else:
        raise IOError('Did not detect any files with test_file_pattern "{}"'.format(opts.test_file_pattern))

    opts.log_dir = get_logs_path(root=opts.local_log_root)

    return opts


def main(opts):
    """Loads data and runs XGBoost model"""
    logging.info('Loading train data from {}'.format(opts.train_data))

    if opts.cache_data:
        dtrain = xgb.DMatrix(opts.train_data + '#dtrain.cache', missing=0.)
        dtest = xgb.DMatrix(opts.test_data + '#dtest.cache', missing=0.)
    else:
        dtrain = xgb.DMatrix(opts.train_data, missing=0.)
        dtest = xgb.DMatrix(opts.test_data, missing=0.)

    logging.debug('Train data shape: {}'.format((dtrain.num_row(), dtrain.num_col())))
    logging.debug('Test data shape: {}'.format((dtest.num_row(), dtest.num_col())))

    params = {
        'silent': opts.silent,
        'eta': opts.eta,
        'gamma': opts.gamma,
        'max_depth': opts.max_depth,
        'min_child_weight': opts.min_child_weight,
        'subsample': opts.subsample,
        'colsample_bytree': opts.colsample_bytree,
        'lambda': opts.l2,
        'alpha': opts.l1,
        'scale_pos_weight': opts.scale_pos_weight,
        'objective': opts.objective,
        'eval_metric': opts.eval_metric,
        'seed': opts.seed
    }

    logging.info('Parameters: {}'.format(params))

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    logging.info('Training...')

    stime_train = time.time()
    model = xgb.train(params, dtrain, opts.num_round, evallist)

    logging.info('Training finished. Took {:.2f} seconds'.format(time.time()-stime_train))

    if xgb.rabit.get_rank() == 0:
        if not os.path.exists(opts.log_dir):
            os.makedirs(opts.log_dir)
        fp_output = os.path.join(opts.log_dir, opts.model_name)
        model.save_model(fp_output)
        logging.info('Model saved to {}'.format(fp_output))
    else:
        time.sleep(3)


if __name__ == '__main__':
    stime = time.time()

    args = parse_args()
    logging.basicConfig(level=args.set_verbosity)

    logging.debug('=' * 20 + ' Environment Variables ' + '=' * 20)
    for k, v in sorted(os.environ.items()):
        logging.debug('{}: {}'.format(k, v))

    logging.debug('=' * 20 + ' Arguments ' + '=' * 20)
    for k, v in sorted(args.__dict__.items()):
        if v is not None:
            logging.debug('{}: {}'.format(k, v))

    # MPI::Init
    logging.info('Initiating rabit.')
    xgb.rabit.init()

    # Some checks on world size, ranks and hostnames
    args.n_workers = xgb.rabit.get_world_size()
    args.rank = xgb.rabit.get_rank()
    logging.info('MPI world-size: '+str(args.n_workers))
    logging.info('MPI get-rank  : '+str(args.rank))
    logging.info('MPI hostname  : '+str(xgb.rabit.get_processor_name()))

    # Run model
    logging.info('Start main routine.')
    main(args)

    # MPI::Finalize
    logging.info('Terminating rabit.')
    xgb.rabit.finalize()

    logging.info('End-to-end time: {} seconds'.format(time.time()-stime))
