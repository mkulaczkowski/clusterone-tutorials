import logging
import os
import random
import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.distributed as dist
from clusterone import get_logs_path

sys.path.append(os.path.dirname(__file__))
from model import CycleGAN
from data import ImageDataset, get_transforms, pil_loader


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--local_log_dir', type=str, default=os.path.abspath('./logs'),
                        help='Where checkpoints will be saved')

    # Model params
    parser.add_argument('--netD', type=str, default='basic', choices=['basic', 'n_layers', 'pixel'],
                        help='Selects model to use for discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9blocks',
                        choices=['resnet_9blocks', 'resnet_6blocks', 'unet_128', 'unet_256'],
                        help='Selects model to use for generator')
    parser.add_argument('--n_layers_D', type=int, default=3,
                        help='only used if netD==n_layers')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of filters in the initial hidden layer for the Generator network')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Number of filters in the initial hidden layer for the Discriminator network')
    parser.add_argument('--norm', type=str, default='instance', choices=['instance', 'batch', 'none'],
                        help='Normalization type')
    parser.add_argument('--no_dropout', type=bool, default=False,
                        help='Do not use dropout')
    parser.add_argument('--lambda_A', type=float, default=10.0,
                        help='Weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0,
                        help='Weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='Use identity mapping. Setting lambda_identity other than 0 has an effect of scaling '
                             'the weight of the identity mapping loss. For example, if the weight of the identity loss '
                             'should be 10 times smaller than the weight of the reconstruction loss, please set lambda_'
                             'identity = 0.1')

    # Training params
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to images (should have subdirectories trainA, trainB, valA, valB, etc.')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='First epoch number')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Coefficient used for computing running averages of gradient (see torch.optim.Adam)')
    parser.add_argument('--print_steps', type=int, default=10,
                        help='How frequently losses will be printed')
    parser.add_argument('--save_steps', type=int, default=10,
                        help='How frequently the model will be saved')
    parser.add_argument('--save_epochs', type=int, default=0,
                        help='Save after N epoch in addition to regular checkpoints. 0 means do not save.')
    parser.add_argument('--max_ckpts', type=int, default=3,
                        help='Number of maximum of checkpoints to save')
    parser.add_argument('--n_test_images', type=int, default=3,
                        help='Number of randomly selected images to be used as test at each checkpoint')
    parser.add_argument('--save_new_images_every_epoch', type=bool, default=False,
                        help='Save generated test images at end of each epoch with different filename instead of '
                             'overwriting old ones. NOT recommended.')

    # Inference params
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='/path/to/#_net_G_B.pth')
    parser.add_argument('--infer_image_path', type=str, default=None,
                        help='/path/to/image.jpeg')

    # Runtime params
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train',
                        help='Train or inference mode')
    parser.add_argument('--gpu', type=bool, default=True,
                        help='Use GPU if available when True. If False, will not use GPU even if it is available')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of subprocesses to use for data loading. 0 means that the data will be loaded in '
                             'the main process')

    # Debug params
    parser.add_argument('--verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
                        help='Logging verbosity.')

    # Distributed params
    master = 'localhost:5000' if os.environ.get('IS_MASTER', 'True') == 'True' else \
             os.environ.get('MASTER', 'localhost:5000')
    parser.add_argument('--master', type=str, default=master,
                        help='Address:port of master node')
    parser.add_argument('--world_size', type=int, default=len(os.environ.get('WORKER_HOSTS', '').split(',')),
                        help='Number of nodes in cluster')
    parser.add_argument('--rank', type=int, default=os.environ.get('TASK_INDEX', 0),
                        help='Rank of the node in the cluster')

    opts = parser.parse_args()

    if opts.mode == 'train' and opts.data_dir is None:
        raise IOError('For train mode, you must provide data directory!')

    opts.log_dir = get_logs_path(root=opts.local_log_dir)

    if opts.mode == 'train':
        opts.path_trainA = os.path.join(opts.data_dir, 'trainA')
        opts.path_trainB = os.path.join(opts.data_dir, 'trainB')

    opts.gpu = opts.gpu and torch.cuda.is_available()

    if opts.gpu:
        opts.device = torch.device('cuda')
        opts.gpu_ids = [i for i in range(torch.cuda.device_count())]
    else:
        opts.device = torch.device('cpu')
        opts.gpu_ids = []

    opts.distributed = opts.world_size > 1

    return opts


def train(model, opts):
    logging.info('Initializing datasets')
    dataset_A = ImageDataset(opts.path_trainA, transform=get_transforms(opts))
    dataset_B = ImageDataset(opts.path_trainB, transform=get_transforms(opts))

    if opts.distributed:
        sampler_A = torch.utils.data.distributed.DistributedSampler(dataset_A)
    else:
        sampler_A = torch.utils.data.sampler.RandomSampler(dataset_A)
    loader_A = torch.utils.data.DataLoader(dataset_A, batch_size=opts.batch_size, sampler=sampler_A,
                                           num_workers=opts.num_workers, pin_memory=True)

    if torch.__version__ >= '1':
        sampler_B = torch.utils.data.RandomSampler(dataset_B, num_samples=opts.batch_size, replacement=True)
        loader_B = torch.utils.data.DataLoader(dataset_B, sampler=sampler_B, num_workers=opts.num_workers,
                                               pin_memory=True)
    else:
        sampler_B = torch.utils.data.RandomSampler(dataset_B)
        loader_B = torch.utils.data.DataLoader(dataset_B, batch_size=opts.batch_size, sampler=sampler_B,
                                               num_workers=opts.num_workers, pin_memory=True)

    if opts.rank == 0:
        test_indices = random.sample(range(len(dataset_A)), k=opts.n_test_images)
        opts.test_image_names = [os.path.basename(dataset_A.image_paths[i]) for i in test_indices]
        if test_indices:
            test_images = torch.stack([dataset_A[i] for i in test_indices]).to(opts.device)
            logging.info('Test images to be used: ' + ', '.join(opts.test_image_names))
        else:
            test_images = []
    else:
        opts.test_image_names = []
        test_images = []

    logging.info('Start training')
    step = last_step = 1
    for epoch in range(opts.start_epoch, opts.start_epoch + opts.epochs):
        if opts.distributed:
            sampler_A.set_epoch(epoch)

        for step, batch_A in enumerate(loader_A, last_step):
            batch_A = batch_A.to(opts.device)
            batch_B = next(iter(loader_B)).to(opts.device)

            model.optimize_step(batch_A, batch_B)
            if opts.rank == 0 and step % opts.print_steps == 0:
                losses = ', '.join('{}: {:.4f}'.format(*loss) for loss in model.losses.items())
                logging.info('Step {} --- {}'.format(step, losses))

            if opts.rank == 0 and step % opts.save_steps == 0:
                logging.info('Saving latest checkpoints (step {})...'.format(step))
                checkpoint_step(model, test_images, step, opts)

        if opts.rank == 0 and opts.save_epochs and epoch % opts.save_epochs == 0:
            logging.info('Saving checkpoint for epoch {} (step {})'.format(epoch, step))
            checkpoint_step(model, test_images, step, opts)

        last_step = step


def checkpoint_step(model, test_images, step, opts):
    model.save_networks(step=step)

    if len(test_images) > 0:
        if opts.save_new_images_every_epoch:
            save_image_paths = [os.path.join(opts.log_dir, 'step{:04d}-{}'.format(step, filename))
                                for filename in opts.test_image_names]
        else:
            save_image_paths = [os.path.join(opts.log_dir, filename) for filename in opts.test_image_names]
        logging.info('Saving {} test images as {}'.format(opts.n_test_images, ', '.join(save_image_paths)))
        model.save_images(test_images, save_image_paths)


def generate_image(model, opts):
    if opts.ckpt_path is None or opts.infer_image_path is None:
        raise IOError('For inference, you must provide a checkpoint and an inference image path')
    model.load_single_net('G_B', opts.ckpt_path)

    with open(os.path.join(opts.log_dir, 'original_image.jpeg'), 'wb') as f:
        pil_loader(opts.infer_image_path).save(f)
    image = model.convert_image(opts.infer_image_path)
    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir)
    with open(os.path.join(opts.log_dir, 'generated_image.jpeg'), 'wb') as f:
        image.save(f)


def main(opts):
    if os.environ.get('JOB_NAME', 'worker') != 'worker':
        import datetime
        while True:
            time.sleep(60)
            logging.info(str(datetime.datetime.now()))

    if opts.distributed:
        os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = opts.master.split(':')
        os.environ['WORLD_SIZE'] = str(opts.world_size)
        os.environ['RANK'] = str(opts.rank)

        if opts.gpu:
            logging.info('Initializing NCCL distributed backend')
            dist.init_process_group('nccl')
        else:
            logging.info('Initializing Gloo distributed backend')
            dist.init_process_group('gloo')

    logging.info('Initializing model')
    model = CycleGAN(opts)

    if opts.mode == 'train':
        train(model, opts)
    else:
        generate_image(model, opts)


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level=args.verbosity)

    logging.debug('='*30 + ' Environment Variables ' + '='*30)
    for k, v in sorted(os.environ.items()):
        logging.debug('{}: {}'.format(k, v))
    logging.debug('='*80)

    logging.debug('='*30 + ' Arguments ' + '='*30)
    for k, v in sorted(args.__dict__.items()):
        logging.debug('{}: {}'.format(k, v))
    logging.debug('='*80)

    main(args)
