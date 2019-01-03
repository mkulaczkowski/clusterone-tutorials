import itertools
import os
import pickle
# import sys
from collections import OrderedDict

import torch
import torch.nn as nn

# sys.path.append(os.path.dirname(__file__))
from .networks import define_D, define_G, GANLoss
from .data import pil_loader, get_inference_transform, get_reverse_transform


class CycleGAN(nn.Module):
    def __init__(self, opts):
        super(CycleGAN, self).__init__()
        self.opts = opts
        self.is_train = opts.mode == 'train'
        self.log_dir = opts.log_dir
        self.ckpt_file = os.path.join(opts.log_dir, 'checkpoints')
        self.max_ckpts = opts.max_ckpts
        self.gpu = opts.gpu
        self.device = opts.device

        if self.is_train:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.netG_A = define_G(opts.netG, opts.ngf, opts.norm, not opts.no_dropout, 'normal', 0.02, opts.gpu_ids)
        self.netG_B = define_G(opts.netG, opts.ngf, opts.norm, not opts.no_dropout, 'normal', 0.02, opts.gpu_ids)

        if opts.distributed:
            self.netG_A = nn.parallel.distributed.DistributedDataParallel(self.netG_A)
            self.netG_B = nn.parallel.distributed.DistributedDataParallel(self.netG_B)

        self.netD_A, self.netD_B = None, None
        self.criterionGAN, self.criterionCycle, self.criterionIdentity = None, None, None
        self.optimizer_D, self.optimizer_G = None, None
        self.optimizers = []
        if self.is_train:
            self.netD_A = define_D(opts.netD, opts.ndf, opts.n_layers_D, opts.norm, False, 'normal', 0.02, opts.gpu_ids)
            self.netD_B = define_D(opts.netD, opts.ndf, opts.n_layers_D, opts.norm, False, 'normal', 0.02, opts.gpu_ids)

            if opts.distributed:
                self.netD_A = nn.parallel.distributed.DistributedDataParallel(self.netD_A)
                self.netD_B = nn.parallel.distributed.DistributedDataParallel(self.netD_B)

            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=True).to(opts.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdentity = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opts.learning_rate, betas=(opts.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opts.learning_rate, betas=(opts.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # Initialize bunch of variables that will be used later
        self.real_A, self.fake_A, self.recr_A, self.idt_A = None, None, None, None
        self.real_B, self.fake_B, self.recr_B, self.idt_B = None, None, None, None

        if opts.lambda_identity > 0.:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        else:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        self.loss_D_A, self.loss_G_A, self.loss_cycle_A, self.loss_idt_A = None, None, None, None
        self.loss_D_B, self.loss_G_B, self.loss_cycle_B, self.loss_idt_B = None, None, None, None
        self.loss = None

    def backward(self):
        """Entire Backpropagation step"""
        # Generators
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # Discriminators
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def backward_D(self):
        """Backpropagation for both of the discriminators"""
        self.loss_D_A = self.backward_D_single(self.netD_A, self.real_A, self.fake_A)
        self.loss_D_B = self.backward_D_single(self.netD_B, self.real_B, self.fake_B)

    def backward_D_single(self, netD, real, fake):
        """Backpropagation for a discriminator network"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) / 2.0
        loss_D.backward()
        return loss_D

    def backward_G(self):
        """Backpropagation for both of the generators"""
        lambda_identity = self.opts.lambda_identity
        lambda_A = self.opts.lambda_A
        lambda_B = self.opts.lambda_B

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_A), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True)

        self.loss_cycle_A = self.criterionCycle(self.recr_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.recr_B, self.real_B) * lambda_B

        if lambda_identity > 0.:
            self.idt_A = self.netG_A(self.real_A)
            self.loss_idt_A = self.criterionIdentity(self.idt_A, self.real_A) * lambda_A * lambda_identity
            self.idt_B = self.netG_B(self.real_B)
            self.loss_idt_B = self.criterionIdentity(self.idt_B, self.real_B) * lambda_B * lambda_identity

            self.loss = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                        self.loss_idt_A + self.loss_idt_B
        else:
            self.loss = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B

    def forward(self, batch_A, batch_B):
        """Run a pair of images through entire network"""
        self.real_A = batch_A
        self.real_B = batch_B

        self.fake_A = self.netG_A(self.real_B)
        self.recr_B = self.netG_B(self.fake_A)

        self.fake_B = self.netG_B(self.real_A)
        self.recr_A = self.netG_A(self.fake_B)

    @property
    def losses(self):
        """Return a ordered dictionary of (loss name, loss value) pairs"""
        losses = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses[name] = float(getattr(self, 'loss_' + name))
        return losses

    def optimize_step(self, batch_A, batch_B):
        """Combines forward and backward step"""
        self.forward(batch_A, batch_B)
        self.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Convenient method to quickly change requires_grad value for all the parameters of a network"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @property
    def saved_checkpoints(self):
        """Returns a list of (step, [path1,path2,path3,path4]) pairs read from checkpoints file"""
        if not os.path.exists(self.ckpt_file):
            return []
        with open(self.ckpt_file, 'r') as f:
            lines = [line.strip().split(':') for line in f.readlines()]
        return [(step, filenames.split(',')) for step, filenames in lines]

    def remove_checkpoints(self, filenames):
        """Deletes the list of filenames"""
        for filename in filenames:
            filepath = os.path.join(self.log_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)

    def update_checkpoint_file(self, checkpoint):
        """Insert a new checkpoint into list of checkpoints. Remove oldest one if there are too many."""
        step, filenames = checkpoint

        old_checkpoints = self.saved_checkpoints
        for i in range(len(old_checkpoints)-1, -1, -1):
            old_step, old_filenames = old_checkpoints[i]
            if old_step == step:
                old_checkpoints.pop(i)
                self.remove_checkpoints(old_filenames)

        if len(old_checkpoints) >= self.max_ckpts:
            oldest_checkpoint = old_checkpoints.pop()
            self.remove_checkpoints(oldest_checkpoint[1])

        checkpoints = [checkpoint] + old_checkpoints
        lines = ['{}:'.format(step) + ','.join(filenames) for step, filenames in checkpoints]
        with open(self.ckpt_file, 'w') as f:
            f.write('\n'.join(lines))
            f.write('\n')

    def save_networks(self, ckpt_dir=None, step='latest'):
        """Save the network weights to files"""
        if ckpt_dir is None:
            ckpt_dir = self.log_dir
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        filenames = []
        for name in self.model_names:
            if isinstance(name, str):
                filename = '{}_net_{}.pth'.format(step, name)
                filenames.append(filename)
                save_path = os.path.join(ckpt_dir, filename)
                net = getattr(self, 'net' + name)

                with open(save_path, 'wb') as f:
                    if self.opts.distributed:
                        torch.save(net.module.cpu().state_dict(), f)
                        # pickle.dump(net.module.cpu().state_dict(), f)
                    else:
                        torch.save(net.cpu().state_dict(), f)
                        # pickle.dump(net.cpu().state_dict(), f)
                net.to(self.device)

        self.update_checkpoint_file((step, filenames))

    def load_networks(self, ckpt_dir=None, step='latest'):
        """Load the network weights from files"""
        if ckpt_dir is None:
            ckpt_dir = self.log_dir
        for name in self.model_names:
            if isinstance(name, str):
                load_path = os.path.join(ckpt_dir, '{}_net_{}.pth'.format(step, name))
                self.load_single_net(name, load_path)

    def load_single_net(self, net_name, load_path):
        """Load the network weights for a single network from file"""
        net = getattr(self, 'net' + net_name)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('Loading the {} model from {}'.format(net_name, load_path))
        with open(load_path, 'rb') as f:
            # state_dict = pickle.load(f)
            state_dict = torch.load(f, map_location=str(self.opts.device))
            # state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_images(self, tensors, save_paths, save_generated_images=True):
        """Takes tensors, performs reverse transform, and saves the image (option to run through the generator first)"""
        if save_generated_images:
            tensors = self.netG_B(tensors)
        tensors = tensors.cpu()

        reverse_transform = get_reverse_transform()
        for tensor, save_path in zip(tensors, save_paths):
            image = reverse_transform(tensor)
            with open(save_path, 'wb') as f:
                image.save(f, 'jpeg')

    def convert_image(self, image_path, direction='AtoB'):
        """Load an iamge from file, convert it using the generator, and save the image"""
        image = pil_loader(image_path)
        tensor = get_inference_transform()(image)
        tensor = torch.unsqueeze(tensor, 0)
        if direction == 'AtoB':
            generated_tensor = self.netG_B(tensor)
        elif direction == 'BtoA':
            generated_tensor = self.netG_A(tensor)
        else:
            raise IOError("direction must be AtoB or BtoA (not found: {})".format(direction))
        generated_tensor = torch.squeeze(generated_tensor)
        return get_reverse_transform()(generated_tensor)
