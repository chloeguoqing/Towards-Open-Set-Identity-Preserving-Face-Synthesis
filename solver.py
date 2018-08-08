from model import Generator
from model import Discriminator
from model import Encoder
from model import Attribute
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torchvision.models.vgg import vgg16
import torchvision.models as models


class Solver(object):
    """Solver for training and testing"""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.a_dim = config.a_dim
        self.id_dim = config.id_dim

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.lr = config.lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator()
        self.D = Discriminator()
        self.I = Encoder()
        self.C = Encoder()
        self.A = Attribute()
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.lr, [self.beta1, self.beta2])
        self.i_optimizer = torch.optim.Adam(self.I.parameters(), self.lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.lr, [self.beta1, self.beta2])
        self.a_optimizer = torch.optim.Adam(self.A.parameters(), self.lr, [self.beta1, self.beta2])


        self.G.to(self.device)
        self.D.to(self.device)
        self.A.to(self.device)
        self.I.to(self.device)
        self.C.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        A_path = os.path.join(self.model_save_dir, '{}-A.ckpt'.format(resume_iters))
        I_path = os.path.join(self.model_save_dir, '{}-I.ckpt'.format(resume_iters))
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(resume_iters))
        self.A.load_state_dict(torch.load(A_path, map_location=lambda storage, loc: storage))
        self.I.load_state_dict(torch.load(I_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.i_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.a_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.i_optimizer.zero_grad()
        self.a_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""

        return F.cross_entropy(logit, target)

    def mse_loss(self, out, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = 0.5 * torch.mean(torch.abs(out - gt)**2)
        return loss

    def L1_loss(self, pred, target):
        """
        Calculate L1 loss
        """
        return torch.mean(torch.abs(pred - target))

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = torch.FloatTensor(np.random.normal(0, 1, (mu.size(0), 8))).to(self.device)
        z = sampled_z * std + mu
        return z

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.

        data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        batch_fixed = next(data_iter)

        for k in batch_fixed:
            batch_fixed[k] = batch_fixed[k].to(self.device)

        # Learning rate cache for decaying.
        lr = self.lr


        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                batch = next(data_iter)
            except:
                data_iter = iter(data_loader)
                batch = next(data_iter)
            for k in batch:
                batch[k] = batch[k].to(self.device)
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            loss = {}
            # get identity z
            id_z, _ = self.I(batch['img_profile'])
            # get attribute z
            mu, logvar = self.A(batch['img_frontal'])
            a_z = self.reparameterization(mu, logvar)
            # get x'
            x = torch.cat([id_z, a_z], 1)

            x_fake = self.G(x)

            # Get the predicted identity
            id_pred, _ = self.C(batch['img_profile'])
            # distinguish the true and the false
            d_real, _ = self.D(batch['img_frontal'])
            d_fake, _ = self.D(x_fake.detach())
            # train I
            loss_Li = self.classification_loss(id_z, batch['label'])
            # train A
            loss_KL = torch.sum(0.5 * (mu**2 + torch.exp(logvar) - logvar - 1))
            loss_GR = self.mse_loss(batch['img_frontal'], x_fake)
            # triain C
            loss_C = self.classification_loss(id_pred, batch['label'])
            # train D
            loss_D = - torch.mean(d_real) + torch.mean(d_fake)
            d_loss = loss_D + loss_C + loss_GR + loss_KL + loss_Li

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            self.c_optimizer.step()
            self.a_optimizer.step()
            self.i_optimizer.step()

            loss['C/loss_C'] = loss_C.item()
            loss['A/loss_GR'] = loss_GR.item()
            loss['I/loss_Li'] = loss_Li.item()
            loss['D/loss_D'] = loss_D.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                id_z, _ = self.I(batch['img_profile'])
                # get attribute z
                mu, logvar = self.A(batch['img_frontal'])
                a_z = self.reparameterization(mu, logvar)
                # get x'
                x = torch.cat([id_z, a_z], 1)

                x_fake = self.G(x)
                # Get the predicted identity
                _, c_f_s = self.C(batch['img_profile'])
                _, c_f_x = self.C(x_fake)
                # distinguish the true and the false
                d_real, d_f_a = self.D(batch['img_frontal'])
                d_fake, d_f_x = self.D(x_fake)

                loss_GR = self.mse_loss(batch['img_frontal'], x_fake)
                # triain C

                loss_GC = self.mse_loss(c_f_x, c_f_s)
                loss_GD = self.mse_loss(d_f_x, d_f_a)
                loss_g = - torch.mean(d_fake)
                g_loss = loss_g + loss_GC + loss_GR + loss_GD

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_GR'] = loss_GR.item()
                loss['G/loss_GC'] = loss_GC.item()
                loss['G/loss_GD'] = loss_GD.item()
                loss['G/loss_g'] = loss_g.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                for k in batch_fixed:
                    batch_fixed[k] = batch_fixed[k].to(self.device)
                with torch.no_grad():
                    x_fake_list = [batch_fixed['img_profile']]

                    id_z, _ = self.I(batch_fixed['img_profile'])
                    # get attribute z
                    mu, logvar = self.A(batch_fixed['img_frontal'])
                    a_z = self.reparameterization(mu, logvar)
                    # get x'
                    x = torch.cat([id_z, a_z], 1)

                    x_fake = self.G(x)

                    x_fake_list.append(x_fake)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=2, padding=5)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                lr -= (self.lr / float(self.num_iters_decay))
                self.update_lr(lr)
                print('Decayed learning rates, lr: {}'.format(lr))

