import os
import argparse
from solver import Solver

from torch.backends import cudnn
from data_utils import MyDataset
import torch


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.

    data_loader = torch.utils.data.DataLoader(MyDataset('./train', crop_size=256, image_size=128),
                                                  config.batch_size,
                                                  shuffle=True, num_workers=2)

    # Solver for training and testing StarGAN.
    solver = Solver(data_loader, config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--a_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--id_dim', type=int, default=347, help='dimension of domain identity')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=None, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='face_id/logs')
    parser.add_argument('--model_save_dir', type=str, default='face_id/models')
    parser.add_argument('--sample_dir', type=str, default='face_id/samples')
    parser.add_argument('--result_dir', type=str, default='face_id/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--model_test_step', type=int, default=3000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)