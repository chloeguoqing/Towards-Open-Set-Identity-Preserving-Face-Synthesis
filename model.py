import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from net_sphere import sphere20a


class ResBlock(nn.Module):
    """
    This residual block is different with the one we usually know which consists of
    [conv - norm - act - conv - norm] and identity mapping(x -> x) for shortcut.
    Also spatial size is decreased by half because of AvgPool2d.
    """

    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.Upsample(scale_factor=2))

        self.short_cut = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out

class ResidualBlock(nn.Module):
    """
    This residual block is different with the one we usually know which consists of
    [conv - norm - act - conv - norm] and identity mapping(x -> x) for shortcut.
    Also spatial size is decreased by half because of AvgPool2d.
    """

    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

        self.short_cut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out


class Encoder(nn.Module):
    """
    Output is mu and log(var) for re-parameterization trick used in Variation Auto Encoder.
    Encoding is done in this order.
    1. Use this encoder and get mu and log_var
    2. std = exp(log(var / 2))
    3. random_z = N(0, 1)
    4. encoded_z = random_z * std + mu (Re-parameterization trick)
    """

    def __init__(self, c_dim=347):
        super(Encoder, self).__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.res_blocks = nn.Sequential(ResidualBlock(64, 128),
                                        ResidualBlock(128, 192),
                                        ResidualBlock(192, 256))
        self.pool_block = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.AvgPool2d(kernel_size=4, stride=4, padding=0))

        # Return mu and log var for re-parameterization trick
        self.fc = nn.Linear(1024, c_dim)

    def forward(self, x):
        # (N, 3, 128, 128) -> (N, 64, 64, 64)
        out = self.conv(x)
        # (N, 64, 64, 64) -> (N, 128, 32, 32) -> (N, 192, 16, 16) -> (N, 256, 8, 8)
        out = self.res_blocks(out)
        # (N, 256, 8, 8) -> (N, 256, 2, 2)
        out = self.pool_block(out)
        # (N, 256, 1, 1) -> (N, 256)
        out = out.view(x.size(0), -1)

        # (N, 256) -> (N, z_dim) x 2
        mu = self.fc(out)

        return mu, out


class Attribute(nn.Module):
    def __init__(self, latent_dim=8, feature=False):
        super(Attribute, self).__init__()
        """
            Output is mu and log(var) for re-parameterization trick used in Variation Auto Encoder.
            Encoding is done in this order.
            1. Use this encoder and get mu and log_var
            2. std = exp(log(var / 2))
            3. random_z = N(0, 1)
            4. encoded_z = random_z * std + mu (Re-parameterization trick)
        """
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Generator(nn.Module):
    def __init__(self, id_dim=347, a_dim=8):
        super(Generator, self).__init__()

        self.fc = nn.Linear(id_dim+a_dim, 512)
        self.up = nn.Upsample(scale_factor=8)

        self.block1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.block2 = ResBlock(256, 256)
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.block4 = ResBlock(256, 256)
        self.block5 = ResBlock(256, 256)
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.block7 = ResBlock(128, 128)
        self.block8 = nn.Sequential(nn.Conv2d(128, 3, kernel_size=7, stride=1, padding=3, bias=False),
                      nn.Tanh())

    def forward(self, x):

        x = self.fc(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.up(x)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)

        return block8


class Discriminator(nn.Module):

    def __init__(self, featmap_dim=512, n_channel=3):
        super(Discriminator, self).__init__()
        self.featmap_dim = featmap_dim
        self.conv1 = nn.Conv2d(n_channel, featmap_dim // 4, 5,
                               stride=2, padding=2)

        self.conv2 = nn.Conv2d(featmap_dim // 4, featmap_dim // 2, 5,
                               stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim // 2)

        self.conv3 = nn.Conv2d(featmap_dim // 2, featmap_dim, 5,
                               stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(featmap_dim)

        self.fc = nn.Linear(featmap_dim * 4 * 4, 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        out = F.sigmoid(self.fc(x))
        return out, x
