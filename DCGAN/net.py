# coding: UTF-8
from torch import nn


class Generator(nn.Module):

    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # -> (ngf*8)x4x4
            nn.ConvTranspose2d(
                ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # -> (ngf*4)x8x8
            nn.ConvTranspose2d(
                ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # -> (ngf*2)x16x16
            nn.ConvTranspose2d(
                ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # -> (ngf)x32x32
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # -> (nc)x64x64
        )

        # weight initialization

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):

    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf)x32x32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2)x16x16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4)x8x8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8)x4x4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
            # 1x1x1
        )

        # weight initialization

    def forward(self, x):
        return self.net(x)
