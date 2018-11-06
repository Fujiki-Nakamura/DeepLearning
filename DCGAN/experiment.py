# coding: UTF-8
import os.path as osp

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from net import Discriminator, Generator


class Experiment():

    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter(args.output_dir)
        self.iter_i = 1

        # data
        transform_list = [
            transforms.Resize(args.imsize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        train_dataset = datasets.CIFAR10(
            './data', train=True, transform=transforms.Compose(transform_list),
            download=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True
        )

        # network
        self.G = Generator(args.nz, args.ngf, args.nc).to(args.device)
        self.D = Discriminator(args.nc, args.ndf).to(args.device)
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(
            self.G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optimizer_D = optim.Adam(
            self.D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        self.real_label = 1
        self.fake_label = 0
        self.fixed_z = torch.randn((args.batch_size, args.nz, 1, 1), device=args.device)

    def train(self, epoch):
        self.G.train()
        self.D.train()

        train_loss_G, train_loss_D = 0., 0.
        n_samples = 0
        for data, _ in self.train_loader:
            batch_size = len(data)
            n_samples += batch_size

            real = data.to(self.args.device)

            # train D
            self.optimizer_D.zero_grad()
            # with real
            label = torch.full((batch_size,), self.real_label, device=self.args.device)
            output_real = self.D(real)
            loss_D_real = self.criterion(output_real, label)
            loss_D_real.backward()

            # with fake
            z = torch.randn((batch_size, self.args.nz, 1, 1), device=self.args.device)
            fake = self.G(z)
            label = label.fill_(self.fake_label)
            output_fake = self.D(fake)
            loss_D_fake = self.criterion(output_fake, label)
            loss_D_fake.backward(retain_graph=True)

            loss_D = loss_D_real + loss_D_fake
            self.optimizer_D.step()

            # train G
            self.optimizer_G.zero_grad()
            label = label.fill_(self.real_label)
            output_fake = self.D(fake)
            loss_G = self.criterion(output_fake, label)
            loss_G.backward()
            self.optimizer_G.step()

            loss_D = loss_D.item()
            loss_G = loss_G.item()
            train_loss_D += loss_D
            train_loss_G += loss_G

            if self.iter_i % self.args.log_freq == 0:
                self.writer.add_scalar('Loss/D', loss_D, self.iter_i)
                self.writer.add_scalar('Loss/G', loss_G, self.iter_i)

                print('Epoch {} Train [{}/{}]:  Loss/D {:.4f} Loss/G {:.4f}'.format(
                    epoch, n_samples, len(self.train_loader.dataset),
                    loss_D / batch_size, loss_G / batch_size))

            self.iter_i += 1

        dataset_size = len(self.train_loader.dataset)
        print('Epoch {} Train: Loss/D {:.4f} Loss/G {:.4f}'.format(
            epoch, train_loss_D / dataset_size, train_loss_G / dataset_size))

    def test(self, epoch):
        self.G.eval()

        with torch.no_grad():
            fake = self.G(self.fixed_z)
            grid = make_grid(fake, normalize=True).cpu()
            self.writer.add_image('Fake', grid, self.iter_i)
            # show(grid)
            fname = osp.join(self.args.output_dir, 'fake_epoch_{}.png'.format(epoch))
            save_image(fake, fname, nrow=8)

    def save(self, epoch):
        # TODO
        torch.save(self.model.state_dict(), './results/checkpoint_{}.pt'.format(epoch))

    def run(self):
        for epoch_i in range(1, 1 + self.args.epochs):
            self.train(epoch_i)
            self.test(epoch_i)
