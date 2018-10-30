import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from model import VAE


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


class Experiment():

    def __init__(self, args):
        self.args = args

        # data
        self.train_loader = DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True
        )
        self.test_loader = DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False
        )
        self.model = VAE().to(args.device)
        self.loss = VAE.loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, epoch):
        self.model.train()
        train_loss = 0.
        n_samples = 0
        iter_i = 1
        for data, _ in self.train_loader:
            n_samples += len(data)
            data = data.to(self.args.device)
            self.optimizer.zero_grad()
            x_rec, mu, logvar = self.model(data)
            loss = self.loss(data, x_rec, mu, logvar)
            loss.backward()
            train_loss += loss.item()  # * len(data)
            self.optimizer.step()

            if iter_i % self.args.log_freq == 0:
                print('Epoch {} Train [{}/{}]:  LossAvg {:.4f}'.format(
                    epoch, n_samples, len(self.train_loader.dataset), loss.item() / len(data)))

            iter_i += 1
        print('Epoch {} Train: LossAvg {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0.
        iter_i = 1
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.args.device)
                x_rec, mu, logvar = self.model(data)
                loss = self.loss(data, x_rec, mu, logvar)
                test_loss += loss.item()  # * len(data)

                if iter_i == 1:
                    n = min(data.size(0), 8)
                    comparison = torch.cat(
                        [data[:n], x_rec.view(self.args.batch_size, 1, 28, 28)[:n]])

                iter_i += 1

        print('Epoch {} Test: LossAvg {:.4f}'.format(
            epoch, test_loss / len(self.test_loader.dataset)))
        self.save(epoch)
        return comparison.cpu()

    def save(self, epoch):
        torch.save(self.model.state_dict(), './results/checkpoint_{}.pt'.format(epoch))

    def run(self):
        for epoch_i in range(1, 1 + self.args.epochs):
            self.train(epoch_i)
            comparison = self.test(epoch_i)
            n = 8
            save_image(comparison, './results/comparison_{}.png'.format(epoch_i), nrow=n)
            visualize = make_grid(comparison, nrow=n)
            show(visualize.cpu())
