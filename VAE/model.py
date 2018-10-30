import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc21 = nn.Linear(200, 20)
        self.fc22 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 28*28)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu, logvar = self.fc21(h), self.fc22(h)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def decode(self, h):
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        h = torch.sigmoid(h)
        return h

    def forward(self, x):
        x = x.view(-1, 1, 28*28)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

    @staticmethod
    def loss(x, x_rec, mu, logvar):
        bce_loss = F.binary_cross_entropy(x_rec, x.view(-1, 1, 28*28), reduction='sum')
        kl_loss = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
        # , dim=1)
        # kl_loss = torch.mean(kl_loss)
        loss = bce_loss + kl_loss

        return loss
