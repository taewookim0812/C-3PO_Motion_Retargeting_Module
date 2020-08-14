"""
Description: Variational AutoEncoder model for skeleton and robot reference motion learning
Author: Tae-woo Kim
Contact: twkim0812@gmail.com
"""

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE NAO Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


# VAE for Skeleton data
class VAE_SK(nn.Module):
    def __init__(self, input_dim, output_dim, use_batch_norm=False, activation='ReLU'):
        super(VAE_SK, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        # self.fc1 = nn.Linear(input_dim, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc41 = nn.Linear(64, 5)
        # self.fc42 = nn.Linear(64, 5)
        # self.fc5 = nn.Linear(5, 64)
        # self.fc6 = nn.Linear(64, 128)
        # self.fc7 = nn.Linear(128, 256)
        # self.fc8 = nn.Linear(256, output_dim)

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc51 = nn.Linear(64, 7)
        self.fc52 = nn.Linear(64, 7)
        self.fc6 = nn.Linear(7, 64)
        self.fc7 = nn.Linear(64, 128)
        self.fc8 = nn.Linear(128, 256)
        self.fc9 = nn.Linear(256, 512)
        self.fc10 = nn.Linear(512, output_dim)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def encode(self, x):
        if self.activation == 'ReLU':
            if self.use_batch_norm:
                h1 = self.bn1(F.relu(self.fc1(x)))
                h2 = self.bn2(F.relu(self.fc2(h1)))
                h3 = self.bn3(F.relu(self.fc3(h2)))
                h4 = F.relu(self.fc4(h3))
            else:
                h1 = F.relu(self.fc1(x))
                h2 = F.relu(self.fc2(h1))
                h3 = F.relu(self.fc3(h2))
                h4 = F.relu(self.fc4(h3))
        elif self.activation == 'Tanh':
            if self.use_batch_norm:
                h1 = self.bn1(torch.tanh(self.fc1(x)))
                h2 = self.bn2(torch.tanh(self.fc2(h1)))
                h3 = self.bn3(torch.tanh(self.fc3(h2)))
                h4 = torch.tanh(self.fc4(h3))
            else:
                h1 = torch.tanh(self.fc1(x))
                h2 = torch.tanh(self.fc2(h1))
                h3 = torch.tanh(self.fc3(h2))
                h4 = torch.tanh(self.fc4(h3))
        else:
            raise ValueError

        return self.fc51(h4), self.fc52(h4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        if self.activation == 'ReLU':
            if self.use_batch_norm:
                h6 = self.bn4(F.relu(self.fc6(z)))
                h7 = self.bn3(F.relu(self.fc7(h6)))
                h8 = self.bn2(F.relu(self.fc8(h7)))
                h9 = F.relu(self.fc9(h8))
            else:
                h6 = F.relu(self.fc6(z))
                h7 = F.relu(self.fc7(h6))
                h8 = F.relu(self.fc8(h7))
                h9 = F.relu(self.fc9(h8))
        elif self.activation == 'Tanh':
            if self.use_batch_norm:
                h6 = self.bn4(torch.tanh(self.fc6(z)))
                h7 = self.bn3(torch.tanh(self.fc7(h6)))
                h8 = self.bn2(torch.tanh(self.fc8(h7)))
                h9 = torch.tanh(self.fc9(h8))
            else:
                h6 = torch.tanh(self.fc6(z))
                h7 = torch.tanh(self.fc7(h6))
                h8 = torch.tanh(self.fc8(h7))
                h9 = torch.tanh(self.fc9(h8))
        else:
            raise ValueError

        return self.fc10(h9)  # torch.sigmoid()

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))   # [:, :-1] to exclude the phase
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=7, use_batch_norm=False, activation='Tanh'):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        # self.fc1 = nn.Linear(input_dim, 400)
        # self.fc2 = nn.Linear(400, 200)
        # self.fc3 = nn.Linear(200, 100)
        # self.fc41 = nn.Linear(100, 7)
        # self.fc42 = nn.Linear(100, 7)
        # self.fc5 = nn.Linear(7, 100)
        # self.fc6 = nn.Linear(100, 200)
        # self.fc7 = nn.Linear(200, 400)
        # self.fc8 = nn.Linear(400, input_dim)

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc41 = nn.Linear(64, latent_dim)  # mean
        self.fc42 = nn.Linear(64, latent_dim)  # variance
        self.fc5 = nn.Linear(latent_dim, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, 256)
        self.fc8 = nn.Linear(256, input_dim)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def encode(self, x):
        if self.activation == 'ReLU':
            if self.use_batch_norm:
                h1 = self.bn1(F.relu(self.fc1(x)))
                h2 = self.bn2(F.relu(self.fc2(h1)))
                h3 = F.relu(self.fc3(h2))
            else:
                h1 = F.relu(self.fc1(x))
                h2 = F.relu(self.fc2(h1))
                h3 = F.relu(self.fc3(h2))
        elif self.activation == 'Tanh':
            if self.use_batch_norm:
                h1 = self.bn1(torch.tanh(self.fc1(x)))
                h2 = self.bn2(torch.tanh(self.fc2(h1)))
                h3 = torch.tanh(self.fc3(h2))
            else:
                h1 = torch.tanh(self.fc1(x))
                h2 = torch.tanh(self.fc2(h1))
                h3 = torch.tanh(self.fc3(h2))
        else:
            raise ValueError

        return self.fc41(h3), self.fc42(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        if self.activation == 'ReLU':
            if self.use_batch_norm:
                h4 = self.bn3(F.relu(self.fc5(z)))
                h5 = self.bn2(F.relu(self.fc6(h4)))
                h6 = F.relu(self.fc7(h5))
            else:
                h4 = F.relu(self.fc5(z))
                h5 = F.relu(self.fc6(h4))
                h6 = F.relu(self.fc7(h5))
        elif self.activation == 'Tanh':
            if self.use_batch_norm:
                h4 = self.bn3(torch.tanh(self.fc5(z)))
                h5 = self.bn2(torch.tanh(self.fc6(h4)))
                h6 = torch.tanh(self.fc7(h5))
            else:
                h4 = torch.tanh(self.fc5(z))
                h5 = torch.tanh(self.fc6(h4))
                h6 = torch.tanh(self.fc7(h5))
        else:
            raise ValueError

        return self.fc8(h6)  # torch.sigmoid()

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
loss = nn.MSELoss()
def loss_function(recon_x, x, mu, logvar, input_dim):
    output = loss(recon_x, x.view(-1, input_dim))
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 200), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # return BCE + KLD
    return output
