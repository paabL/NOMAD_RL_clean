from __future__ import annotations

import torch
import torch.nn as nn


def mlp(dims, activation=nn.SiLU):
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), activation()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class TrajectoryEncoder(nn.Module):
    def __init__(self, horizon, channels, latent_dim=8, hidden=(128, 128)):
        super().__init__()
        self.horizon = int(horizon)
        self.channels = int(channels)
        self.net = mlp([self.horizon * self.channels, *hidden, 2 * int(latent_dim)])

    def forward(self, tau):
        h = self.net(tau.reshape(tau.shape[0], -1))
        return h.chunk(2, dim=-1)


class BoundedContextDecoder(nn.Module):
    def __init__(self, latent_dim, low, high, hidden=(128, 128)):
        super().__init__()
        low = torch.as_tensor(low, dtype=torch.float32).reshape(1, -1)
        high = torch.as_tensor(high, dtype=torch.float32).reshape(1, -1)
        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.net = mlp([int(latent_dim), *hidden, low.shape[1]])

    def forward(self, z):
        return self.low + (self.high - self.low) * torch.sigmoid(self.net(z))


def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_standard_normal(mu, logvar):
    return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
