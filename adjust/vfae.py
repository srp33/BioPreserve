import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, s_dim, latent_dim, hidden_dim=512):
        super(Encoder, self).__init__()
        # Example: Concatenate x and s (one-hot encoded if categorical)
        # Effective input dimension becomes input_dim + s_dim_effective
        self.fc1 = nn.Linear(input_dim + s_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, s_one_hot):
        # Concatenate x and s_one_hot
        combined_input = torch.cat([x, s_one_hot], dim=1)
        h = F.relu(self.fc1(combined_input))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, s_dim, output_dim, hidden_dim=512):
        super(Decoder, self).__init__()
        # Example: Concatenate z and s (one-hot encoded if categorical)
        self.fc1 = nn.Linear(latent_dim + s_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, s_one_hot):
        # Concatenate z and s_one_hot
        combined_input = torch.cat([z, s_one_hot], dim=1)
        h = F.relu(self.fc1(combined_input))
        x_recon = self.fc_out(h)
        return x_recon

class VFAE(nn.Module):
    def __init__(self, input_dim, num_s_categories, latent_dim, hidden_dim_enc, hidden_dim_dec):
        super(VFAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_s_categories = num_s_categories

        self.encoder = Encoder(input_dim, num_s_categories, latent_dim, hidden_dim_enc)
        self.decoder = Decoder(latent_dim, num_s_categories, input_dim, hidden_dim_dec) # output_dim = input_dim for reconstruction

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, s_indices):
        # Assume s is categorical and needs one-hot encoding
        s_one_hot = F.one_hot(s_indices, num_classes=self.num_s_categories).float()

        mu, logvar = self.encoder(x, s_one_hot)
        z = self.sample(mu, logvar)
        x_recon = self.decoder(z, s_one_hot)
        return x_recon, mu, logvar, z


def rbf_kernel(X, Y, gamma=1.0):
    XX = torch.matmul(X, X.t())
    XY = torch.matmul(X, Y.t())
    YY = torch.matmul(Y, Y.t())

    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    K_XY = torch.exp(-gamma * (
        -2 * XY + X_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0)
    ))
    return K_XY

def mmd_rbf_loss(X, Y, gamma=1.0):
    if X.shape == 0 or Y.shape == 0:
        return torch.tensor(0.0, device=X.device)

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    mmd_sq = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean() # biased estimator
    return mmd_sq



