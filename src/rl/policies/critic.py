#!/usr/bin/env python3
import torch
import torch.nn as nn
from utils import net


class Critic(nn.Module):
    def __init__(self, latent_dim, mlp_dims, action_shape):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + action_shape[0], mlp_dims[0]),
            nn.LayerNorm(mlp_dims[0]),
            nn.Tanh(),
        )
        self._critic1 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self._critic2 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self.apply(net.orthogonal_init)

    def forward(self, z, a):
        feature = torch.cat([z, a], dim=-1)
        feature = self.trunk(feature)
        return self._critic1(feature), self._critic2(feature)
