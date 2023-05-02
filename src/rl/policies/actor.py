#!/usr/bin/env python3
from src.custom_types import State
import torch
import torch.nn as nn
from utils import helper as h
from utils import net


class Actor(nn.Module):
    def __init__(self,
        trunk_network:nn.Module,
        actor_network:nn.Module):
        super().__init__()
        self.trunk = trunk_network
        self._actor = actor_network
        self.apply(net.orthogonal_init)
    # def __init__(self, latent_dim, mlp_dims, action_shape):
    #     super().__init__()
    #     self.trunk = nn.Sequential(
    #         nn.Linear(latent_dim, mlp_dims[0]), nn.LayerNorm(mlp_dims[0]), nn.Tanh()
    #     )
    #     self._actor = net.mlp(mlp_dims[0], mlp_dims[1:], action_shape[0])
    #     self.apply(net.orthogonal_init)

    def forward(self, state:State, std:float):
        feature = self.trunk(obs)
        mu = self._actor(feature)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return h.TruncatedNormal(mu, std)

class Actor(nn.Module):
    def __init__(self, latent_dim, mlp_dims, action_shape):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, mlp_dims[0]), nn.LayerNorm(mlp_dims[0]), nn.Tanh()
        )
        self._actor = net.mlp(mlp_dims[0], mlp_dims[1:], action_shape[0])
        self.apply(net.orthogonal_init)

    def forward(self, obs, std):
        feature = self.trunk(obs)
        mu = self._actor(feature)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return h.TruncatedNormal(mu, std)
