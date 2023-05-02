#!/usr/bin/env python3
from typing import List

import torch
import torch.distributions as td
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from src.custom_types import (
    Action,
    Observation,
    Prediction,
    ReplayBuffer,
    ReplayBuffer_to_dynamics_DataLoader,
    ReplayBuffer_to_dynamics_TensorDataset,
)
from torch.utils.data import DataLoader
from torchtyping import TensorType

from ..models import DynamicModel
from ..models import DynamicModel


class Ensemble(DynamicModel):
    def __init__(
        self,
        networks: List[LightningModule],
        trainers: List[Trainer],
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 1,
        delta_state: bool = True,
    ):
        # TODO pass delta_state to DynamicsModel?
        super(Ensemble, self).__init__()
        # self.networks = nn.ModuleList(networks)
        self.networks = networks
        self.trainers = trainers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.delta_state = delta_state

    def predict(self, observation: Observation, action: Action) -> Prediction:
        x = torch.concat([observation, action], -1)
        return self.forward(x)

    def train(self, replay_buffer: ReplayBuffer):
        print("BEFORE FIT")
        self.batch_size = 16
        for network, trainer in zip(self.networks, self.trainers):
            dataset = ReplayBuffer_to_dynamics_TensorDataset(
                replay_buffer, delta_state=self.delta_state
            )
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                # drop_last=True,
                num_workers=self.num_workers,
            )
            # train_loader = ReplayBuffer_to_dynamics_DataLoader(
            #     replay_buffer=replay_buffer,
            #     batch_size=self.batch_size,
            #     shuffle=self.shuffle,
            #     delta_state=delta_state,
            #     num_workers=self.num_workers,
            # )
            trainer.fit(network, train_loader)
            print("AFTER FIT")
        print("AFTER FIT FOR ALL")

    def forward(self, x) -> Prediction:
        """Ensemble output is a Gaussian"""
        ys = []
        # TODO make this run in parallel
        for network in self.networks:
            ys.append(network(x=x))
        ys = torch.stack(ys, -1)  # [num_data, output_dim, ensemble_size]

        f_mean = torch.mean(ys, -1)  # variance over ensembles
        f_var = torch.var(ys, -1)  # variance over ensembles
        f_dist = td.Normal(loc=f_mean, scale=torch.sqrt(f_var))
        print("f_dist")
        print(f_dist)
        print("f_mean.shape")
        print(f_mean.shape)
        print(f_var.shape)

        # TODO should output_dist be ys or f_dist???
        return Prediction(latent_dist=f_dist, output_dist=f_dist)

    def _single_forward(self, x, ensemble_idx: int) -> TensorType["N", "out_size"]:
        dist = self.networks[ensemble_idx](x)
        return dist

    @property
    def ensemble_size(self):
        return len(self.networks)
