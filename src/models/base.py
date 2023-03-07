#!/usr/bin/env python3
from typing import Callable, NamedTuple, Optional

import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim
from custom_types import Action, Prediction, State
from torchrl.data import ReplayBuffer


class DynamicModel:
    def __call__(self, state: State, action: Action) -> Prediction:
        return self.predict(state=state, action=action)

    def predict(
        self, state: State, action: Action, data_new: Optional = None
    ) -> Prediction:
        x = torch.concat([state, action], -1)
        return self.forward(x)

    def forward(self, x, data_new: Optional = None) -> Prediction:
        raise NotImplementedError

    def train(self, replay_buffer: ReplayBuffer, delta_state: bool = True):
        raise NotImplementedError


# class DynamicsModel(pl.LightningModule):
#     def __init__(
#         self,
#         model,
#         trainer,
#         batch_size: int = 64,
#         num_workers: int = 1,
#         shuffle: bool = True,
#     ):
#         self.model = model
#         self.trainer = trainer
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.shuffle = shuffle

#     def predict(self, state: State, action: Action) -> Prediction:
#         x = torch.concat([state, action], -1)
#         return self.forward(x)

#     def forward(self) -> Prediction:
#         raise NotImplementedError

#     def train(self, replay_buffer: ReplayBuffer, delta_state: bool = True):
#         train_loader = ReplayBuffer_to_dynamics_DataLoader(
#             replay_buffer=replay_buffer,
#             batch_size=self.batch_size,
#             shuffle=self.shuffle,
#             delta_state=delta_state,
#             num_workers=self.num_workers,
#         )
#         self.trainer.fit(self.model, train_loader)

#     def training_step(self, batch, batch_idx):
#         raise NotImplementedError

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
