#!/usr/bin/env python3
from typing import Optional

import gpytorch
import pytorch_lightning as pl
import torch
import torch.distributions as td
from custom_types import (
    Action,
    Prediction,
    # ReplayBuffer_to_dynamics_TensorDataset,
    State,
)
from torchrl.data import ReplayBuffer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from .models import DynamicModel


class GPModule(pl.LightningModule):
    def __init__(
        self, gp: gpytorch.models.GP, likelihood: gpytorch.likelihoods.Likelihood
    ):
        super(GPModule, self).__init__()
        self.gp = gp
        self.likelihood = likelihood

    def forward_(self, x) -> Prediction:
        # self.gp.eval()
        # self.likelihood.eval()
        latent = self.gp(x)
        print("latent {}".format(latent))
        output = self.likelihood(latent)
        print("output {}".format(output))
        f_dist = td.Normal(loc=latent.mean, scale=torch.sqrt(latent.variance))
        print("f_dist {}".format(f_dist))
        y_dist = td.Normal(loc=latent.mean, scale=torch.sqrt(output.variance))
        print("y_dist {}".format(y_dist))
        noise_var = output.variance - latent.variance
        print("noise_var {}".format(noise_var))
        # pred = Prediction(latent=f_dist, output=y_dist, noise_var=noise_var)
        pred = Prediction(latent_dist=f_dist, output_dist=y_dist, noise_var=noise_var)
        print("pred {}".format(pred))
        return pred

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError


# class BaseGPDynamicsModel(DynamicModel):
#     # TODO implement delta_state properly
#     def __init__(
#         self,
#         gp_module: GPModule,
#         trainer: Optional[Trainer] = None,
#         delta_state: bool = True,
#         num_workers: int = 1,
#     ):
#         super(BaseGPDynamicsModel, self).__init__()
#         self.gp_module = gp_module
#         if trainer is None:
#             self.trainer = Trainer()
#         else:
#             self.trainer = trainer
#         self.delta_state = delta_state
#         self.num_workers = num_workers

#     def forward(self, x) -> Prediction:
#         self.gp_module.gp.eval()
#         self.gp_module.likelihood.eval()
#         return self.gp_module.forward(x)

#     def replay_buffer_to_data_loader(self, replay_buffer: ReplayBuffer) -> DataLoader:
#         dataset = ReplayBuffer_to_dynamics_TensorDataset(
#             replay_buffer, delta_state=self.delta_state
#         )
#         train_x = dataset.tensors[0]
#         train_y = dataset.tensors[1]
#         num_data = train_x.shape[0]
#         train_loader = DataLoader(
#             dataset,
#             batch_sampler=torch.utils.data.BatchSampler(
#                 torch.utils.data.SequentialSampler(dataset),
#                 batch_size=num_data,
#                 drop_last=True,
#             ),
#             num_workers=self.num_workers,
#         )
#         self.gp_module.gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
#         return train_loader

#     def fit(self, replay_buffer: ReplayBuffer):
#         train_loader = self.replay_buffer_to_data_loader(replay_buffer=replay_buffer)
#         print("before FIT")
#         self.gp_module.gp.train()
#         self.gp_module.likelihood.train()
#         self.trainer.fit(self.gp_module, train_dataloaders=train_loader)
#         print("after FIT")
#         self.gp_module.gp.eval()
#         self.gp_module.likelihood.eval()
