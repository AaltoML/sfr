#!/usr/bin/env python3
import logging
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gpytorch
import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import wandb
from custom_types import (
    Action,
    DeltaStateMean,
    DeltaStateVar,
    NoiseVar,
    Prediction,
    State,
    StateMean,
    StateVar,
)
from models import TransitionModel
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from torchrl.data import ReplayBuffer
from utils import make_data_loader


class SVGPTransitionModel(TransitionModel):
    # TODO implement delta_state properly
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
        num_inducing: int = 16,
        learning_rate: float = 0.1,
        # batch_size: int = 16,
        # trainer: Optional[Trainer] = None,
        num_iterations: int = 1000,
        delta_state: bool = True,
        num_workers: int = 1,
        learn_inducing_locations: bool = True,
    ):
        # TODO is this the bset way to set out_size?
        in_size = 6
        out_size = covar_module.batch_shape[0]

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.delta_state = delta_state
        self.num_workers = num_workers

        # Learn seperate hyperparameters for each output dim
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=out_size
            )
        # # TODO remove this override
        # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        #     num_tasks=out_size
        # )

        self.gp = IndependentMultitaskGPModel(
            in_size=in_size,
            out_size=out_size,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing=num_inducing,
            learn_inducing_locations=learn_inducing_locations,
        )
        self.likelihood = likelihood

        print("GP params:")
        for param in self.gp.parameters():
            print(param)
        print("Lik params:")
        for param in self.likelihood.parameters():
            print(param)
        # self.gp = torch.compile(self.gp)
        # self.likelihood = torch.compile(self.likelihood)

    def forward(self, x, data_new: Optional = None) -> Prediction:
        # TODO should I reshape x here as [N, D]?
        self.gp.eval()
        self.likelihood.eval()
        # TODO add data_new here
        latent = self.gp(x)
        output = self.likelihood(latent)
        noise_var = output.variance - latent.variance
        return latent.mean, latent.variance, noise_var

    def train(self, replay_buffer: ReplayBuffer):
        print("here")
        num_data = len(replay_buffer)
        self.gp.train()
        self.likelihood.train()
        print("here 2")

        optimizer = torch.optim.Adam(
            [
                {"params": self.gp.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.learning_rate,
        )

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp, num_data=num_data)
        logger.info("Data set size: {}".format(num_data))

        for i in range(self.num_iterations):
            # print("Iteration {}".format(i))
            sample = replay_buffer.sample()
            obs = sample["state_vector"]
            x = torch.concat([obs, sample["action"]], -1)  # obs_action_input
            # print("obs_action_input: {}".format(x.shape))
            y = sample["next"]["state_vector"] - obs  # obs_diff_output
            # y = torch.concat([y, sample["reward"]], -1)
            # print("obs_diff_output: {}".format(y.shape))
            optimizer.zero_grad()

            # y_pred = self.forward(x)
            # latent = self.gp(x, data_new=data_new)
            latent = self.gp(x)
            loss = -mll(latent, y)
            logger.info("Iteration {}, Loss: {}".format(i, loss))
            loss.backward()
            optimizer.step()
            wandb.log({"model loss": loss})

        self.gp.eval()
        self.likelihood.eval()


class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
        num_inducing: int = 16,
        learn_inducing_locations: bool = True,
    ):
        # TODO initialise inducing points from data
        inducing_points = torch.rand(out_size, num_inducing, in_size)
        # inducing_points = torch.rand(out_size, num_inducing, 1)
        # inducing_points = torch.rand(num_inducing, 1)
        print("inducing_points")
        print(inducing_points.shape)

        # Learn a variational distribution for each output dim
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([out_size]),
        )
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=learn_inducing_locations,
                ),
                num_tasks=out_size,
            )
        )

        super().__init__(variational_strategy)
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean(
                batch_shape=torch.Size([out_size])
            )
        if covar_module is None:
            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([out_size])),
                batch_shape=torch.Size([out_size]),
            )
        # # TODO delete this mean/covar override
        # mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([out_size]))
        # covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size([out_size])),
        #     batch_shape=torch.Size([out_size]),
        # )
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x, data_new: Optional = None):
        if data_new is None:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
        else:
            raise NotImplementedError("# TODO Paul implement fast update here")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
