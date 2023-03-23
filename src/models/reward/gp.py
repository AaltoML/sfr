#!/usr/bin/env python3
import logging
from typing import Any, Callable, NamedTuple, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gpytorch
import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim
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
from models.base import RewardModel
from torchrl.data import ReplayBuffer
from torchrl.modules import SafeModule, WorldModelWrapper
from torchtyping import TensorType


class GPRewardModel(RewardModel):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
        num_inducing: int = 16,
        learning_rate: float = 0.1,
        num_iterations: int = 1000,
        num_workers: int = 1,
        learn_inducing_locations: bool = True,
    ):
        # TODO set in_size somewhere else
        in_size = 5

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_workers = num_workers

        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.gp = GPModel(
            in_size=in_size,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing=num_inducing,
            learn_inducing_locations=learn_inducing_locations,
        )
        self.likelihood = likelihood

        print("Reward GP params:")
        for param in self.gp.parameters():
            print(param)
        print("Reward lik params:")
        for param in self.likelihood.parameters():
            print(param)
        # self.gp = torch.compile(self.gp)
        # self.likelihood = torch.compile(self.likelihood)

    def __call__(
        self,
        state_mean: StateMean,
        state_var: StateVar,
        noise_var: NoiseVar,
        # self, state_mean: StateMean, state_var: StateVar, num_samples: int = 5
    ) -> TensorType["one"]:
        """Returns expected reward"""
        # print("inside reward_model.call()")
        num_samples = [3]
        state_dist = td.Normal(loc=state_mean, scale=torch.sqrt(state_var))
        # print("state_dist {}".format(state_dist))
        state_samples = state_dist.sample(num_samples)
        # print("state_samples {}".format(state_samples.shape))
        # reward_samples = torch.vmap(self.forward, randomness="same")(state_samples)
        reward_samples = []
        for state_sample in state_samples:
            reward_samples.append(self.forward(state_sample))
        reward_samples = torch.stack(reward_samples, 0)
        # reward_samples = self.forward(state_samples)
        # print("reward_samples {}".format(reward_samples.shape))
        expected_reward = torch.mean(reward_samples, 0)
        return expected_reward

    def forward(self, state: State):
        """Returns reward"""
        # def forward(self, x, data_new: Optional = None) -> Prediction:
        # TODO should I reshape x here as [N, D]?
        self.gp.eval()
        self.likelihood.eval()
        # TODO add data_new here
        # print("state yo yo: {}".format(state.shape))
        # latent = self.gp.forward(state)
        # print(
        #     "inducing_points: {}".format(
        #         self.gp.variational_strategy.inducing_points.shape
        #     )
        # )
        latent = self.gp(state)
        # print("latent {}".format(latent))
        output = self.likelihood(latent)
        # print("output {}".format(output))
        noise_var = output.variance - latent.variance
        # print("noise_var {}".format(noise_var))
        return latent.mean
        # return latent.mean, latent.variance, noise_var

    def train(self, replay_buffer: ReplayBuffer):
        num_data = len(replay_buffer)
        self.gp.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [
                {"params": self.gp.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.learning_rate,
        )

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp, num_data=num_data)
        logger.info("Data set size: {}".format(num_data))

        print("wa")
        for i in range(self.num_iterations):
            sample = replay_buffer.sample()
            x = sample["next"]["state_vector"]
            # print("x {}".format(x.shape))
            y = sample["reward"][..., 0]
            # print("y {}".format(y.shape))
            optimizer.zero_grad()
            latent = self.gp(x)
            # print("latent {}".format(latent))
            loss = -mll(latent, y)
            # print("loss {}".format(loss.shape))
            logger.info("Reward iteration {}, Loss: {}".format(i, loss))
            loss.backward()
            optimizer.step()
            wandb.log({"Reward loss": loss})

        self.gp.eval()
        self.likelihood.eval()


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        in_size: int,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
        num_inducing: int = 16,
        learn_inducing_locations: bool = True,
    ):
        # TODO initialise inducing points from data
        inducing_points = torch.rand(num_inducing, in_size)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super(GPModel, self).__init__(variational_strategy)

        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
        if covar_module is None:
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x, data_new: Optional = None):
        # print("insode GPModel forward")
        # print(x.shape)
        if data_new is None:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
        else:
            raise NotImplementedError("# TODO Paul implement fast update here")
        # print("mean_x: {}".format(mean_x.shape))
        # print("covar_x: {}".format(covar_x.shape))
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        # print("dist {}".format(dist))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
