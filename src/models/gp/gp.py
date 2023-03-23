#!/usr/bin/env python3
from typing import Optional

import gpytorch
import pytorch_lightning as pl
import torch
import torch.distributions as td
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from torchrl.data import ReplayBuffer
from custom_types import (
    Action,
    State,
    Prediction,
    # ReplayBuffer_to_dynamics_TensorDataset,
)
from models import TransitionModel


class GPTransitionModel(TransitionModel):
    # TODO implement delta_state properly
    def __init__(
        self,
        # gp_module: GPModule,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
        learning_rate: float = 1,
        trainer: Optional[Trainer] = None,
        delta_state: bool = True,
        num_workers: int = 1,
    ):
        super(GPTransitionModel, self).__init__()
        # self.gp_module = gp_module
        self.gp_module = ExactGPModule(
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            learning_rate=learning_rate,
        )
        if trainer is None:
            self.trainer = Trainer()
        else:
            self.trainer = trainer
        self.delta_state = delta_state
        self.num_workers = num_workers

    def forward(self, x) -> Prediction:
        self.gp_module.eval()
        latent = self.gp_module(x)
        print("latent")
        print(latent)
        output = self.gp_module.likelihood(latent)
        print("output")
        print(output)
        f_dist = td.Normal(loc=latent.mean, scale=torch.sqrt(latent.variance))
        print("f_dist")
        print(f_dist)
        y_dist = td.Normal(loc=latent.mean, scale=torch.sqrt(output.variance))
        print("y_dist")
        print(y_dist)
        noise_var = output.variance - latent.variance
        print("noise_var")
        print(noise_var)
        # pred = Prediction(latent=f_dist, output=y_dist, noise_var=noise_var)
        pred = Prediction(latent_dist=f_dist, output_dist=y_dist, noise_var=noise_var)
        print("pred")
        print(pred)
        return pred

    def train(self, replay_buffer: ReplayBuffer):
        dataset = ReplayBuffer_to_dynamics_TensorDataset(
            replay_buffer, delta_state=self.delta_state
        )
        train_x = dataset.tensors[0]
        train_y = dataset.tensors[1]
        num_data = train_x.shape[0]
        train_loader = DataLoader(
            dataset,
            batch_sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(dataset),
                batch_size=num_data,
                drop_last=True,
            ),
            num_workers=self.num_workers,
        )

        self.gp_module.set_train_data(inputs=train_x, targets=train_y, strict=False)
        print("before FIT")
        self.gp_module.gp.train()
        self.gp_module.likelihood.train()
        self.trainer.fit(self.gp_module, train_dataloaders=train_loader)
        print("after FIT")
        self.gp_module.gp.eval()
        self.gp_module.likelihood.eval()


class ExactGPModule(pl.LightningModule):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
        # learning_rate: float = 1,
    ):
        super(ExactGPModule, self).__init__()
        self.automatic_optimization = False
        self.learning_rate = learning_rate

        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
        if covar_module is None:
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        class ExactMultiOutputGPModel(gpytorch.models.ExactGP):
            def __init__(self):
                super(ExactMultiOutputGPModel, self).__init__(
                    train_inputs=None, train_targets=None, likelihood=likelihood
                )
                self.mean_module = mean_module
                self.covar_module = covar_module

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                # return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                return (
                    gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                        gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                    )
                )

        self.gp = ExactMultiOutputGPModel()
        self.likelihood = likelihood
        # self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        #     self.gp.likelihood, self.gp
        # )

    def set_train_data(self, inputs, targets, strict=False):
        self.gp.set_train_data(inputs=inputs, targets=targets, strict=strict)

    def training_step(self, batch, batch_idx):
        x, y = batch
        opt = self.optimizers()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

        # Display all model layer weights
        for name, param in self.named_parameters():
            print("{}: {}".format(name, param))

        def closure():
            output = self.gp(x)
            loss = -mll(output, y)
            opt.zero_grad()
            self.manual_backward(loss)
            self.log("model_loss", loss)
            return loss

        opt.step(closure=closure)
        output = self.gp(x)
        loss = -mll(output, y)
        return {"loss": loss}

    def configure_optimizers(self):
        # return torch.optim.Adam(self.gp.parameters(), lr=self.learning_rate)
        return torch.optim.LBFGS(
            # self.parameters(),
            # lr=self.learning_rate,
            # max_iter=self.max_iter,
            # max_eval=self.max_eval
            # tolerance_grad=1e-07,
            # tolerance_change=1e-09,
            # history_size=100,
            # line_search_fn=None,
            lr=0.1,
            # lr=1,
            max_iter=20,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=100,
            line_search_fn=None,
        )
