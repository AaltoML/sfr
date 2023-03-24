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
        batch_size: int = 16,
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
        self.batch_size = batch_size
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

    # def forward(self, x, data_new: Optional = None) -> Prediction:
    #     # TODO should I reshape x here as [N, D]?
    #     self.gp.eval()
    #     self.likelihood.eval()
    #     # TODO add data_new here
    #     latent = self.gp(x)
    #     output = self.likelihood(latent)
    #     noise_var = output.variance - latent.variance
    #     return latent.mean, latent.variance, noise_var

    def forward(self, x, data_new: Optional = None) -> Prediction:
        self.gp.eval()
        self.likelihood.eval()

        if data_new != None:
            X, Y = data_new

            # # make copy of self
            model = self.make_copy()
            var_strat = self.model.variational_strategy.base_variational_strategy
            inducing_points = var_strat.inducing_points

            var_dist = var_strat.variational_distribution
            var_mean = var_dist.mean
            var_cov = var_dist.lazy_covariance_matrix

            # GPyTorch's way of computing Kuf:
            # full_inputs = torch.cat([inducing_points, X], dim=-2)
            full_inputs = torch.cat([torch.squeeze(inducing_points).T, X], dim=-2)
            full_covar = self.model.covar_module(full_inputs)

            # Covariance terms
            num_induc = inducing_points.size(-2)
            induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
            induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()

            K_uf = induc_data_covar

            # Kuu = self.covar_module(inducing_points)
            Kuu = induc_induc_covar
            # Kuu_root = Kuu.cholesky()

            lambda_1, lambda_2 = mean_cov_to_natural_param(var_mean, var_cov, Kuu)

            lambda_1_t = torch.zeros_like(lambda_1)
            lambda_2_t = torch.zeros_like(lambda_2)
            #
            # online_update
            for _ in range(1):  # TODO: make parameter
                # grad_varexp_natural_params
                with torch.no_grad():
                    # Xt = torch.tile(X, Y.shape[:-2] + (1, 1, 1))
                    #                 if Y.shape[-1] == 1:
                    #                     Xt.unsqueeze_(-1)
                    pred = model.forward(X)
                    mean = pred.latent_dist.mean
                    var = pred.latent_dist.variance
                mean.requires_grad_()
                var.requires_grad_()

                # variational expectations
                f_dist_b = MultivariateNormal(
                    mean.T, torch.diag_embed(var.T)
                )  # Mean: B x N Cov: B x N x N
                f_dist = (
                    gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                        f_dist_b
                    )
                )
                # f_dist = MultivariateNormal(mean, DiagLazyTensor(var))
                # f_dist = gpytorch.distributions.MultitaskMultivariateNormal(mean, var)'
                ve_terms = self.likelihood.expected_log_prob(
                    Y, f_dist
                )  # TODO: Is this right?
                ve = (
                    ve_terms.sum()
                )  # TODO: CHECK: divide by num_data ? but only one point at a time so probably fine

                ve.backward(inputs=[mean, var])
                d_exp_dm = mean.grad  # [batch, N]
                d_exp_dv = var.grad  # [batch, N]

                eps = 1e-8
                d_exp_dv.clamp_(max=-eps)

                grad_nat_1 = d_exp_dm - 2.0 * (d_exp_dv * mean)
                grad_nat_2 = d_exp_dv

                grad_mu_1 = torch.einsum("bmc, cb -> bm", K_uf, grad_nat_1)

                grad_mu_2 = torch.einsum("bmc, cb, bnc -> bmn", K_uf, grad_nat_2, K_uf)

                lr = 0.8  # TODO: set as a parameter
                scale = 1.0

                lambda_1_t_new = (1.0 - lr) * lambda_1_t + lr * scale * grad_mu_1[
                    ..., None
                ]
                lambda_2_t_new = (1.0 - lr) * lambda_2_t + lr * scale * (-2) * grad_mu_2

                lambda_1_new = lambda_1 - lambda_1_t + lambda_1_t_new
                lambda_2_new = lambda_2 - lambda_2_t + lambda_2_t_new

                new_mean, new_cov = conditional_from_precision_sites_white_full(
                    Kuu,
                    lambda_1_new,
                    lambda_2_new,
                    jitter=getattr(self, "tsvgp_jitter", 0.0),
                )
                new_mean = new_mean.squeeze(-1)
                new_cov_root = new_cov.cholesky()

                # fantasy_var_dist = fantasy_model.variational_strategy._variational_distribution
                with torch.no_grad():
                    var_dist = (
                        self.model.variational_strategy.base_variational_strategy.variational_distribution
                    )
                    var_dist.mean.set_(new_mean)
                    var_dist.covariance_matrix.set_(new_cov)

                lambda_1 = lambda_1_new
                lambda_2 = lambda_2_new
                lambda_1_t = lambda_1_t_new
                lambda_2_t = lambda_2_t_new

        # self.gp_module.eval()
        # f = self.gp_module.forward(x)
        f = self.gp(x)
        # print("latent {}".format(f.variance))

        output = self.likelihood(f)
        # print("output {}".format(output))
        f_dist = td.Normal(loc=f.mean, scale=torch.sqrt(f.variance))
        # print("f_dist {}".format(f_dist))
        y_dist = td.Normal(loc=f.mean, scale=torch.sqrt(output.variance))
        # print("y_dist {}".format(y_dist))
        noise_var = output.variance - f.variance

        # pred = Prediction(latent_dist=f_dist, output_dist=y_dist, noise_var=noise_var)
        # return pred
        return f.mean, f.variance, noise_var

    @property
    def model(self):
        return self.gp

    def make_copy(self):
        with torch.no_grad():
            inducing_points = (
                self.gp.variational_strategy.base_variational_strategy.inducing_points.detach().clone()
            )

            if hasattr(self, "input_transform"):
                [p.detach_() for p in self.input_transform.buffers()]

            # new_covar_module = deepcopy(self.gp_module.covar_module)

            new_model = self.__class__(
                likelihood=deepcopy(self.likelihood),
                mean_module=deepcopy(self.gp.mean_module),
                covar_module=deepcopy(self.gp.covar_module),
                learning_rate=deepcopy(self.learning_rate),
                # TODO add other arguments
            )
            #             new_model.mean_module = deepcopy(self.mean_module)
            #             new_model.likelihood = deepcopy(self.likelihood)

            var_dist = (
                self.gp.variational_strategy.base_variational_strategy.variational_distribution
            )
            mean = var_dist.mean.detach().clone()
            cov = var_dist.covariance_matrix.detach().clone()

            new_var_dist = (
                new_model.model.variational_strategy.base_variational_strategy.variational_distribution
            )
            with torch.no_grad():
                new_var_dist.mean.set_(mean)
                new_var_dist.covariance_matrix.set_(cov)
                new_model.model.variational_strategy.base_variational_strategy.inducing_points.set_(
                    inducing_points
                )

            new_model.model.variational_strategy.base_variational_strategy.variational_params_initialized.fill_(
                1
            )

        return new_model

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
            # sample = replay_buffer.sample()
            sample = replay_buffer.sample(batch_size=self.batch_size)
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
            logger.info("Transition model iteration {}, Loss: {}".format(i, loss))
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


def mean_cov_to_natural_param(mu, Su, K_uu):
    """
    Transforms (m,S) to (λ₁,P) tsvgp_white parameterization
    """
    mu = torch.unsqueeze(mu, dim=2)
    lamb1 = K_uu.matmul(Su.inv_matmul(mu))
    lamb2 = K_uu.matmul(Su.inv_matmul(K_uu.evaluate())) - K_uu.evaluate()

    return lamb1, lamb2


def conditional_from_precision_sites_white_full(
    Kuu,
    lambda1,
    Lambda2,
    jitter=1e-9,
):
    """
    Given a g₁ and g2, and distribution p and q such that
      p(g₂) = N(g₂; 0, Kuu)
      p(g₁) = N(g₁; 0, Kff)
      p(g₁ | g₂) = N(g₁; Kfu (Kuu⁻¹) g₂, Kff - Kfu (Kuu⁻¹) Kuf)
    And  q(g₂) = N(g₂; μ, Σ) such that
        Σ⁻¹  = Kuu⁻¹  + Kuu⁻¹LLᵀKuu⁻¹
        Σ⁻¹μ = Kuu⁻¹l
    This method computes the mean and (co)variance of
      q(g₁) = ∫ q(g₂) p(g₁ | g₂) dg₂ = N(g₂; μ*, Σ**)
    with
    Σ** = k** - kfu Kuu⁻¹ kuf - kfu Kuu⁻¹ Σ Kuu⁻¹ kuf
        = k** - kfu Kuu⁻¹kuf + kfu (Kuu + LLᵀ)⁻¹ kuf
    μ* = k*u Kuu⁻¹ m
       = k*u Kuu⁻¹ Λ⁻¹ Kuu⁻¹ l
       = k*u (Kuu + LLᵀ)⁻¹ l
    Inputs:
    :param Kuu: tensor M x M
    :param l: tensor M x 1
    :param L: tensor M x M
    """
    # TODO: rewrite this

    R = (Lambda2 + Kuu).add_jitter(jitter)

    mean = Kuu.matmul(R.inv_matmul(lambda1))
    cov = Kuu.matmul(R.inv_matmul(Kuu.evaluate()))  # TODO: symmetrise?
    return mean, cov
