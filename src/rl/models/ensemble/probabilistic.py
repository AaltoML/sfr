#!/usr/bin/env python3
from typing import List, Union

import torch
import torch.distributions as td
from pytorch_lightning import Trainer
from src.custom_types import Prediction
from torchtyping import TensorType

from ..networks import GaussianMLP
from .deterministic import Ensemble


class ProbabilisticEnsemble(Ensemble):
    def __init__(
        self,
        networks: List[GaussianMLP],
        trainers: List[Trainer],
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 1,
    ):
        super(ProbabilisticEnsemble, self).__init__(
            networks=networks,
            trainers=trainers,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def forward(self, x) -> Prediction:
        dists, means, vars = [], [], []
        # TODO make this run in parallel
        for network in self.networks:
            dist = network(x, dist=True)
            dists.append(dist)
            means.append(dist.mean())
            vars.append(dist.variance())
        means = torch.stack(means, -1)  # [num_data, output_dim, ensemble_size]
        vars = torch.stack(vars, -1)  # [num_data, output_dim, ensemble_size]

        f_vars = torch.var(means, -1)  # variance over ensembles
        f_dist = td.Normal(loc=means, scale=torch.sqrt(f_vars))

        ensemble_dists = td.Normal(loc=means, scale=torch.sqrt(vars))

        # Ensemble output is a uniform mixture of Gaussians
        ensemble_dist = td.MixtureSameFamily(
            mixture_distribution=td.Categorical(
                torch.ones(self.ensemble_size) / self.ensemble_size
            ),
            component_distribution=ensemble_dists,
        )

        noise_var = torch.mean(vars, -1)
        noise_var_var = torch.var(vars, -1)
        noise_var_dist = td.Normal(loc=noise_var, scale=torch.sqrt(noise_var_var))

        return Prediction(
            latent_dist=f_dist, output_dist=ensemble_dist, noise_var=noise_var_dist
        )

    def _single_forward(
        self, x, ensemble_idx: int, dist: bool = False
    ) -> Union[TensorType["N", "out_size*2"], td.Normal]:
        return self.networks[ensemble_idx](x, dist=dist)
