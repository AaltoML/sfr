#!/usr/bin/env python3
import logging
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from src.custom_types import Data
from src.likelihoods import Likelihood
from src.priors import Prior

from .sfr import SFR


class NN2GPSubset(SFR):
    def __init__(
        self,
        network: torch.nn.Module,
        prior: Prior,
        likelihood: Likelihood,
        output_dim: int,
        subset_size: int,
        dual_batch_size: Optional[int] = None,
        jitter: float = 1e-6,
        device: str = "cpu",
    ):
        super().__init__(
            network=network,
            prior=prior,
            likelihood=likelihood,
            output_dim=output_dim,
            dual_batch_size=dual_batch_size,
            num_inducing=subset_size,
            jitter=jitter,
            device=device,
        )
        self.subset_size = subset_size

    @torch.no_grad()
    def set_data(self, train_data: Data):
        """Sets training data, samples inducing points, calcs dual parameters, builds predict fn"""
        X_train, Y_train = train_data
        X_train = torch.clone(X_train)
        Y_train = torch.clone(Y_train)
        print("X_train {}".format(X_train.shape))
        print("Y_train {}".format(Y_train.shape))
        assert X_train.shape[0] == Y_train.shape[0]
        self._num_data = Y_train.shape[0]
        indices = torch.randperm(self._num_data)[: self.subset_size]
        X_subset = X_train[indices.to(X_train.device)].to(self.device)
        Y_subset = Y_train[indices.to(Y_train.device)].to(self.device)
        self.train_data = (X_subset, Y_subset)
        self.Z = X_subset
        self._build_sfr()

    @property
    def num_data(self):
        return self._num_data
