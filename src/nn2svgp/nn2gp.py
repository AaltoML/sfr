#!/usr/bin/env python3
import logging
from typing import Callable, List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import torch.nn as nn
from src.nn2svgp.custom_types import (  # Lambda_1,; Lambda_2,
    Alpha,
    AlphaInducing,
    Beta,
    BetaInducing,
    Data,
    FuncData,
    FuncMean,
    FuncVar,
    InducingPoints,
    InputData,
    Lambda,
    NTK,
    NTK_single,
    OutputData,
    OutputMean,
    OutputVar,
    TestInput,
)
from src.nn2svgp.likelihoods import Likelihood
from src.nn2svgp.priors import Prior
from torch.func import functional_call, hessian, jacrev, jvp, vjp, vmap
from torch.utils.data import DataLoader, TensorDataset
from torchtyping import TensorType

from .ntksvgp import build_ntk, calc_lambdas, NTKSVGP


class NN2GPSubset(NTKSVGP):
    def __init__(
        self,
        network: torch.nn.Module,
        prior: Prior,
        likelihood: Likelihood,
        output_dim: int,
        subset_size: int,
        dual_batch_size: Optional[int]=None,
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
        # self.train_data = (X_train, Y_train)
        self._num_data = Y_train.shape[0]
        indices = torch.randperm(self._num_data)[: self.subset_size]
        X_subset = X_train[indices.to(X_train.device)].to(self.device)
        Y_subset = Y_train[indices.to(Y_train.device)].to(self.device)
        self.train_data = (X_subset, Y_subset)
        self.Z = X_subset
        self.build_dual_svgp()

    @property
    def num_data(self):
        return self._num_data
