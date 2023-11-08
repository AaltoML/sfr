#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Iterator, Callable
from torch.nn import Parameter

ParamFn = Callable[[bool], Iterator[Parameter]]


class Prior(nn.Module):
    def __init__(self, params: ParamFn):
        super().__init__()
        self.params = params

    def log_prob(self):
        raise NotImplementedError

    def nn_loss(self):
        raise NotImplementedError


class Gaussian(Prior):
    def __init__(self, params: ParamFn, prior_precision: float = 0.001):
        super().__init__(params=params)
        self.prior_precision = prior_precision

    def log_prob(self):
        raise NotImplementedError

    def nn_loss(self):
        squared_params = torch.cat(
            [torch.square(param.view(-1)) for param in self.params()]
        )
        l2r = 0.5 * torch.sum(squared_params)
        return self.prior_precision * l2r
