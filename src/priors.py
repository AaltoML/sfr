#!/usr/bin/env python3
import torch
import torch.nn as nn

from typing import Union, Optional
from src.custom_types import FuncData, FuncMean, FuncVar, OutputData


class Prior(nn.Module):
    def __init__(self, params: nn.Parameter):
        super().__init__()
        self.params = params

    def log_prob(self):
        raise NotImplementedError

    def nn_loss(self):
        raise NotImplementedError


class Gaussian(Prior):
    def __init__(self, params: torch.nn.Parameter, delta: float = 0.001):
        super().__init__(params=params)
        self.delta = delta

    def log_prob(self):
        # TODO is this right? Should it be delta**2?
        return torch.distributions.Normal(
            torce.ones_like(self.params), self.delta**2 * torch.ones_like(self.params)
        ).log_prob(self.params)

    def nn_loss(self):
        # print("PARAMS: {}".format(self.params))
        # for param in self.params():
        # print("param:m {}".format(param))
        squared_params = torch.cat(
            [torch.square(param.view(-1)) for param in self.params()]
        )
        l2r = 0.5 * torch.sum(squared_params)
        return self.delta * l2r
