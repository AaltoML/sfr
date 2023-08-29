#!/usr/bin/env python3
import torch
import torch.nn as nn


class Prior(nn.Module):
    def __init__(self, params: nn.Parameter):
        super().__init__()
        self.params = params

    def log_prob(self):
        raise NotImplementedError

    def nn_loss(self):
        raise NotImplementedError


class Gaussian(Prior):
    def __init__(self, params: torch.nn.Parameter, prior_precision: float = 0.001):
        super().__init__(params=params)
        self.prior_precision = prior_precision

    def log_prob(self):
        raise NotImplementedError
        return torch.distributions.Normal(
            torch.ones_like(self.params),
            self.prior_precision * torch.ones_like(self.params),
        ).log_prob(self.params)

    def nn_loss(self):
        squared_params = torch.cat(
            [torch.square(param.view(-1)) for param in self.params()]
        )
        l2r = 0.5 * torch.sum(squared_params)
        return self.prior_precision * l2r
