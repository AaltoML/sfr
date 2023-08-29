#!/usr/bin/env python3
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from src.custom_types import FuncData, FuncMean, FuncVar, OutputData
from torch.distributions import Bernoulli, Categorical, Normal

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Likelihood:
    def __call__(
        self, f_mean: Union[FuncData, FuncMean], f_var: Optional[FuncVar] = None
    ):
        raise NotImplementedError

    def log_prob(self, f: FuncData, y: OutputData):
        raise NotImplementedError

    def nn_loss(self, f: FuncData, y: OutputData):
        raise NotImplementedError

    def residual(self, f: FuncData, y: OutputData):
        raise NotImplementedError

    def Hessian(self, f: FuncData):
        raise NotImplementedError


class Gaussian(Likelihood):
    def __init__(self, sigma_noise: Union[float, torch.nn.Parameter] = 1.0):
        self.sigma_noise = sigma_noise

    def __call__(
        self, f_mean: Union[FuncData, FuncMean], f_var: Optional[FuncVar] = None
    ):
        if f_var is None:
            f_var = torch.zeros_like(f_mean)
        return f_mean, f_var + self.sigma_noise**2

    def log_prob(self, f: FuncData, y: OutputData):
        log_prob = torch.distributions.Normal(loc=f, scale=self.sigma_noise).log_prob(y)
        if log_prob.ndim > 1:
            # sum over independent output dimensions
            log_prob = torch.sum(log_prob, -1)
        return log_prob

    def nn_loss(self, f: FuncData, y: OutputData):
        loss = 0.5 * torch.nn.MSELoss(reduction="mean")(f, y)
        return loss

    def residual(self, y, f):
        return (y - f) / self.sigma_noise**2

    def Hessian(self, f):
        H = torch.ones_like(f) / (self.sigma_noise**2)
        return torch.diag_embed(H)


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class BernoulliLh(Likelihood):
    def __init__(self, EPS: float = 0.0001):
        self.EPS = EPS

    def __call__(
        self, f_mean: Union[FuncData, FuncMean], f_var: Optional[FuncVar] = None
    ):
        if f_var is None:
            p = self.inv_link(f_mean)
        else:
            p = self.prob(f_mean=f_mean, f_var=f_var)
        mean = p
        var = p - torch.square(p)
        return mean, var

    def log_prob(self, f: FuncData, y: OutputData):
        dist = Bernoulli(logits=f)
        return dist.log_prob(y)

    def prob(self, f_mean: FuncMean, f_var: FuncVar):
        return inv_probit(f_mean / torch.sqrt(1 + f_var))

    def Hessian(self, f):
        p = torch.clamp(self.inv_link(f), self.EPS, 1 - self.EPS)
        H = p * (1 - p)
        return torch.diag_embed(H)

    def inv_link(self, f):
        return inv_probit(f)

    def residual(self, y, f):
        if f.ndim > 1:
            f = f[:, 0]
        res = y - self.inv_link(f)
        res = res[..., None]
        return res

    def nn_loss(self, f: FuncData, y: OutputData):
        if f.shape > y.shape:
            f = f[..., 0]
        return torch.nn.functional.binary_cross_entropy(
            self.inv_link(f), y, reduction="mean"
        )


class CategoricalLh(Likelihood):
    def __init__(self, EPS: float = 0.01, num_classes: int = None):
        self.EPS = EPS
        self.num_classes = num_classes

    def __call__(
        self,
        f_mean: Union[FuncData, FuncMean],
        f_var: Optional[FuncVar] = None,
        num_samples: int = 100,
    ):
        if f_var is None:
            p = self.prob(f=f_mean)
        else:
            if (f_var < 1e-5).sum().item() > 0 and (f_var >= -1e-5).sum().item():
                logger.info(f"f_var==0: {(f_var == 0).sum().item()}")
                logger.info(
                    f"f_var: num_el {f_var.numel()} - equal zero el {(f_var == 0).sum().item()}"
                )
            if (f_var < 0).sum().item() > 0:
                logger.info(
                    f"f_var: num_el {f_var.numel()} - less than zero el {(f_var < 0).sum().item()}"
                )

            dist = Normal(f_mean, torch.sqrt(f_var.clamp(10 ** (-32))))
            logit_samples = dist.sample((num_samples,))
            samples = self.inv_link(logit_samples)
            p = torch.mean(samples, 0)

        mean = p
        var = p - torch.square(p)
        return mean, var

    def prob(self, f):
        if self.num_classes is None:
            num_classes = f.shape[-1]
            return torch.nn.Softmax(dim=num_classes)(f)
        else:
            return torch.nn.Softmax(dim=self.num_classes)(f)

    def log_prob(self, f: FuncData, y: OutputData):
        dist = Categorical(logits=f)
        return dist.log_prob(y)

    def residual(self, y, f):
        y_expand = torch.zeros_like(f)
        ixs = torch.arange(0, len(y)).long()
        y_expand[ixs, y.long()] = 1
        return y_expand - self.inv_link(f)

    def Hessian(self, f):
        p = torch.clamp(self.inv_link(f), self.EPS, 1 - self.EPS)
        H = torch.diag_embed(p) - torch.einsum("ij,ik->ijk", p, p)
        return H

    def inv_link(self, f):
        return torch.nn.functional.softmax(f, dim=-1)

    def nn_loss(self, f: FuncData, y: OutputData):
        return torch.nn.CrossEntropyLoss(reduction="mean")(f, y)
