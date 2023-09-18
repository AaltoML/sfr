#!/usr/bin/env python3
import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from netcal.metrics import ECE
from src.custom_types import FuncData, FuncMean, FuncVar, OutputData
from torch.distributions import Bernoulli, Categorical, Normal


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
        if not isinstance(sigma_noise, torch.Tensor):
            sigma_noise = torch.tensor(sigma_noise)

        # self.log_sigma_noise = sigma_noise
        self.log_sigma_noise = sigma_noise

    @property
    def sigma_noise(self):
        return torch.exp(self.log_sigma_noise)

    def __call__(
        self, f_mean: Union[FuncData, FuncMean], f_var: Optional[FuncVar] = None
    ):
        if f_var is None:
            f_var = torch.zeros_like(f_mean)
        return f_mean, f_var + self.sigma_noise**2

    def log_prob(self, f: FuncData, y: OutputData, f_var=None):
        y_var = torch.ones_like(f) * self.sigma_noise**2
        # print(f"y_var {y_var.shape}")
        # print(f"y {y.shape}")
        # print(f"f_var {f_var.shape}")
        # print(f"f {f.shape}")
        if f_var is not None:
            y_var += f_var
        #     print(f"y_var+f_var {y_var.shape}")
        # print("yo")
        dist = torch.distributions.Normal(
            loc=f, scale=torch.sqrt(y_var.clamp(10 ** (-32)))
        )
        # print(f"dist {dist}")
        log_prob = dist.log_prob(y)
        # print(f"log_prob {log_prob.shape}")
        if log_prob.ndim > 1:
            # sum over independent output dimensions
            log_prob = torch.sum(log_prob, -1)
            # print(f"log_prob {log_prob.shape}")
        # print(f"log_prob {log_prob.shape}")
        return log_prob

    def nn_loss(self, f: FuncData, y: OutputData):
        # print(f"f {f.shape}")
        # print(f"y {y.shape}")
        # loss = 0.5 * torch.nn.MSELoss(reduction="mean")(f, y)
        # return loss
        log_prob = self.log_prob(f=f, y=y)
        # print(f"log_prob {log_prob.shape}")
        return -log_prob.mean()

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

    def log_prob(self, f: FuncData, y: OutputData, f_var=None, num_samples: int = 100):
        if f_var:
            dist = Normal(f, torch.sqrt(f_var.clamp(10 ** (-32))))
            logit_samples = dist.sample((num_samples,))
            samples = self.inv_link(logit_samples)
            prob_samples = torch.mean(samples, 0)
            print(f"prob_samples {prob_samples.shape}")
            print(f"y {y.shape}")
            log_prob_samples = Bernoulli(probs=prob_samples).log_prob(y)
            print(f"log_prob_samples {log_prob_samples}")
            log_prob = torch.sum(log_prob_samples, 0)
            print(f"log_prob {log_prob}")
        else:
            log_prob = Bernoulli(logits=f).log_prob(y)
        return log_prob

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

    def log_prob(self, f: FuncData, y: OutputData, f_var=None, num_samples: int = 100):
        if f_var:
            dist = Normal(f_mean, torch.sqrt(f_var.clamp(10 ** (-32))))
            logit_samples = dist.sample((num_samples,))
            samples = self.inv_link(logit_samples)
            log_p = torch.mean(samples, 0)
        else:
            dist = Categorical(logits=f)
            log_p = dist.log_prob(y)
        return log_p

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
