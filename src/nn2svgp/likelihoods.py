#!/usr/bin/env python3
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from src.nn2svgp.custom_types import FuncData, FuncMean, FuncVar, OutputData
from torch.distributions import Bernoulli, Categorical


EPS = 1e-7

# class Likelihood:
#     # def __init__(self, network: nn.Module):
#     #     self.network = network

#     # def __call__(self, params: Params, x: InputData):
#     #     f = functional_call(self.network, params, (x.unsqueeze(0),))[0, ...]
#     #     return self.inv_link(f)

#     # def predict_given_params(self, params: Params, x: InputData):
#     #     return functional_call(self.network, params, (x.unsqueeze(0),))[0, ...]

#     def __cal__(self, f_mean: FuncMean, f_var: Optional[FuncVar] = None):
#         raise NotImplementedError

#     def nn_loss(self, params, y: OutputData):
#         raise NotImplementedError

#     def predict(self, x: InputData, params: Params):
#         raise NotImplementedError

#     def residual(self, x: InputData, params: Params, y: OutputData):
#         raise NotImplementedError

#     def Hessian(self, x: InputData, params: Params, y: OutputData):
#         raise NotImplementedError


# class GaussianWeightSpace(nn.Module):
#     def __init__(
#         self, network: nn.Module, sigma_noise: Union[float, torch.nn.Parameter] = 1.0
#     ):
#         super().__init__(self, network=network)
#         self.sigma_noise = sigma_noise

#     # def __call__(self, params: Params, x: InputData):
#     #     # def predict(self, f_mean: FuncMean, f_var: Optional[FuncVar] = None):
#     #     f = self.predict_f(params, x)
#     #     f_var = torch.zeros_like(f_mean)
#     #     return f_mean, f_var + self.sigma_noise

#     def log_prob(self, f: FuncData, y: OutputData):
#         # TODO check this works
#         return torch.distributions.Normal(f, self.sigma_noise).log_prob(y)

#     def nn_loss(self, f: FuncData, y: OutputData):
#         loss = torch.nn.MSELoss()(f, y)
#         return 0.5 * loss * y.shape[-1]

#     def residual(self, y, f):
#         # TODO should this just be y?
#         return y - f

#     def Hessian(self, f):
#         assert f.size(1) == 1
#         return torch.ones_like(f).unsqueeze(-1) / (self.sigma_noise**2)


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
        # TODO check this works
        return torch.distributions.Normal(f, self.sigma_noise).log_prob(y)

    def nn_loss(self, f: FuncData, y: OutputData):
        # loss = torch.nn.MSELoss()(f, y)
        # loss = torch.nn.MSELoss(reduction="sum")(f, y)
        loss = 0.5 * torch.nn.MSELoss(reduction="mean")(f, y)
        return loss
        # return 0.5 * loss * y.shape[-1]
        # return -torch.sum(self.log_prob(f=f, y=y))
        # return -torch.mean(self.log_prob(f=f, y=y))

    def residual(self, y, f):
        # TODO should this just be y?
        return y - f

    def Hessian(self, f):
        # assert f.size(1) == 1
        # return torch.ones_like(f).unsqueeze(-1) / (self.sigma_noise**2)
        H = torch.ones_like(f) / (self.sigma_noise**2)
        return torch.diag_embed(H)


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class BernoulliLh(Likelihood):
    def log_prob(self, f: FuncData, y: OutputData):
        dist = Bernoulli(logits=f)
        return dist.log_prob(y)
        # return torch.sum(dist.log_prob(y))

    def prob(self, f_mean: FuncMean, f_var: FuncVar):
        return inv_probit(f_mean / torch.sqrt(1 + f_var))
        # dist = Bernoulli(logits=f)
        # return torch.sum(dist.log_prob(y))

    def Hessian(self, f):
        p = self.inv_link(f)
        # p = torch.clamp(self.inv_link(f), EPS, 1 - EPS)
        H = p * (1 - p)
        return torch.diag_embed(H)

    def inv_link(self, f):
        return inv_probit(f)
        # return torch.sigmoid(f)

    def residual(self, y, f):
        return self.inv_link(f) - y

    def nn_loss(self, f: FuncData, y: OutputData):
        print("calling nn_loss")
        # print("nn_l9oss")
        print("f {}".format(f.shape))
        print("y {}".format(y.shape))
        # print("f {}".format(f))
        # print("y {}".format(y))
        # print("f {}".format(f))
        # torch.math.log(torch.where(torch.equal(x, 1), p, 1 - p))
        # print("log(f) {}".format(torch.log(f)))
        # log_prob = y * torch.log(f) + (1 - y) * torch.log(1 - f)
        # log_prob = (1 - y) * torch.log(f) + (y) * torch.log(1 - f)
        # return -torch.sum(log_prob)
        # return self.nn_loss_func()(f, y)
        # return -torch.sum(self.log_prob(f=f, y=y))
        return torch.nn.functional.binary_cross_entropy(
            self.inv_link(f), y, reduction="mean"
        )

    def nn_loss_func(self):
        return lambda logits, y: -torch.sum(self.log_prob(logits, y))
        # return lambda logits, y: -torch.mean(self.log_prob(logits, y))
        # return lambda logits, y: -torch.mean(self.log_prob(logits, y))

    # raise ValueError('No extendable nn loss for backpack in Bernoulli case')


class CategoricalLh(Likelihood):
    def log_prob(self, f: FuncData, y: OutputData):
        dist = Categorical(logits=f)
        return dist.log_prob(y)
        # return torch.sum(dist.log_prob(y))

    def residual(self, y, f):
        print("RESIDUAL")
        print("y {}".format(y.shape))
        print("f {}".format(f.shape))
        print("f {}".format(f))
        y_expand = torch.zeros_like(f)
        # y_expand = torch.ones_like(f)
        # print("y_expand {}".format(y_expand.shape))
        # print("len(y) {}".format(len(y)))
        ixs = torch.arange(0, len(y)).long()
        # print("y.long {}".format(y))
        # print("ixs {}".format(ixs.shape))
        # ixs = self.inv_link(f) < 0.5
        # print("ixs {}".format(ixs.shape))
        # y_expand[ixs] = 1
        y_expand[ixs, y.long()] = 1
        # y_expand[ixs, y.long()] = 0
        # y_expand = torch.ones_like(f)
        # y_expand[y.long()] = 0
        # y_expand[..., y[:, 0].long()] = 1
        # y_expand = torch.ones_like(f)
        # y_expand[:, y.long()] = 1
        print("y_expand {}".format(y_expand))
        print("self.inv_link(f) {}".format(self.inv_link(f).shape))
        return self.inv_link(f) - y_expand

    def Hessian(self, f):
        print("self.inv_link(f) {}".format(self.inv_link(f)))
        p = torch.clamp(self.inv_link(f), EPS, 1 - EPS)
        # p = self.inv_link(f)
        print("p {}".format(p.shape))
        H = torch.diag_embed(p) - torch.einsum("ij,ik->ijk", p, p)
        print("H {}".format(H.shape))
        return H

    def inv_link(self, f):
        return torch.nn.functional.softmax(f, dim=-1)
        # return torch.nn.Softmax(dim=-1)(f)

    def nn_loss(self, f: FuncData, y: OutputData):
        # if f.ndim == 1:
        #     f = f[None, ...]
        # if y.ndim == 1:
        #     y = y[None, ...]
        print("YOYOYO")
        print("f {}".format(f.shape))
        print("y {}".format(y.shape))
        # # return torch.nn.CrossEntropyLoss(reduction="sum")(f, y)
        # #    y_onehot =
        # log_probs = torch.sum(y * torch.log(f), axis=-1)
        # # print("log_probs {}".format(log_probs.shape))
        # return -torch.mean(log_probs)
        # return -torch.sum(log_probs)
        # return torch.nn.CrossEntropyLoss(reduction="sum")(f, y)
        # return -torch.sum(self.log_prob(f=f, y=y))
        # print("self.log_prob(f=f, y=y) {}".format(self.log_prob(f=f, y=y).shape))
        # return -torch.mean(self.log_prob(f=f, y=y))
        return torch.nn.CrossEntropyLoss(reduction="mean")(f, y)
        # return torch.nn.CrossEntropyLoss(reduction="mean")(f, y)
        # return torch.nn.functional.cross_entropy(f, y)
        # return torch.nn.CrossEntropyLoss(reduction="mean")

    # def nn_loss_func(self):
    #     return torch.nn.CrossEntropyLoss(reduction="mean"), 1
