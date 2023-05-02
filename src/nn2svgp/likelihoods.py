#!/usr/bin/env python3
from typing import Optional, Union

import torch
import torch.nn as nn
from src.nn2svgp.custom_types import FuncData, FuncMean, FuncVar, OutputData


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
        return f_mean, f_var + self.sigma_noise

    def log_prob(self, f: FuncData, y: OutputData):
        # TODO check this works
        return torch.distributions.Normal(f, self.sigma_noise).log_prob(y)

    def nn_loss(self, f: FuncData, y: OutputData):
        loss = torch.nn.MSELoss()(f, y)
        return 0.5 * loss * y.shape[-1]

    def residual(self, y, f):
        # TODO should this just be y?
        return y - f

    def Hessian(self, f):
        assert f.size(1) == 1
        return torch.ones_like(f).unsqueeze(-1) / (self.sigma_noise**2)


class Softmax(nn.Module):
    def nll(f: FuncData, y: OutputData):
        # return 0.5 * torch.nn.MSELoss(reduction="sum")(f, y)
        loss = torch.nn.MSELoss()(f, y)
        return 0.5 * loss * y.shape[-1]
