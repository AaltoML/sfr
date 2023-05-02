#!/usr/bin/env python3
import logging
from typing import Callable, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import torch.nn as nn
from src.nn2svgp.custom_types import (
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
    Lambda_1,
    Lambda_2,
    NTK,
    OutputData,
    OutputMean,
    OutputVar,
    TestInput,
)
from src.nn2svgp.likelihoods import Likelihood
from src.nn2svgp.priors import Prior
from torch.func import functional_call, hessian, jacrev, jvp, vjp, vmap
from torchtyping import TensorType


class NTKSVGP(nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        train_data: Data,
        prior: Prior,
        likelihood: Likelihood,
        num_inducing: int = 30,
        jitter: float = 1e-6,
    ):
        super().__init__()
        self.network = network
        self.prior = prior
        self.likelihood = likelihood
        self.train_data = train_data
        X_train, Y_train = train_data
        assert X_train.ndim == 2
        assert Y_train.ndim == 2
        assert X_train.shape[0] == Y_train.shape[0]
        num_data, output_dim = Y_train.shape
        print("Y_train.shape {}".format(Y_train.shape))
        indices = torch.randperm(num_data)[:num_inducing]
        Z = X_train[indices]
        # num_inducing = 100
        # Z = torch.linspace(0, 3, num_inducing).reshape(-1, 1)
        # Z = torch.rand(num_inducing, 1) * 3
        assert Z.ndim == 2
        self.num_inducing, self.input_dim = Z.shape
        self.Z = Z
        self.jitter = jitter

        self.build_dual_svgp()

    def build_dual_svgp(self):
        if isinstance(self.prior, src.nn2svgp.priors.Gaussian):
            delta = self.prior.delta
        else:
            raise NotImplementedError(
                "what should delta be if not using Gaussian prior???"
            )
        self.kernel = build_ntk(
            network=self.network,
            num_data=self.num_data,
            output_dim=self.output_dim,
            delta=delta,
        )

        self.alpha, self.beta = calc_sparse_dual_params(
            network=self.network,
            train_data=self.train_data,
            Z=self.Z,
            kernel=self.kernel,
            nll=self.likelihood.nn_loss,
        )
        print("alpha {}".format(self.alpha.shape))
        print("beta {}".format(self.beta.shape))
        assert self.alpha.ndim == 2
        assert self.beta.ndim == 3
        assert self.alpha.shape[0] == self.output_dim
        assert self.alpha.shape[1] == self.num_inducing
        assert self.beta.shape[0] == self.output_dim
        assert self.beta.shape[1] == self.num_inducing
        assert self.beta.shape[2] == self.num_inducing

        self._predict_fn = predict_from_duals(
            alpha=self.alpha,
            beta=self.beta,
            kernel=self.kernel,
            Z=self.Z,
            jitter=self.jitter,
        )

    def forward(self, x: InputData):
        return self.predict(x=x)

    @torch.no_grad()
    def predict(self, x: TestInput) -> Tuple[FuncMean, FuncVar]:
        f_mean, f_var = self._predict_fn(x, full_cov=False)
        return self.likelihood(f_mean=f_mean, f_var=f_var)

    @torch.no_grad()
    def predict_f(self, x: TestInput) -> Tuple[FuncMean, FuncVar]:
        # TODO implement full_cov=True
        # TODO implement noise_var correctly
        f_mean, f_var = self._predict_fn(x, full_cov=False)
        return f_mean, f_var

    def loss(self, x: InputData, y: OutputData):
        f = self.network(x)
        neg_log_likelihood = self.likelihood.nn_loss(f=f, y=y)
        neg_log_prior = self.prior.nn_loss()
        return neg_log_likelihood + neg_log_prior

    def update(self, x: InputData, y: OutputData):
        # TODO what about classificatin
        assert x.ndim == 2 and y.ndim == 2
        num_new_data, input_dim = x.shape
        Kui = self.kernel(self.Z, x)
        print("Kui {}".format(Kui.shape))
        print("alpha {}".format(self.alpha.shape))
        print("beta {}".format(self.beta.shape))
        print("x {}".format(x.shape))
        print("y {}".format(y.shape))

        # lambda_1, lambda_2 = calc_lambdas(Y=Y, F=F, nll=nll)

        self.alpha += (Kui @ y.T[..., None])[..., 0]
        self.beta += (
            Kui
            @ (1**-1 * torch.eye(num_new_data)[None, ...])
            @ torch.transpose(Kui, -1, -2)
        )
        print("ALPHA {}".format(self.alpha.shape))
        print("BETA {}".format(self.beta.shape))

        self._predict_fn = predict_from_duals(
            alpha=self.alpha,
            beta=self.beta,
            kernel=self.kernel,
            Z=self.Z,
            jitter=self.jitter,
        )

    @property
    def num_data(self):
        return self.train_data[0].shape[0]

    @property
    def output_dim(self):
        return self.train_data[1].shape[1]


def build_ntk(
    network: nn.Module, num_data: int, output_dim: int, delta: float = 1.0
) -> NTK:
    # Detaching the parameters because we won't be calling Tensor.backward().
    params = {k: v.detach() for k, v in network.named_parameters()}

    def fnet_single(params, x, i):
        return functional_call(network, params, (x.unsqueeze(0),))[0, ...][:, i]

    def single_output_ntk(x1: InputData, x2: InputData, i):
        # func_x1 = partial(fnet_single, x=x1, i=i)
        # func_x2 = partial(fnet_single, x=x2, i=i)
        def func_x1(params):
            return fnet_single(params, x1, i=i)

        def func_x2(params):
            return fnet_single(params, x2, i=i)

        output, vjp_fn = vjp(func_x1, params)
        # print("output {}".format(output))

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # print("vjps {}".format(vjps))
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            # print("jvps {}".format(jvps))
            return jvps

        # Here's our identity matrix
        basis = torch.eye(
            output.numel(), dtype=output.dtype, device=output.device
        ).view(output.numel(), -1)
        # print("basis {}".format(basis))
        return 1 / (delta * num_data) * vmap(get_ntk_slice)(basis)

    def ntk(x1: InputData, x2: InputData) -> TensorType[""]:
        K = torch.empty(output_dim, x1.shape[0], x2.shape[0])
        # print("K building {}".format(K.shape))
        for i in range(output_dim):
            # print("output dim {}".format(i))
            K[i, :, :] = single_output_ntk(x1, x2, i=i)
        # print("K {}".format(K.shape))
        return K

    return ntk


def predict_from_duals(
    alpha: Alpha, beta: Beta, kernel: NTK, Z: InducingPoints, jitter: float = 1e-3
):
    print("Z {}".format(Z.shape))
    Kuu = kernel(Z, Z)
    output_dim = Kuu.shape[0]
    print("Kuu {}".format(Kuu.shape))
    Iu = torch.eye(Kuu.shape[-1])[None, ...].repeat(output_dim, 1, 1)
    print("Iu {}".format(Iu.shape))
    Kuu += Iu * jitter
    # beta += I
    print("Kuu {}".format(Kuu.shape))

    assert beta.shape == Kuu.shape
    # iBKuu = torch.linalg.solve(beta + Kuu, torch.eye(Kuu.shape[-1]))
    # print("iBKuu {}".format(iBKuu.shape))
    # V = torch.matmul(torch.matmul(Kuu, iBKuu), Kuu)
    V = torch.matmul(Kuu, torch.linalg.solve(beta + Kuu, Kuu))
    print("V {}".format(V.shape))
    # iKuuViKuu = torch.linalg.solve(torch.linalg.solve(Kuu, V), Kuu, left=False)
    # print("iKuuViKuu {}".format(iKuuViKuu.shape))
    # iKuuViKuua = torch.matmul(iKuuViKuu, alpha[..., None])
    # print("iKuuVKuua {}".format(iKuuViKuua.shape))

    def predict(x: TestInput, full_cov: bool = False) -> Tuple[OutputMean, OutputVar]:
        Kxx = kernel(x, x)
        print("Kxx {}".format(Kxx.shape))
        Kxu = kernel(x, Z)
        print("Kxu {}".format(Kxu.shape))

        # f_mean = torch.matmul(Kxu, iKuuViKuua)
        # print("f_mean {}".format(f_mean.shape))
        # f_mean = f_mean[..., 0].T
        # print("f_mean {}".format(f_mean.shape))
        # Iu = torch.eye(Kuu.shape[-1])[None, ...].repeat(ouput_dim, 1, 1)
        print("Iu {}".format(Iu.shape))
        print("V {}".format(V.shape))
        # print("alpha[...,None] {}".format(alpha[..., None].shape))
        # print(
        #     "torch.eye(Kuu.shape[-1])[None, ...] {}".format(
        #         torch.eye(Kuu.shape[-1])[None, ...].shape
        #     )
        # )
        # print(
        #     "torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...]) {}".format(
        #         torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...]).shape
        #     )
        # )
        m_u = (
            V
            @ torch.linalg.solve(Kuu, Iu)
            # @ torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...])
            @ alpha[..., None]
        )
        print("m_u {}".format(m_u.shape))

        # print(
        #     "torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...] {}".format(
        #         torch.linalg.solve(Kuu, Iu).shape
        #     )
        # )
        f_mean = Kxu @ torch.linalg.solve(Kuu, Iu) @ m_u
        print("f_mean {}".format(f_mean.shape))
        f_mean = f_mean[..., 0].T
        print("f_mean {}".format(f_mean.shape))
        beta_u = torch.linalg.solve(Kuu, Iu) - torch.linalg.solve(beta + Kuu, Iu)
        print("beta_u {}".format(beta_u.shape))
        print("Kuu {}".format(Kuu.shape))
        print("Iu {}".format(Iu.shape))

        if full_cov:
            f_cov = Kxx - torch.matmul(
                torch.matmul(Kxu, iBKuu), torch.transpose(Kxu, -1, -2)
            )
            print("f_cov full_cov {}".format(f_cov.shape))
            return f_mean, f_cov
        else:
            # TODO implement more efficiently
            # f_cov = Kxx - torch.matmul(
            #     torch.matmul(Kxu, iBKuu), torch.transpose(Kxu, -1, -2)
            # )
            f_cov = Kxx - torch.matmul(
                torch.matmul(Kxu, beta_u), torch.transpose(Kxu, -1, -2)
            )
            print("f_cov {}".format(f_cov.shape))
            f_var = torch.diagonal(f_cov, dim1=-2, dim2=-1).T
            print("f_var {}".format(f_var.shape))
            return f_mean, f_var

    return predict


def calc_sparse_dual_params(
    network: torch.nn.Module,
    train_data: Tuple[InputData, OutputData],
    Z: InducingPoints,
    kernel: NTK,
    nll: Callable[[FuncData, OutputData], float],
) -> Tuple[AlphaInducing, BetaInducing]:
    num_inducing, input_dim = Z.shape
    X, Y = train_data
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == input_dim
    Kuf = kernel(Z, X)
    print("Kuf {}".format(Kuf.shape))
    F = network(X)
    print("F {}".format(F.shape))
    lambda_1, lambda_2 = calc_lambdas(Y=Y, F=F, nll=nll)
    print("lambda_1 {}".format(lambda_1.shape))
    print("lambda_2 {}".format(lambda_2.shape))
    alpha, beta = calc_sparse_dual_params_from_lambdas(
        lambda_1=lambda_1, lambda_2=lambda_2, Kuf=Kuf
    )
    print("alpha {}".format(alpha.shape))
    print("beta {}".format(beta.shape))
    return alpha, beta


def calc_sparse_dual_params_from_lambdas(
    lambda_1: Lambda_1,
    lambda_2: Lambda_2,
    Kuf: TensorType["output_dim", "num_inducing", "num_data"],
) -> Tuple[AlphaInducing, BetaInducing]:
    assert lambda_1.ndim == 2
    num_data, output_dim = lambda_1.shape
    assert lambda_2.ndim == 3
    assert lambda_2.shape[0] == num_data
    assert lambda_2.shape[1] == lambda_2.shape[2] == output_dim
    assert Kuf.ndim == 3
    assert Kuf.shape[0] == output_dim
    assert Kuf.shape[2] == num_data
    alpha_u = torch.matmul(Kuf, torch.transpose(lambda_1, -1, -2)[..., None])[..., 0]
    print("alpha_u {}".format(alpha_u.shape))
    lambda_2_diag = torch.diagonal(lambda_2, dim1=-2, dim2=-1)  # [num_data, output_dim]
    # TODO broadcast lambda_2 correctly for multiple output dims
    print("lambda_2_diag {}".format(lambda_2_diag.shape))
    # inv_lambda_2 = (
    #     torch.transpose(lambda_2_diag, -1, -2) ** -1 * torch.repeat(torch.eye(num_data)[None, ...]
    # )  # [output_dim, num_data, num_data]
    # print("inv_lambda_2 {}".format(inv_lambda_2.shape))
    inv_lambda_2 = torch.diag_embed(lambda_2_diag.T**-1)
    print("inv_lambda_2 {}".format(inv_lambda_2.shape))
    print("inv_lambda_2 {}".format(inv_lambda_2))
    beta_u = torch.matmul(
        torch.matmul(Kuf, inv_lambda_2),
        torch.transpose(Kuf, -1, -2),
    )
    print("beta_u {}".format(beta_u.shape))
    return alpha_u, beta_u


def calc_lambdas(
    Y: OutputData,  # [num_data, output_dim]
    F: FuncData,  # [num_data, output_dim]
    nll: Callable[[FuncData, OutputData], float],
) -> Tuple[Lambda_1, Lambda_2]:
    assert Y.ndim == 2
    assert F.ndim == 2
    assert Y.shape[0] == F.shape[0]
    assert Y.shape[1] == F.shape[1]
    nll_jacobian_fn = jacrev(nll)
    nll_hessian_fn = torch.vmap(hessian(nll))

    # nll_jacobian_fn = torch.gradient(nll)
    lambda_1 = nll_jacobian_fn(F, Y)
    lambda_2 = nll_hessian_fn(F, Y)
    # lambda_1, lambda_2 = [], []
    # TODO we can do better than a for loop...
    # for y, f in zip(Y, F):
    #     # lambda_1.append(nll_jacobian_fn(f, y))
    #     print("nll_hessian_fn(f, y) {}".format(nll_hessian_fn(f, y).shape))
    #     lambda_2.append(nll_hessian_fn(f, y))
    #     # TODO implement clipping for lambdas
    # lambda_1 = torch.stack(lambda_1, dim=0)  # [num_data, output_dim]
    # TODO should lambda_1 just be Y?
    lambda_1 = Y
    print("lambda_2 {}".format(lambda_2))
    # lambda_2 = torch.stack(lambda_2, dim=0)  # [num_data, output_dim, output_dim]
    return lambda_1, lambda_2
