#!/usr/bin/env python3
import logging
from typing import Callable, Tuple, Optional, List


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
    NTK_single,
    OutputData,
    OutputMean,
    OutputVar,
    TestInput,
)
from src.nn2svgp.likelihoods import Likelihood
from src.nn2svgp.priors import Prior
from torch.func import functional_call, hessian, jacrev, jvp, vjp, vmap
from torchtyping import TensorType
from torch.utils.data import DataLoader, TensorDataset


class NTKSVGP(nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        # train_data: Data,
        prior: Prior,
        likelihood: Likelihood,
        output_dim: int,
        num_inducing: int = 30,
        dual_batch_size: Optional[int] = None,
        jitter: float = 1e-6,
        device: str = "cuda",
    ):
        super().__init__()
        self.network = network
        self.prior = prior
        self.likelihood = likelihood
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        self.dual_batch_size = dual_batch_size
        self.jitter = jitter
        self.device = device

    def set_data(self, train_data: Data):
        """Sets training data, samples inducing points, calcs dual parameters, builds predict fn"""
        X_train, Y_train = train_data
        if X_train.ndim > 2:
            # TODO flatten dims for images
            logger.info("X_train.ndim>2 so flattening {}".format(X_train.shape))
            X_train = torch.flatten(X_train, 1, -1)
            logger.info("X_train shape after flattening {}".format(X_train.shape))
        assert X_train.ndim == 2
        # assert X_train.ndim >= 2
        # assert Y_train.ndim == 2
        assert X_train.shape[0] == Y_train.shape[0]
        self.train_data = (X_train.to(self.device), Y_train.to(self.device))
        # num_data, input_dim = X_train.shape
        num_data = Y_train.shape[0]
        # print("Y_train.shape {}".format(Y_train.shape))
        indices = torch.randperm(num_data)[: self.num_inducing]
        # TODO will this work for image classification??
        self.Z = X_train[indices].to(self.device)
        # self.Z = X_train
        assert self.Z.ndim == 2
        # num_inducing = 100
        # self.Z = torch.linspace(-1, 3, self.num_inducing).reshape(self.num_inducing, -1)
        # Z = torch.rand(num_inducing, 1) * 3
        # self.num_inducing, self.input_dim = Z.shape

        self.build_dual_svgp()

    def build_dual_svgp(self):
        logger.info("Calculating dual params and building prediction fn...")
        if isinstance(self.prior, src.nn2svgp.priors.Gaussian):
            delta = self.prior.delta
        else:
            raise NotImplementedError(
                "what should delta be if not using Gaussian prior???"
            )
        self.kernel, self.kernel_single = build_ntk(
            network=self.network,
            num_data=self.num_data,
            output_dim=self.output_dim,
            delta=delta,
        )

        if self.dual_batch_size:
            self.alpha, self.beta = calc_sparse_dual_params_batch(
                network=self.network,
                train_loader=DataLoader(
                    TensorDataset(*(self.train_data)),
                    batch_size=self.dual_batch_size,
                    shuffle=False,
                ),
                Z=self.Z,
                likelihood=self.likelihood,
                kernel=self.kernel_single,
                nll=self.likelihood.nn_loss,
                out_dim=self.output_dim,
            )
        else:
            self.alpha, self.beta = calc_sparse_dual_params(
                network=self.network,
                train_data=self.train_data,
                Z=self.Z,
                kernel=self.kernel,
                nll=self.likelihood.nn_loss,
                likelihood=self.likelihood,
            )

        logger.info("Finished calculating dual params")

        # print("alpha {}".format(self.alpha.shape))
        # print("beta {}".format(self.beta.shape))
        assert self.alpha.ndim == 2
        assert self.beta.ndim == 3
        assert self.alpha.shape[0] == self.output_dim
        assert self.alpha.shape[1] == self.num_inducing
        assert self.beta.shape[0] == self.output_dim
        assert self.beta.shape[1] == self.num_inducing
        assert self.beta.shape[2] == self.num_inducing

        Kzz = self.kernel(self.Z, self.Z)
        output_dim = Kzz.shape[0]
        # print("Kuu {}".format(Kzz.shape))
        Iz = torch.eye(Kzz.shape[-1])[None, ...].repeat(output_dim, 1, 1)
        # print("Iu {}".format(Iz.shape))
        Kzz += Iz * self.jitter
        self.alpha_2 = torch.linalg.solve((Kzz + self.beta), self.alpha[..., None])
        self._predict_fn = predict_from_duals(
            alpha=self.alpha,
            beta=self.beta,
            kernel=self.kernel,
            Z=self.Z,
            jitter=self.jitter,
        )
        logger.info("Finished building predict fn")

    def forward(self, x: InputData):
        return self.predict(x=x)

    @torch.no_grad()
    def predict_mean(self, x: TestInput) -> FuncMean:
        # Kxx = self.kernel(x, x)
        # print("Kxx {}".format(Kxx.shape))
        Kxz = self.kernel(x, self.Z)
        # print("Kxz {}".format(Kxz.shape))

        f_mean = Kxz @ self.alpha_2
        # print("f_mean {}".format(f_mean.shape))
        # f_mean = f_mean @ alpha[..., None]
        # print("f_mean {}".format(f_mean.shape))
        f_mean = f_mean[..., 0].T
        # print("f_mean {}".format(f_mean.shape))
        return f_mean

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
        # neg_log_prior = self.prior.nn_loss(self.network.parameters)
        return neg_log_likelihood + neg_log_prior

    def update(self, x: InputData, y: OutputData):
        if not isinstance(self.likelihood, src.nn2svgp.likelihoods.Gaussian):
            raise NotImplementedError
        logger.info("Updating dual params...")
        # TODO what about classificatin
        # assert x.ndim == 2 and y.ndim == 2
        if x.ndim > 2:
            print("x before flatten {}".format(x.shape))
            x = torch.flatten(x, 1, -1)
            print("x AFTER flatten {}".format(x.shape))
        assert x.ndim == 2
        # if y.ndim == 1:
        #     y = y[None, ...]
        #     # TODO should this be somewhere else?
        #     print("y.shape {}".format(y.shape))
        #     y = torch.nn.functional.one_hot(y, num_classes=2)
        #     y = y.to(torch.double)
        #     print("y.shape {}".format(y.shape))
        #     print("y.shape {}".format(y))
        # y = y[:, None]
        # assert x.ndim == 2 and y.ndim == 2
        num_new_data, input_dim = x.shape
        Kzx = self.kernel(self.Z, x)
        # print("Kzx {}".format(Kzx.shape))
        # print("alpha {}".format(self.alpha.shape))
        # print("beta {}".format(self.beta.shape))
        # print("x {}".format(x.shape))
        # print("y {}".format(y.shape))

        # lambda_1, lambda_2 = calc_lambdas(Y=Y, F=F, nll=nll)

        # TODO only works for regression (should be lambda_1)

        # f = self.network(x)
        # lambda_1, _ = calc_lambdas(Y=y, F=f, nll=None, likelihood=self.likelihood)
        # lambda_1_minus_y = y-lambda_1
        # print(" hereh lambda_1 {}".format(lambda_1.shape))
        # self.alpha += (Kzx @ lambda_1_minus_y.T[..., None])[..., 0]
        self.alpha += (Kzx @ y.T[..., None])[..., 0]
        self.beta += (
            Kzx
            @ (1**-1 * torch.eye(num_new_data)[None, ...])
            @ torch.transpose(Kzx, -1, -2)
        )
        # print("ALPHA {}".format(self.alpha.shape))
        # print("BETA {}".format(self.beta.shape))

        logger.info("Building predict fn...")
        self._predict_fn = predict_from_duals(
            alpha=self.alpha,
            beta=self.beta,
            kernel=self.kernel,
            Z=self.Z,
            jitter=self.jitter,
        )
        logger.info("Finished build predict fn")

    @property
    def num_data(self):
        return self.train_data[0].shape[0]

    # @property
    # def num_inducing(self):
    #     try:
    #         self._num_in
    #     return self.Z.shape[0]

    # @property
    # def output_dim(self):
    #     return self.train_data[1].shape[1]


def build_ntk(
    network: nn.Module, num_data: int, output_dim: int, delta: float = 1.0
) -> Tuple[NTK, NTK_single]:
    # Detaching the parameters because we won't be calling Tensor.backward().
    params = {k: v.detach() for k, v in network.named_parameters()}

    def fnet_single(params, x, i):
        # print("fnet_single {}".format(x.shape))
        # print("fnet_single {}".format(x.unsqueeze(0).shape))
        f = functional_call(network, params, (x,))[:, i]
        # print("f {}".format(f.shape))
        # f = functional_call(network, params, (x.unsqueeze(0),))[0, ...]
        # print("f {}".format(f.shape))
        # f = functional_call(network, params, (x.unsqueeze(0),))[0, ...][:, i]
        return f
        # return functional_call(network, params, (x.unsqueeze(0),))[0, ...][:, i]

    def single_output_ntk_contraction(x1, x2, i):
        def fnet_single(params, x):
            # print("fnet_single {}".format(x.shape))
            # print("fnet_single {}".format(x.unsqueeze(0).shape))
            f = functional_call(network, params, (x.unsqueeze(0),))[:, i]
            # f = functional_call(network, params, (x,))[:, i]
            # f = functional_call(network, params, (x.unsqueeze(0)))[
            #     :, i
            # ]  # TODO: Why using self.net doesn't work?
            # print("f {}".format(f.shape))
            # f = functional_call(network, params, (x.unsqueeze(0),))[0, ...]
            # print("f {}".format(f.shape))
            # f = functional_call(network, params, (x.unsqueeze(0),))[0, ...][:, i]
            return f

        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        # print("jac1 {}".format(jac1))
        jac1 = [j.flatten(2) for j in jac1.values()]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2.values()]

        # Compute J(x1) @ J(x2).T
        einsum_expr = None
        compute = "full"
        if compute == "full":
            einsum_expr = "Naf,Mbf->NMab"
        elif compute == "trace":
            einsum_expr = "Naf,Maf->NM"
        elif compute == "diagonal":
            einsum_expr = "Naf,Maf->NMa"
        else:
            assert False

        result = torch.stack(
            [torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)]
        )
        result = result.sum(0)
        # return result
        return 1 / (delta * num_data) * result

    # @torch.compile(backend="eager")
    def single_output_ntk(x1, x2, i):
        # def single_output_ntk(x1: InputData, x2: InputData, i):
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
        # return 1 / (delta) * vmap(get_ntk_slice)(basis)

    def ntk(x1, x2):
        # print("INSIDE kernel {}".format(output_dim))
        # def ntk(x1: InputData, x2: InputData) -> TensorType[""]:
        K = torch.empty(output_dim, x1.shape[0], x2.shape[0])
        # print("K building {}".format(K.shape))
        # print("x1 {}".format(x1.shape))
        # print("x2 {}".format(x2.shape))
        # Ks = []
        for i in range(output_dim):
            # print("output dim {}".format(i))
            # Ks.append(single_output_ntk(x1, x2, i=i))
            K[i, :, :] = single_output_ntk(x1, x2, i=i)
            # k = single_output_ntk(x1, x2, i=i)
            # k = single_output_ntk_contraction(x1, x2, i=i)
            # print("k {}".format(k))
            # K[i, :, :] = k[..., 0, 0]
        # K = torch.stack(Ks, 0)
        # print("K {}".format(K.shape))
        return K

    return ntk, single_output_ntk


def predict_from_duals(
    alpha: Alpha, beta: Beta, kernel: NTK, Z: InducingPoints, jitter: float = 1e-3
):
    # print("Z {}".format(Z.shape))
    Kzz = kernel(Z, Z)
    output_dim = Kzz.shape[0]
    # print("Kuu {}".format(Kzz.shape))
    Iz = torch.eye(Kzz.shape[-1])[None, ...].repeat(output_dim, 1, 1)
    # print("Iu {}".format(Iz.shape))
    Kzz += Iz * jitter
    # beta += I
    # print("Kuu {}".format(Kzz.shape))

    assert beta.shape == Kzz.shape
    # iBKuu = torch.linalg.solve(beta + Kuu, torch.eye(Kuu.shape[-1]))
    # print("iBKuu {}".format(iBKuu.shape))
    # V = torch.matmul(torch.matmul(Kuu, iBKuu), Kuu)0
    # V = torch.matmul(Kzz, torch.linalg.solve(beta + Kzz, Kzz))
    # print("V {}".format(V.shape))
    # iKuuViKuu = torch.linalg.solve(torch.linalg.solve(Kuu, V), Kuu, left=False)
    # print("iKuuViKuu {}".format(iKuuViKuu.shape))
    # iKuuViKuua = torch.matmul(iKuuViKuu, alpha[..., None])
    # print("iKuuVKuua {}".format(iKuuViKuua.shape))

    # @torch.compile(backend="aot_eager")
    # @torch.jit.script
    def predict(x, full_cov: bool = False):
        # def predict(x: TestInput, full_cov: bool = False) -> Tuple[OutputMean, OutputVar]:
        Kxx = kernel(x, x)
        # print("Kxx {}".format(Kxx.shape))
        Kxz = kernel(x, Z)
        # print("Kxz {}".format(Kxz.shape))

        # f_mean = torch.matmul(Kxu, iKuuViKuua)
        # print("f_mean {}".format(f_mean.shape))
        # f_mean = f_mean[..., 0].T
        # print("f_mean {}".format(f_mean.shape))
        # Iu = torch.eye(Kuu.shape[-1])[None, ...].repeat(ouput_dim, 1, 1)
        # print("Iu {}".format(Iz.shape))
        # print("V {}".format(V.shape))
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
        # Kux = torch.transpose(Kxu, -1, -2)
        # m_u = Kux @ alpha
        # print("m_u {}".format(m_u.shape))

        # print(
        #     "torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...] {}".format(
        #         torch.linalg.solve(Kuu, Iu).shape
        #     )
        # )
        # beta_u = torch.linalg.solve(Kuu, Iu) - torch.linalg.solve(beta + Kuu, Iu)
        # Kzx = torch.transpose(Kxz, -1, -2)
        # beta_u = Kux @ beta @ Kxu
        beta_u = beta
        # print("beta_u {}".format(beta_u.shape))
        # print("Kzz {}".format(Kzz.shape))
        # print("Iz {}".format(Iz.shape))

        # m_u = (
        #     V
        #     @ torch.linalg.solve((Kuu), Iu)
        #     # @ torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...])
        #     @ alpha[..., None]
        # )
        # # Kux = torch.transpose(Kxu, -1, -2)
        # print("alpha {}".format(alpha.shape))
        # # print("Kux {}".format(Kux.shape))
        # # m_u = Kux @ alpha[..., None]
        # # print("m_u {}".format(m_u.shape))
        # f_mean = Kxu @ torch.linalg.solve((Kuu), Iu) @ m_u
        # f_mean = f_mean[..., 0].T

        # f_mean = Kxz @ torch.linalg.solve((Kzz + beta_u), Iz)
        f_mean = Kxz @ torch.linalg.solve((Kzz + beta_u), alpha[..., None])
        # print("f_mean {}".format(f_mean.shape))
        # f_mean = f_mean @ alpha[..., None]
        # print("f_mean {}".format(f_mean.shape))
        f_mean = f_mean[..., 0].T
        # print("f_mean {}".format(f_mean.shape))

        if full_cov:
            raise NotImplementedError
            f_cov = Kxx - torch.matmul(
                torch.matmul(Kxz, iBKuu), torch.transpose(Kxz, -1, -2)
            )
            # print("f_cov full_cov {}".format(f_cov.shape))
            return f_mean, f_cov
        else:
            # TODO implement more efficiently
            # f_cov = Kxx - torch.matmul(
            #     torch.matmul(Kxu, iBKuu), torch.transpose(Kxu, -1, -2)
            # )
            # beta_u = torch.linalg.solve(Kuu, Iu) - torch.linalg.solve(beta + Kuu, Iu)
            tmp = torch.linalg.solve(Kzz, Iz) - torch.linalg.solve(beta_u + Kzz, Iz)
            f_cov = Kxx - torch.matmul(
                torch.matmul(Kxz, tmp), torch.transpose(Kxz, -1, -2)
            )
            # f_cov = Kxx - torch.matmul(
            #     Kxu, torch.matmul(beta_u, torch.transpose(Kxu, -1, -2))
            # )
            # print("f_cov {}".format(f_cov.shape))
            f_var = torch.diagonal(f_cov, dim1=-2, dim2=-1).T
            # print("f_var {}".format(f_var.shape))
            return f_mean, f_var

    return predict


def calc_sparse_dual_params_batch(
    network: torch.nn.Module,
    train_loader: DataLoader,
    # train_data: Tuple[InputData, OutputData],
    Z: InducingPoints,
    kernel: NTK_single,
    nll: Callable[[FuncData, OutputData], float],
    likelihood,
    batch_size: int = 1000,
    out_dim: int = 10,
    subset_out_dim: Optional[List] = None,
    device: str = "cpu",
) -> Tuple[AlphaInducing, BetaInducing]:
    ################  Compute lambdas batched version START ################
    num_train = len(train_loader.dataset)
    items_shape = (num_train, out_dim)
    lambda_1 = torch.zeros(items_shape).to(device)
    lambda_2_diag = torch.zeros(items_shape).to(device)

    start_idx = 0
    end_idx = 0

    for batch in train_loader:
        x_i, y_i = batch[0], batch[1]
        x_i, y_i = x_i.to(device), y_i.to(device)
        batch_size = x_i.shape[0]
        logits_i = network(x_i)
        if logits_i.ndim == 1:
            logits_i = logits_i.unsqueeze(-1)
        lambda_1_i, lambda_2_i = calc_lambdas(
            Y=y_i, F=logits_i, nll=nll, likelihood=likelihood
        )
        lambda_2_i = torch.vmap(torch.diag)(lambda_2_i)

        end_idx = start_idx + batch_size
        lambda_1[start_idx:end_idx] = lambda_1_i
        lambda_2_diag[start_idx:end_idx] = lambda_2_i
        start_idx = end_idx

    # Clip the lambdas  (?)
    # lambda_1 = torch.clip(lambda_1, min=1e-7)
    # lambda_2_diag = torch.clip(lambda_2_diag, min=1e-7)
    ################  Compute lambdas batched version END ################

    ################  Compute alpha/beta batched version START ################
    alpha = torch.zeros((out_dim, Z.shape[0])).to(device)
    beta = torch.zeros((out_dim, Z.shape[0], Z.shape[0])).to(device)

    ## TODO: lambda2 ^ -1 ???
    def compute_beta_i(kiu, lambda2_i):
        # return torch.einsum('u, i, m -> um', kiu, lambda2_i**-1, kiu)
        return torch.outer(kiu, kiu) * lambda2_i

    if subset_out_dim is not None:
        out_dim_range = subset_out_dim
    else:
        out_dim_range = list(range(out_dim))

    for output_c in out_dim_range:
        start_idx = 0
        end_idx = 0
        for batch in train_loader:
            x_i, y_i = batch[0], batch[1]
            x_i, y_i = x_i.to(device), y_i.to(device)
            batch_size = x_i.shape[0]
            end_idx = start_idx + batch_size
            Kui_c = kernel(Z, x_i, output_c)
            alpha[output_c] += torch.einsum(
                "ub, b -> u", Kui_c, lambda_1[start_idx:end_idx, output_c]
            )
            beta[output_c] += (
                vmap(compute_beta_i)(
                    Kui_c.T, lambda_2_diag[start_idx:end_idx, output_c][:, None]
                )
            ).sum(dim=0)
            start_idx = end_idx
    return alpha, beta
    ################  Compute alpha/beta batched version END ################


def calc_sparse_dual_params(
    network: torch.nn.Module,
    train_data: Tuple[InputData, OutputData],
    Z: InducingPoints,
    kernel: NTK,
    nll: Callable[[FuncData, OutputData], float],
    likelihood,
) -> Tuple[AlphaInducing, BetaInducing]:
    num_inducing, input_dim = Z.shape
    X, Y = train_data
    # if Y.ndim == 1:
    #     Y = Y[..., None]
    assert X.ndim == 2
    # assert Y.ndim == 2
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == input_dim
    Kzx = kernel(Z, X)
    # print("Kzx {}".format(Kzx.shape))
    F = network(X)
    # print("F {}".format(F.shape))
    lambda_1, lambda_2 = calc_lambdas(Y=Y, F=F, nll=nll, likelihood=likelihood)
    # print("lambda_1 {}".format(lambda_1.shape))
    # print("lambda_2 {}".format(lambda_2.shape))
    alpha, beta = calc_sparse_dual_params_from_lambdas(
        lambda_1=lambda_1, lambda_2=lambda_2, Kzx=Kzx
    )
    # print("alpha {}".format(alpha.shape))
    # print("beta {}".format(beta.shape))
    return alpha, beta


def calc_sparse_dual_params_from_lambdas(
    lambda_1: Lambda_1,
    lambda_2: Lambda_2,
    Kzx: TensorType["output_dim", "num_inducing", "num_data"],
) -> Tuple[AlphaInducing, BetaInducing]:
    assert lambda_1.ndim == 2
    num_data, output_dim = lambda_1.shape
    assert lambda_2.ndim == 3
    assert lambda_2.shape[0] == num_data
    assert lambda_2.shape[1] == lambda_2.shape[2] == output_dim
    assert Kzx.ndim == 3
    assert Kzx.shape[0] == output_dim
    assert Kzx.shape[2] == num_data
    alpha_u = torch.matmul(Kzx, torch.transpose(lambda_1, -1, -2)[..., None])[..., 0]
    # print("alpha_u {}".format(alpha_u.shape))
    lambda_2_diag = torch.diagonal(lambda_2, dim1=-2, dim2=-1)  # [num_data, output_dim]
    # TODO broadcast lambda_2 correctly for multiple output dims
    # print("lambda_2_diag {}".format(lambda_2_diag.shape))
    # inv_lambda_2 = (
    #     torch.transpose(lambda_2_diag, -1, -2) ** -1 * torch.repeat(torch.eye(num_data)[None, ...]
    # )  # [output_dim, num_data, num_data]
    # print("inv_lambda_2 {}".format(inv_lambda_2.shape))
    lambda_2 = torch.diag_embed(lambda_2_diag.T)
    # print("lambda_2 {}".format(lambda_2.shape))
    # print("lambda_2 {}".format(lambda_2))
    beta_u = torch.matmul(
        torch.matmul(Kzx, lambda_2),
        torch.transpose(Kzx, -1, -2),
    )
    # print("beta_u {}".format(beta_u.shape))
    return alpha_u, beta_u


def calc_lambdas(
    Y: OutputData,  # [num_data, output_dim]
    F: FuncData,  # [num_data, output_dim]
    nll: Callable[[FuncData, OutputData], float],
    likelihood,
) -> Tuple[Lambda_1, Lambda_2]:
    # assert Y.ndim == 2
    assert F.ndim == 2
    assert Y.shape[0] == F.shape[0]
    nll = likelihood.nn_loss
    #  assert Y.shape[1] == F.shape[1]
    nll_jacobian_fn = jacrev(nll)
    nll_hessian_fn = torch.vmap(hessian(nll))

    # nll_jacobian_fn = torch.gradient(nll)
    # lambda_1 = nll_jacobian_fn(F, Y)
    lambda_2 = 2 * nll_hessian_fn(F, Y)  # [num_data, output_dim, output_dim]
    lambda_1 = -1 * nll_jacobian_fn(F, Y)  # [num_data, output_dim]
    # lambda_2 = torch.vmap(likelihood.Hessian, in_dims=0)(F)
    # lambda_1 = -torch.vmap(likelihood.residual, in_dims=0)(F)
    lambda_2 = 2 * likelihood.Hessian(f=F)
    # lambda_1 = -likelihood.residual(y=Y, f=F)
    # print("lambda_1 {}".format(lambda_1.shape))
    # print("lambda_2 {}".format(lambda_2.shape))
    # print("lambda_1 {}".format(lambda_1))
    # print("lambda_2 {}".format(lambda_2))
    # exit()
    lambda_2_diag = torch.diagonal(lambda_2, dim1=-2, dim2=-1)  # [num_data, output_dim]
    # print("Y-lambda_1 {}".format(Y - lambda_1))
    # print("lambda_2_diag {}".format(lambda_2_diag.shape))
    # print("F {}".format(F.shape))
    lambda_1 += F * lambda_2_diag
    # lambda_1 = F * lambda_2_diag
    # lambda_1 = F * lambda_2_diag
    # print("lambda_1 {}".format(lambda_1.shape))
    # print("lambda_1 {}".format(lambda_1))
    # print("Y {}".format(Y))
    # print("Y-lambda_1 {}".format(Y - lambda_1))
    # lambda_1, lambda_2 = [], []
    # TODO we can do better than a for loop...
    # for y, f in zip(Y, F):
    #     # lambda_1.append(nll_jacobian_fn(f, y))
    #     print("nll_hessian_fn(f, y) {}".format(nll_hessian_fn(f, y).shape))
    #     lambda_2.append(nll_hessian_fn(f, y))
    #     # TODO implement clipping for lambdas
    # lambda_1 = torch.stack(lambda_1, dim=0)  # [num_data, output_dim]
    # TODO should lambda_1 just be Y?
    # lambda_1 = Y
    # print("lambda_2 {}".format(lambda_2.shape))
    # lambda_2 = torch.stack(lambda_2, dim=0)  # [num_data, output_dim, output_dim]
    return lambda_1, lambda_2


# TODO: need that to be out of the function
def loss_cl(
    x: InputData,
    y: OutputData,
    network: nn.Module,
    likelihood: Likelihood,
    prior: Prior,
):
    f = network(x)
    neg_log_likelihood = likelihood.nn_loss(f=f, y=y)
    neg_log_prior = prior.nn_loss()
    return neg_log_likelihood + neg_log_prior
