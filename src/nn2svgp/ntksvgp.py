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
    Beta,
    AlphaInducing,
    BetaInducing,
    Data,
    FuncData,
    FuncMean,
    FuncVar,
    InducingPoints,
    InputData,
    # Lambda_1,
    Lambda,
    # Lambda_2,
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
        device: str = "cpu",
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
        # if X_train.ndim > 2:
        #     # TODO flatten dims for images
        #     logger.info("X_train.ndim>2 so flattening {}".format(X_train.shape))
        #     X_train = torch.flatten(X_train, 1, -1)
        #     logger.info("X_train shape after flattening {}".format(X_train.shape))
        # assert X_train.ndim == 2
        # assert X_train.ndim >= 2
        # assert Y_train.ndim == 2
        assert X_train.shape[0] == Y_train.shape[0]
        self.train_data = (X_train, Y_train)
        # self.train_data = (X_train.to(self.device), Y_train.to(self.device))
        # num_data, input_dim = X_train.shape
        num_data = Y_train.shape[0]
        indices = torch.randperm(num_data)[: self.num_inducing]
        # TODO will this work for image classification??
        self.Z = X_train[indices.to(X_train.device)].to(self.device)
        # self.Z = X_train
        # assert self.Z.ndim == 2
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
            self.alpha_u, self.beta_u = calc_sparse_dual_params_batch(
                network=self.network,
                train_loader=DataLoader(
                    TensorDataset(*(self.train_data)),
                    batch_size=self.dual_batch_size,
                    shuffle=False,
                ),
                Z=self.Z,
                likelihood=self.likelihood,
                kernel=self.kernel_single,
                jitter=self.jitter,
                out_dim=self.output_dim,
                device=self.device,
            )
        else:
            self.alpha_u, self.beta_u, self.Lambda_u = calc_sparse_dual_params(
                network=self.network,
                train_data=self.train_data,
                Z=self.Z,
                kernel=self.kernel,
                likelihood=self.likelihood,
                jitter=self.jitter,
            )

        logger.info("Finished calculating dual params")

        assert self.alpha_u.ndim == 2
        assert self.beta_u.ndim == 3
        assert self.alpha_u.shape[0] == self.output_dim
        assert self.alpha_u.shape[1] == self.num_inducing
        assert self.beta_u.shape[0] == self.output_dim
        assert self.beta_u.shape[1] == self.num_inducing
        assert self.beta_u.shape[2] == self.num_inducing

        # self.alpha_2 = torch.linalg.solve((Kzz + self.beta_u), self.alpha_u[..., None])
        self._predict_fn = predict_from_sparse_duals(
            alpha_u=self.alpha_u,
            beta_u=self.beta_u,
            kernel=self.kernel,
            Z=self.Z,
            jitter=self.jitter,
        )
        logger.info("Finished building predict fn")

    def forward(self, x: InputData):
        return self.predict(x=x)

    @torch.no_grad()
    def predict_mean(self, x: TestInput) -> FuncMean:
        x = x.to(self.Z.device)
        Kxz = self.kernel(x, self.Z)
        # print("kxz {}".format(Kxz.shape))
        # print("alpa_u {}".format(self.alpha_u.shape))
        f_mean = (Kxz @ self.alpha_u[..., None])[..., 0].T
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
        num_new_data, input_dim = x.shape
        Kzx = self.kernel(self.Z, x)

        # lambda_1, lambda_2 = calc_lambdas(Y=Y, F=F, nll=nll)

        # TODO only works for regression (should be lambda_1)

        # f = self.network(x)
        # lambda_1, _ = calc_lambdas(Y=y, F=f, nll=None, likelihood=self.likelihood)
        # lambda_1_minus_y = y-lambda_1
        # print(" hereh lambda_1 {}".format(lambda_1.shape))
        # self.alpha += (Kzx @ lambda_1_minus_y.T[..., None])[..., 0]
        # self.alpha_u += (Kzx @ y.T[..., None])[..., 0]
        self.beta_u += (
            Kzx
            @ (1**-1 * torch.eye(num_new_data).to(self.Z)[None, ...])
            @ torch.transpose(Kzx, -1, -2)
        )
        self.Lambda_u += (Kzx @ y.T[..., None])[..., 0]
        self.alpha_u = calc_alpha_u_from_lambda(
            beta_u=self.beta_u,
            Lambda_u=self.Lambda_u,
            Z=self.Z,
            kernel=self.kernel,
            jitter=self.jitter,
        )

        logger.info("Building predict fn...")
        self._predict_fn = predict_from_sparse_duals(
            alpha_u=self.alpha_u,
            beta_u=self.beta_u,
            kernel=self.kernel,
            Z=self.Z,
            jitter=self.jitter,
        )
        logger.info("Finished build predict fn")

    @property
    def num_data(self):
        return self.train_data[0].shape[0]


def build_ntk(
    network: nn.Module, num_data: int, output_dim: int, delta: float = 1.0
) -> Tuple[NTK, NTK_single]:
    network = network.eval()
    # Detaching the parameters because we won't be calling Tensor.backward().
    params = {k: v.detach() for k, v in network.named_parameters()}

    def fnet_single(params, x, i):
        f = functional_call(network, params, (x,))[:, i]
        return f

    def single_output_ntk_contraction(
        x1: InputData, x2: InputData, i: int, full_cov: Optional[bool] = True
    ):
        def fnet_single(params, x):
            f = functional_call(network, params, (x.unsqueeze(0),))[:, i]
            return f

        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        # print("jac1 {}".format(jac1.shape))
        jac1 = [j.flatten(2) for j in jac1.values()]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        # print("jac2 {}".format(jac2.shape))
        jac2 = [j.flatten(2) for j in jac2.values()]

        # Compute J(x1) @ J(x2).T
        einsum_expr = None
        if full_cov:
            einsum_expr = "Naf,Mbf->NMab"
        else:
            einsum_expr = "Naf,Maf->NMa"
        result = torch.stack(
            [torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)]
        )
        result = result.sum(0)
        if full_cov:
            return 1 / (delta * num_data) * result[..., 0, 0]
        else:
            # TODO this could be more efficient
            result = torch.diagonal(result[..., 0], dim1=-1, dim2=-2)
            return 1 / (delta * num_data) * result

    # @torch.compile(backend="eager")
    def single_output_ntk(
        X1: InputData, X2: InputData, i: int, full_cov: Optional[bool] = True
    ):
        def func_X1(params):
            return fnet_single(params, X1, i=i)

        def func_X2(params):
            return fnet_single(params, X2, i=i)

        output, vjp_fn = vjp(func_X1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # print("vjps {}".format(vjps))
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_X2, (params,), vjps)
            # print("jvps {}".format(jvps))
            return jvps

        # Here's our identity matrix
        basis = torch.eye(
            output.numel(), dtype=output.dtype, device=output.device
        ).view(output.numel(), -1)
        return 1 / (delta * num_data) * vmap(get_ntk_slice)(basis)

    def ntk(X1: InputData, X2: Optional[InputData], full_cov: bool = True):
        if X2 is None:
            X2 = X1
        if full_cov:
            K = torch.empty(output_dim, X1.shape[0], X2.shape[0]).to(X1.device)
        else:
            K = torch.empty(output_dim, X1.shape[0]).to(X1.device)
        for i in range(output_dim):
            # K[i, ...] = single_output_ntk(X1, X2, i=i, full_cov=full_cov)
            K[i, ...] = single_output_ntk_contraction(X1, X2, i=i, full_cov=full_cov)
        return K

    return ntk, single_output_ntk_contraction


def predict_from_sparse_duals(
    alpha_u: AlphaInducing,
    beta_u: BetaInducing,
    kernel: NTK,
    Z: InducingPoints,
    jitter: float = 1e-3,
):
    Kzz = kernel(Z, Z)
    output_dim = Kzz.shape[0]
    Iz = (
        torch.eye(Kzz.shape[-1], dtype=torch.float64)
        .to(Z.device)[None, ...]
        .repeat(output_dim, 1, 1)
    )
    Kzz += Iz * jitter
    KzzplusBeta = (Kzz + beta_u) + Iz * jitter
    assert beta_u.shape == Kzz.shape

    Lm = torch.linalg.cholesky(Kzz, upper=True)
    Lb = torch.linalg.cholesky(KzzplusBeta, upper=True)

    def predict(x, full_cov: bool = False) -> Tuple[OutputMean, OutputVar]:
        Kxx = kernel(x, x, full_cov=full_cov)
        Kxz = kernel(x, Z)

        f_mean = (Kxz @ alpha_u[..., None])[..., 0].T

        if full_cov:
            # TODO tmp could be computed before
            tmp = torch.linalg.solve(Kzz, Iz) - torch.linalg.solve(beta_u + Kzz, Iz)
            f_cov = Kxx - torch.matmul(
                torch.matmul(Kxz, tmp), torch.transpose(Kxz, -1, -2)
            )
            return f_mean, f_cov
        else:
            Kzx = torch.transpose(Kxz, -1, -2)
            Am = torch.linalg.solve_triangular(
                torch.transpose(Lm, -1, -2), Kzx, upper=False
            )
            Ab = torch.linalg.solve_triangular(
                torch.transpose(Lb, -1, -2), Kzx, upper=False
            )
            f_var = (
                Kxx - torch.sum(torch.square(Am), -2) + torch.sum(torch.square(Ab), -2)
            )
            return f_mean, f_var.T

    return predict


@torch.no_grad()
def calc_sparse_dual_params_batch(
    network: torch.nn.Module,
    train_loader: DataLoader,
    Z: InducingPoints,
    kernel: NTK_single,
    likelihood,
    out_dim: int = 10,
    jitter: float = 1e-4,
    subset_out_dim: Optional[List] = None,
    device: str = "cpu",
) -> Tuple[AlphaInducing, BetaInducing]:
    ################  Compute lambdas batched version START ################
    num_train = len(train_loader.dataset)
    items_shape = (num_train, out_dim)

    # rename lambda_1 is Lambda, lamba2 is beta
    Lambda = torch.zeros(items_shape).cpu()
    beta_diag = torch.zeros(items_shape).cpu()

    start_idx = 0
    end_idx = 0

    for batch in train_loader:
        x_i, y_i = batch[0], batch[1]
        x_i, y_i = x_i.to(device), y_i.to(device)
        batch_size = x_i.shape[0]
        logits_i = network(x_i)
        if logits_i.ndim == 1:
            logits_i = logits_i.unsqueeze(-1)
        Lambda_i, beta_i = calc_lambdas(Y=y_i, F=logits_i, likelihood=likelihood)
        beta_i = torch.vmap(torch.diag)(beta_i)

        end_idx = start_idx + batch_size
        Lambda[start_idx:end_idx] = Lambda_i
        beta_diag[start_idx:end_idx] = beta_i
        start_idx = end_idx

    # Clip the lambdas  (?)
    # lambda_1 = torch.clip(lambda_1, min=1e-7)
    # lambda_2_diag = torch.clip(lambda_2_diag, min=1e-7)
    ################  Compute lambdas batched version END ################

    ################  Compute alpha/beta batched version START ################
    alpha_u = torch.zeros((out_dim, Z.shape[0])).cpu()
    beta_u = torch.zeros((out_dim, Z.shape[0], Z.shape[0])).cpu()

    if subset_out_dim is not None:
        out_dim_range = subset_out_dim
    else:
        out_dim_range = list(range(out_dim))

    for output_c in out_dim_range:
        start_idx = 0
        end_idx = 0
        logging.info(f"Computing covariance for class {output_c}")
        for batch in train_loader:
            x_i, y_i = batch[0], batch[1]
            x_i, y_i = x_i.to(device), y_i.to(device)
            batch_size = x_i.shape[0]
            end_idx = start_idx + batch_size
            Kui_c = kernel(Z, x_i, output_c).cpu()

            Lambda_batch = Lambda[start_idx:end_idx, output_c]
            beta_batch = beta_diag[start_idx:end_idx, output_c]
            alpha_batch = torch.einsum("mb, b -> m", Kui_c, Lambda_batch)
            beta_batch = torch.einsum("mb, b, nb -> mn", Kui_c, beta_batch, Kui_c)

            alpha_u[output_c] += alpha_batch.cpu()
            beta_u[output_c] += beta_batch.cpu()

            start_idx = end_idx
            del Kui_c
        torch.cuda.empty_cache()
        Kzz_c = (
            kernel(Z, Z, output_c).cpu() + torch.eye(Z.shape[0], device="cpu") * jitter
        )
        alpha_u[output_c] = torch.linalg.solve(
            (Kzz_c + beta_u[output_c]), alpha_u[output_c]
        )

    return alpha_u.to(device), beta_u.to(device)
    ################  Compute alpha/beta batched version END ################


def calc_sparse_dual_params(
    network: torch.nn.Module,
    train_data: Tuple[InputData, OutputData],
    Z: InducingPoints,
    kernel: NTK,
    likelihood: Likelihood,
    jitter: float = 1e-3,
) -> Tuple[AlphaInducing, BetaInducing, Lambda]:
    num_inducing, input_dim = Z.shape
    X, Y = train_data
    assert X.ndim == 2
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == input_dim
    Kzx = kernel(Z, X)

    F = network(X)
    Lambda, beta = calc_lambdas(Y=Y, F=F, likelihood=likelihood)
    assert Lambda.ndim == 2
    num_data, output_dim = Lambda.shape
    assert beta.ndim == 3
    assert beta.shape[0] == num_data
    assert beta.shape[1] == beta.shape[2] == output_dim
    assert Kzx.ndim == 3
    assert Kzx.shape[0] == output_dim
    assert Kzx.shape[2] == num_data
    beta_diag = torch.diagonal(beta, dim1=-2, dim2=-1)  # [num_data, output_dim]
    beta = torch.diag_embed(beta_diag.T)  # [output_dim, num_data, num_data]
    beta_u = torch.matmul(torch.matmul(Kzx, beta), torch.transpose(Kzx, -1, -2))
    # print("beta_u {}".format(beta_u.shape))
    # beta_u = beta_u + Iz * jitter

    Lambda_u = torch.matmul(Kzx, torch.transpose(Lambda, -1, -2)[..., None])[..., 0]
    # print("Lambda_u {}".format(Lambda_u.shape))

    alpha_u = calc_alpha_u_from_lambda(
        beta_u=beta_u, Lambda_u=Lambda_u, Z=Z, kernel=kernel, jitter=jitter
    )
    # print("(Kzz + beta_u) {}".format(Kzz + beta_u))
    # KzzplusBeta = (Kzz + beta_u) + Iz * jitter
    # alpha_u = torch.linalg.solve(KzzplusBeta, Lambda_u[..., None])[..., 0]
    # alpha_u = torch.linalg.solve((Kzz + beta_u), Lambda_u[..., None])[..., 0]
    # print("alpha_u {}".format(alpha_u.shape))
    return alpha_u, beta_u, Lambda_u


def calc_alpha_u_from_lambda(
    beta_u: BetaInducing,
    Lambda_u: Lambda,
    Z: InducingPoints,
    kernel: NTK,
    jitter: float = 1e-3,
) -> AlphaInducing:
    Kzz = kernel(Z, Z)
    output_dim = Kzz.shape[0]
    Iz = (
        torch.eye(Kzz.shape[-1], dtype=torch.float64)
        .to(Z.device)[None, ...]
        .repeat(output_dim, 1, 1)
    )
    Kzz += Iz * jitter
    KzzplusBeta = (Kzz + beta_u) + Iz * jitter
    alpha_u = torch.linalg.solve(KzzplusBeta, Lambda_u[..., None])[..., 0]
    return alpha_u


def calc_lambdas(
    Y: OutputData, F: FuncData, likelihood: Likelihood
) -> Tuple[Lambda, Beta]:
    beta = calc_beta(F=F, likelihood=likelihood)
    Lambda = calc_lambda(Y=Y, F=F, likelihood=likelihood, beta=beta)
    return Lambda, beta


def calc_beta(F: FuncData, likelihood: Likelihood) -> Tuple[Lambda, Beta]:
    assert F.ndim == 2
    beta = 2 * likelihood.Hessian(f=F)
    return beta


def calc_lambda(
    Y: OutputData, F: FuncData, likelihood: Likelihood, beta: Beta
) -> Tuple[Lambda, Beta]:
    assert F.ndim == 2
    assert Y.shape[0] == F.shape[0]
    assert beta.ndim == 3
    nll = likelihood.nn_loss
    nll_jacobian_fn = jacrev(nll)
    alpha = -1 * nll_jacobian_fn(F, Y)  # [num_data, output_dim]
    beta_diag = torch.diagonal(beta, dim1=-2, dim2=-1)  # [num_data, output_dim]
    Lambda = alpha + F * beta_diag
    return Lambda


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
