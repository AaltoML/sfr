#!/usr/bin/env python3
import logging
from typing import List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import torch.nn as nn
from src.custom_types import (  # Lambda_1,; Lambda_2,
    AlphaInducing,
    Beta,
    BetaInducing,
    Data,
    FuncData,
    FuncMean,
    FuncVar,
    InducingPoints,
    InputData,
    Lambda,
    NTK,
    NTK_single,
    OutputData,
    OutputMean,
    OutputVar,
    TestInput,
)
from src.likelihoods import Likelihood
import src
from src.priors import Prior
from torch.func import functional_call, jacrev, jvp, vjp, vmap
from torch.utils.data import DataLoader, TensorDataset

from scipy.linalg import cho_factor, cho_solve
import numpy as np


class SFR(nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
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
        self.computed_Kss_Ksz = False

    def __call__(
        self,
        x: InputData,
        pred_type: str = "gp",  # "gp" or "nn"
        num_samples: int = 100,
    ):
        f_mean, f_var = self._predict_fn(x, full_cov=False)
        if pred_type in "nn":
            f_mean = self.network(x)
        if isinstance(self.likelihood, src.likelihoods.CategoricalLh):
            return self.likelihood(f_mean=f_mean, f_var=f_var, num_samples=num_samples)
        else:
            return self.likelihood(f_mean=f_mean, f_var=f_var)

    def fit(self, train_loader: DataLoader):
        all_train = DataLoader(
            train_loader.dataset, batch_size=len(train_loader.dataset)
        )
        train_data = next(iter(all_train))
        # train_data = train_loader.dataset
        # X = train_loader.dataset[0]
        if train_data[0].dtype == torch.float32:
            train_data[0].to(torch.float64)
        if train_data[1].dtype == torch.float32:
            train_data[1].to(torch.float64)
            # data = (X_train.to(torch.float64), y_train)
        self.set_data(train_data=train_data)

    @torch.no_grad()
    def set_data(self, train_data: Data):
        """Sets training data, samples inducing points, calcs dual parameters, builds predict fn"""
        X_train, Y_train = train_data
        X_train = torch.clone(X_train)
        Y_train = torch.clone(Y_train)
        assert X_train.shape[0] == Y_train.shape[0]
        self.train_data = (X_train, Y_train)
        num_data = Y_train.shape[0]
        indices = torch.randperm(num_data)[: self.num_inducing]
        self.Z = X_train[indices.to(X_train.device)].to(self.device)

        self.build_sfr_d()
        # self.build_sfr()
        

    @torch.no_grad()
    def build_sfr_d(self):
        logger.info("Calculating dual params and building prediction fn...")
        if isinstance(self.prior, src.priors.Gaussian):
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
            scaled=False
        )
        if self.dual_batch_size:
            self.alpha_u, self.beta_u, self.Lambda_u = calc_sparse_dual_params_batch(
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
        self._predict_fn_d = self.predict_from_sparse_duals_d(
            alpha_u=self.alpha_u,
            beta_u=self.beta_u,
            kernel=self.kernel,
            delta=delta,
            num_data=self.num_data,
            Z=self.Z,
            jitter=self.jitter,
        )
        logger.info("Finished building predict fn")


    @torch.no_grad()
    def build_sfr(self):
        logger.info("Calculating dual params and building prediction fn...")
        if isinstance(self.prior, src.priors.Gaussian):
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
            self.alpha_u, self.beta_u, self.Lambda_u = calc_sparse_dual_params_batch(
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
        f_mean = (Kxz @ self.alpha_u[..., None])[..., 0].T
        return f_mean

    @torch.no_grad()
    def predict(self, x: TestInput) -> Tuple[FuncMean, FuncVar]:
        f_mean, f_var = self._predict_fn(x, full_cov=False)
        return self.likelihood(f_mean=f_mean, f_var=f_var)

    @torch.no_grad()
    def predict_f(self, x: TestInput) -> Tuple[FuncMean, FuncVar]:
        f_mean, f_var = self._predict_fn(x, full_cov=False)
        return f_mean, f_var

    def loss(self, x: InputData, y: OutputData):
        f = self.network(x)
        neg_log_likelihood = self.likelihood.nn_loss(f=f, y=y)
        neg_log_prior = self.prior.nn_loss()
        return neg_log_likelihood + neg_log_prior

    @torch.no_grad()
    def update(self, x: InputData, y: OutputData):
        logger.info("Updating dual params...")
        if x.ndim > 2:
            x = torch.flatten(x, 1, -1)
        assert x.ndim == 2
        num_new_data, input_dim = x.shape
        Kzx = self.kernel(self.Z, x)

        if isinstance(self.likelihood, src.likelihoods.Gaussian):
            self.beta_u += (
                Kzx
                @ (1**-1 * torch.eye(num_new_data).to(self.Z)[None, ...])
                @ torch.transpose(Kzx, -1, -2)
            )
            self.Lambda_u += (Kzx @ y.T[..., None])[..., 0]
        elif isinstance(self.likelihood, src.likelihoods.CategoricalLh) or isinstance(
            self.likelihood, src.likelihoods.BernoulliLh
        ):
            f = self.network(x)
            Lambda_new, beta_new = calc_lambdas(Y=y, F=f, likelihood=self.likelihood)
            beta_new = torch.diagonal(beta_new, dim1=-2, dim2=-1)  # [N, F]
            self.beta_u += torch.einsum("fmn, nf, fun -> fmu", Kzx, beta_new, Kzx)
            self.Lambda_u += torch.einsum("fmn, nf -> fm", Kzx, Lambda_new)
        else:
            raise NotImplementedError

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
    
    @torch.no_grad()
    def predict_from_sparse_duals_d(
        self,
        alpha_u: AlphaInducing,
        beta_u: BetaInducing,
        kernel: NTK,
        delta: float,
        num_data: int,
        Z: InducingPoints,
        test_loader: DataLoader = None,
        full_cov: bool = False,
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
        KzzplusBeta = ( Kzz / (delta * num_data) + beta_u) + Iz * jitter
        
        if test_loader and self.computed_Kss_Ksz == False:
            Kxx_cached = []
            Kxz_cached = []
            for x in test_loader:
                Kxx = kernel(x, x, full_cov).detach().cpu()
                Kxz = kernel(x, Z).detach().cpu()
                Kxx_cached.append(Kxx)
                Kxz_cached.append(Kxz)
            self.computed_Kss_Ksz = True

        assert beta_u.shape == Kzz.shape

        K, M, _ = Kzz.shape
        Kzznp = Kzz.detach().cpu().numpy()
        KzzplusBetanp = KzzplusBeta.detach().cpu().numpy()
        L_Kzz = np.zeros_like(Kzznp)
        L_Bu = np.zeros_like(Kzznp)

        for k in range(K):
            L_Kzz[k], _ = cho_factor(Kzznp[k])
            L_Bu[k], _ = cho_factor(KzzplusBetanp[k])
        

        @torch.no_grad()
        def predict(x, index, full_cov: bool = False) -> Tuple[OutputMean, OutputVar]:

            if isinstance(x, torch.Tensor):
                Kxx = kernel(x, x, full_cov=full_cov).detach().cpu().numpy()
                Kxz = kernel(x, Z).detach().cpu().numpy()
            else:
                Kxx = Kxx_cached[index].numpy()
                Kxz = Kxz_cached[index].numpy()

            K, M, _ = Kzz.shape
            f_mean = (Kxz @ alpha_u[..., None])[..., 0].T

            if full_cov:
                raise NotImplementedError
            else:
                fvarnp = []
                for k in range(K):
                    Kxxk = Kxx[k]
                    Kxzk = Kxz[k]
                    Kzxk = Kxzk.T
                    # Note the first argument is a tuple that takes the result 
                    # from the cho_factor (by default lower=False, then (A, False) 
                    # is fed to cho_solve)
                    Amk = cho_solve((L_Kzz[k], False), Kzxk)
                    Abk = cho_solve((L_Bu[k], False), Kzxk)
                    fvark = (Kxxk - (Amk ** 2).sum()) / delta / num_data + (Abk ** 2).sum()
                    fvarnp.append(fvark)
                fvarnp = np.array(fvarnp)
                fvar = torch.from_numpy(fvarnp.T).to(f_mean.device)

            return f_mean, fvar

        return predict


@torch.no_grad()
def build_ntk(
    network: nn.Module, num_data: int, output_dim: int, delta: float = 1.0,
    scaled: bool = True
) -> Tuple[NTK, NTK_single]:
    network = network.eval()
    params = {k: v.detach() for k, v in network.named_parameters()}

    @torch.no_grad()
    def single_output_ntk_contraction(
        x1: InputData, x2: InputData, i: int, full_cov: Optional[bool] = True
    ):
        def fnet_single(params, x):
            f = functional_call(network, params, (x.unsqueeze(0),))[:, i]
            return f

        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1.values()]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
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
            if scaled:
                return 1 / (delta * num_data) * result[..., 0, 0]
            else:
                return result[..., 0, 0]
        else:
            result = torch.diagonal(result[..., 0], dim1=-1, dim2=-2)
            if scaled:
                return 1 / (delta * num_data) * result
            else:
                return result

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
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_X2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(
            output.numel(), dtype=output.dtype, device=output.device
        ).view(output.numel(), -1)
        return 1 / (delta * num_data) * vmap(get_ntk_slice)(basis)

    @torch.no_grad()
    def ntk(X1: InputData, X2: Optional[InputData], full_cov: bool = True):
        if X2 is None:
            X2 = X1
        if full_cov:
            K = torch.empty(output_dim, X1.shape[0], X2.shape[0]).to(X1.device)
        else:
            K = torch.empty(output_dim, X1.shape[0]).to(X1.device)
        for i in range(output_dim):
            K[i, ...] = single_output_ntk_contraction(X1, X2, i=i, full_cov=full_cov)
        return K

    return ntk, single_output_ntk_contraction


@torch.no_grad()
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

    K, M, _ = Kzz.shape
    Kzznp = Kzz.detach().cpu().numpy()
    KzzplusBetanp = KzzplusBeta.detach().cpu().numpy()
    Lmnp = np.zeros_like(Kzznp)
    Lbnp = np.zeros_like(KzzplusBetanp)


    for k in range(K):
        Lmnp[k], _ = cho_factor(Kzznp[k])
        Lbnp[k], _ = cho_factor(KzzplusBetanp[k])

    @torch.no_grad()
    def predict(x, full_cov: bool = False) -> Tuple[OutputMean, OutputVar]:
        Kxx = kernel(x, x, full_cov=full_cov)
        Kxz = kernel(x, Z)

        K, M, _ = Kzz.shape
        f_mean = (Kxz @ alpha_u[..., None])[..., 0].T

        if full_cov:
            raise NotImplementedError
        else:
            fvarnp = []
            for k in range(K):
                Kxxk = Kxx[k].detach().cpu().numpy()
                Kxzk = Kxz[k].detach().cpu().numpy()
                Kzxk = Kxzk.T
                # Note the first argument is a tuple that takes the result 
                # from the cho_factor (by default lower=False, then (A, False) 
                # is fed to cho_solve)
                Amk = cho_solve((Lmnp[k], False), Kzxk)
                Abk = cho_solve((Lbnp[k], False), Kzxk)
                fvark = Kxxk - (Amk ** 2).sum() + (Abk ** 2).sum()
                fvarnp.append(fvark)
            fvarnp = np.array(fvarnp)
            fvar = torch.from_numpy(fvarnp.T).to(f_mean.device)

        return f_mean, fvar

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
) -> Tuple[AlphaInducing, BetaInducing, Lambda]:
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

    alpha_u = torch.zeros((out_dim, Z.shape[0])).cpu()
    Lambda_u = torch.zeros((out_dim, Z.shape[0])).cpu()
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
        Lambda_u[output_c] = alpha_u[output_c]
        Kzz_c = (
            kernel(Z, Z, output_c).cpu() + torch.eye(Z.shape[0], device="cpu") * jitter
        )
        alpha_u[output_c] = torch.linalg.solve(
            (Kzz_c + beta_u[output_c]), alpha_u[output_c]
        )
    torch.cuda.empty_cache()
    return alpha_u.to(device), beta_u.to(device), Lambda_u.to(device)


@torch.no_grad()
def calc_sparse_dual_params(
    network: torch.nn.Module,
    train_data: Tuple[InputData, OutputData],
    Z: InducingPoints,
    kernel: NTK,
    likelihood: Likelihood,
    jitter: float = 1e-3,
) -> Tuple[AlphaInducing, BetaInducing, Lambda]:
    X, Y = train_data
    Kzx = kernel(Z, X)

    F = network(X)
    Lambda, beta = calc_lambdas(Y=Y, F=F, likelihood=likelihood)

    beta_diag = torch.diagonal(beta, dim1=-2, dim2=-1)  # [num_data, output_dim]
    beta = torch.diag_embed(beta_diag.T)  # [output_dim, num_data, num_data]
    beta_u = torch.matmul(torch.matmul(Kzx, beta), torch.transpose(Kzx, -1, -2))

    Lambda_u = torch.matmul(Kzx, torch.transpose(Lambda, -1, -2)[..., None])[..., 0]

    alpha_u = calc_alpha_u_from_lambda(
        beta_u=beta_u, Lambda_u=Lambda_u, Z=Z, kernel=kernel, jitter=jitter
    )
    return alpha_u, beta_u, Lambda_u


@torch.no_grad()
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


@torch.no_grad()
def calc_lambdas(
    Y: OutputData, F: FuncData, likelihood: Likelihood
) -> Tuple[Lambda, Beta]:
    beta = calc_beta(F=F, likelihood=likelihood)
    Lambda = calc_lambda(Y=Y, F=F, likelihood=likelihood, beta=beta)
    return Lambda, beta


@torch.no_grad()
def calc_beta(F: FuncData, likelihood: Likelihood) -> Tuple[Lambda, Beta]:
    assert F.ndim == 2
    beta = likelihood.Hessian(f=F)
    return beta


@torch.no_grad()
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
