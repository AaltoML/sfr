#!/usr/bin/env python3
import logging
from typing import List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src
import torch
import torch.distributions as dists
import torch.nn as nn
from scipy.linalg import cho_factor, cho_solve
from src.custom_types import (
    Alpha,
    AlphaInducing,
    Beta,
    BetaDiag,
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
from src.likelihoods import BernoulliLh, CategoricalLh, Likelihood
from src.priors import Prior
from torch.func import functional_call, jacrev, jvp, vjp, vmap
from torch.utils.data import DataLoader, TensorDataset


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
        # self.computed_Kss_Ksz = False

        if isinstance(self.prior, src.priors.Gaussian):
            self._prior_precision = self.prior.prior_precision
        else:
            raise NotImplementedError(
                "what should prior_precision be if not using Gaussian prior???"
            )

    def __call__(
        self,
        x: InputData,
        idx=None,
        pred_type: str = "gp",  # "gp" or "nn"
        num_samples: int = 100,
    ):
        f_mean, f_var = self.predict_f(x, full_cov=False)
        if pred_type in "nn":
            f_mean = self.network(x)
        if isinstance(self.likelihood, src.likelihoods.CategoricalLh):
            return self.likelihood(f_mean=f_mean, f_var=f_var, num_samples=num_samples)
        else:
            return self.likelihood(f_mean=f_mean, f_var=f_var)

    # def forward(self, x: InputData):
    #     return self.predict(x=x)

    # @torch.no_grad()
    # def predict(self, x: TestInput) -> Tuple[FuncMean, FuncVar]:
    #     f_mean, f_var = self.predict_f(x, full_cov=False)
    #     return self.likelihood(f_mean=f_mean, f_var=f_var)

    @torch.no_grad()
    def predict_f(
        self, x, full_cov: Optional[bool] = False
    ) -> Tuple[FuncMean, FuncVar]:
        Kxx = self.kernel(x, x, full_cov=full_cov).detach().cpu()
        Kxz = self.kernel(x, self.Z).detach().cpu()
        # print(f"Kxx {Kxx}")
        # print(f"Kxz {Kxz}")
        # alpha_u = alpha_u.detach().cpu()

        f_mean = (Kxz @ self.alpha_u[..., None])[..., 0].T / (
            self.prior_precision * self.num_data
        )
        # f_mean = (Kxz @ self.alpha_u[..., None])[..., 0].T
        # print(f"f_mean {f_mean}")

        if full_cov:
            raise NotImplementedError
            # TODO tmp could be computed before
            tmp = torch.linalg.solve(self.Kzz, self.Iz) - torch.linalg.solve(
                self.beta_u + self.Kzz, self.Iz
            )
            f_cov = Kxx - torch.matmul(
                torch.matmul(Kxz, tmp), torch.transpose(Kxz, -1, -2)
            )
            return f_mean, f_cov
        else:
            Kzx = torch.transpose(Kxz, -1, -2)
            # print(f"Kzx {Kzx}")
            Am = torch.linalg.solve_triangular(
                torch.transpose(self.Lm, -1, -2), Kzx, upper=False
            )
            # print(f"Am {Am}")
            Ab = torch.linalg.solve_triangular(
                torch.transpose(self.Lb, -1, -2), Kzx, upper=False
            )
            # print(f"Ab {Ab}")
            f_var = (
                Kxx - torch.sum(torch.square(Am), -2) + torch.sum(torch.square(Ab), -2)
            ) / (self.prior_precision * self.num_data)
            # print(f"f_var {f_var}")
            return f_mean.to(self.device), f_var.T.to(self.device)

    @torch.no_grad()
    def predict_mean(self, x: TestInput) -> FuncMean:
        x = x.to(self.Z.device)
        Kxz = self.kernel(x, self.Z)
        f_mean = (Kxz @ self.alpha_u[..., None])[..., 0].T / (
            self.prior_precision * self.num_data
        )
        return f_mean

    def fit(self, train_loader: DataLoader):
        all_train = DataLoader(
            train_loader.dataset, batch_size=len(train_loader.dataset)
        )
        train_data = next(iter(all_train))
        # train_data = train_loader.dataset
        # X = train_loader.dataset[0]
        if train_data[0].dtype == torch.float32:
            train_data[0] = train_data[0].to(torch.float64)
        if train_data[1].dtype == torch.float32:
            train_data[1] = train_data[1].to(torch.float64)
        if isinstance(self.likelihood, src.likelihoods.CategoricalLh):
            print("making outputs long type")
            # train_data[1].to(torch.long)
            train_data[1] = train_data[1].long()
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

        self._build_sfr()

    @torch.no_grad()
    def _build_sfr(self):
        # Build kernel
        self.kernel = build_ntk(
            network=self.network,
            num_data=self.num_data,
            output_dim=self.output_dim,
            prior_precision=self.prior_precision,
            scaled=False,
        )

        # Calculate dual parameters
        if self.dual_batch_size is None:
            self.dual_batch_size = self.num_data
        train_loader = DataLoader(
            TensorDataset(*(self.train_data)),
            batch_size=self.dual_batch_size,
            shuffle=False,
        )
        logger.info("Calculating dual params...")
        self.alpha, self.beta_diag, self.y_tilde = calc_dual_params(
            network=self.network,
            train_loader=train_loader,
            likelihood=self.likelihood,
            output_dim=self.output_dim,
            device=self.device,
        )
        logger.info("Finished calculating dual params")

        # Project dual parameters onto inducing points
        logger.info("Project dual params onto inducing points...")
        (
            self.alpha_u,
            self.beta_u,
            self.y_tilde_u,
        ) = project_dual_params_onto_inducing_points(
            Z=self.Z,
            kernel=self.kernel,
            train_loader=train_loader,
            # alpha=self.alpha,
            beta_diag=self.beta_diag,
            y_tilde=self.y_tilde,
            output_dim=self.output_dim,
            num_data=self.num_data,
            prior_precision=self.prior_precision,
            jitter=self.jitter,
            device=self.device,
        )
        self.alpha_u = self.alpha_u.detach().cpu()
        self.beta_u = self.beta_u.detach().cpu()
        self.y_tilde_u = self.y_tilde_u.detach().cpu()
        # print(f"beta_u {beta_u}")
        logger.info("Finished projecting dual params onto inducing points")

        # Calcualte kernel
        self.Kzz = self.kernel(self.Z, self.Z)
        print(f"Kzz {self.Kzz.shape}")
        num_inducing = self.Kzz.shape[-1]
        self.Iz = (
            torch.eye(num_inducing, dtype=torch.float64)
            .to(self.Z.device)[None, ...]
            .repeat(self.output_dim, 1, 1)
        )
        print(f"Iz {self.Iz.shape}")
        self.Kzz += self.Iz * self.jitter
        self.Kzz = self.Kzz.detach().cpu()
        print(f"Kzz {self.Kzz.shape}")
        # print(f"Kzz {Kzz}")
        print(f"Iz {self.Iz.device}")
        print(f"beta_u.device {self.beta_u.device}")
        print(f"Kzz.device {self.Kzz.device}")
        print(f"Iz.device {self.Iz.device}")

        assert self.beta_u.shape == self.Kzz.shape

        self.Iz = self.Iz.detach().cpu()
        print(f"Iz {self.Iz.device}")
        KzzplusBeta = (self.Kzz + self.beta_u) + self.Iz * self.jitter
        print(f"KzzplusBeta {KzzplusBeta.shape}")

        self.Lm = cholesky_add_jitter_until_psd(self.Kzz, jitter=self.jitter)
        print(f"Lm.device {self.Lm.device}")
        self.Lb = cholesky_add_jitter_until_psd(KzzplusBeta, jitter=self.jitter)
        print(f"Lb.device {self.Lb.device}")

    def loss(self, x: InputData, y: OutputData):
        f = self.network(x)
        neg_log_likelihood = self.likelihood.nn_loss(f=f, y=y)
        neg_log_prior = self.prior.nn_loss()
        return neg_log_likelihood + neg_log_prior

    @torch.no_grad()
    def update(
        self, data_loader: DataLoader = None, x: InputData = None, y: OutputData = None
    ):
        if data_loader is None:
            if x is None or y is None:
                raise NotImplementedError
            else:
                data_loader = DataLoader(
                    TensorDataset(*(x, y)),
                    batch_size=self.dual_batch_size,
                    shuffle=False,
                )
        logger.info("Updating dual params...")

        alpha_new, beta_diag_new, y_tilde_new = calc_dual_params(
            network=self.network,
            train_loader=data_loader,
            likelihood=self.likelihood,
            output_dim=self.output_dim,
            device=self.device,
        )
        logger.info("Finished calculating new dual params")

        # Project dual parameters onto inducing points
        logger.info("Project new dual params onto inducing points...")
        (
            alpha_u_new,
            beta_u_new,
            y_tilde_u_new,
        ) = project_dual_params_onto_inducing_points(
            Z=self.Z,
            kernel=self.kernel,
            train_loader=data_loader,
            beta_diag=beta_diag_new,
            y_tilde=y_tilde_new,
            output_dim=self.output_dim,
            num_data=self.num_data,
            prior_precision=self.prior_precision,
            jitter=self.jitter,
            device=self.device,
        )
        # print(f"beta_u {beta_u}")
        logger.info("Finished projecting new dual params onto inducing points")

        # beta_u_new = beta_u_new * num_new_data

        logger.info("Adding new and old dual params ")
        print(f"beta_u_new {beta_u_new.device}")
        print(f"y_tilde_u_new {y_tilde_u_new.device}")
        print(f"self.beta_u {self.beta_u.device}")
        print(f"self.y_tilde_u {self.y_tilde_u.device}")
        self.beta_u += beta_u_new.detach().cpu()
        self.y_tilde_u += y_tilde_u_new.detach().cpu()
        logger.info("Finished adding new and old dual params")

        self.alpha_u = calc_alpha_u(
            self.Kzz,
            beta_u=self.beta_u,
            y_tilde_u=self.y_tilde_u,
            output_dim=self.output_dim,
            jitter=self.jitter,
        )
        print(f"alpha_u {self.alpha_u.device}")
        self.alpha_u = self.alpha_u.detach().cpu()

        logger.info("Caching tensors for faster predictions...")
        KzzplusBeta = (self.Kzz + self.beta_u) + self.Iz * self.jitter
        print(f"KzzplusBeta {KzzplusBeta.device}")
        self.Lb = cholesky_add_jitter_until_psd(KzzplusBeta, jitter=self.jitter)
        logger.info("Finished caching tensors for faster predictions")
        print(f"Lb {self.Lb.device}")

    def optimize_prior_precision(
        self,
        pred_type,  # "nn" or "gp"
        method="grid",  # "grid" or "bo"
        val_loader: DataLoader = None,
        n_samples: int = 100,
        prior_prec_min: float = 1e-8,
        prior_prec_max: float = 1.0,
        num_trials: int = 20,
    ):
        prior_prec_before = self.prior_precision
        logger.info(f"prior_prec_before {prior_prec_before}")
        nll_before = self.nlpd(
            data_loader=val_loader,
            pred_type=pred_type,
            n_samples=n_samples,
            # prior_prec=prior_prec,
        )
        logger.info(f"nll_before {nll_before}")

        if method == "grid":
            log_prior_prec_min = np.log(prior_prec_min)
            log_prior_prec_max = np.log(prior_prec_max)
            interval = torch.logspace(
                log_prior_prec_min, log_prior_prec_max, num_trials
            )
            prior_precs, nlls = [], []
            for prior_prec in interval:
                prior_prec = prior_prec.item()
                # self.update_pred_fn(prior_prec)
                self.prior_precision = prior_prec
                nll = self.nlpd(
                    data_loader=val_loader,
                    pred_type=pred_type,
                    n_samples=n_samples,
                    prior_prec=prior_prec,
                )
                nll = nll.detach().numpy()
                logger.info(f"Prior prec {prior_prec} nll: {nll}")
                nlls.append(nll)
                prior_precs.append(prior_prec)
            best_nll = np.min(nlls)
            best_prior_prec = prior_precs[np.argmin(nlls)]
        elif method == "bo":
            from ax.service.managed_loop import optimize

            def nlpd_objective(params):
                nll = self.nlpd(
                    data_loader=val_loader,
                    pred_type=pred_type,
                    n_samples=n_samples,
                    prior_prec=params["prior_prec"],
                )
                return nll.detach().numpy()

            best_parameters, values, experiment, model = optimize(
                parameters=[
                    {
                        "name": "prior_prec",
                        "type": "range",
                        "bounds": [prior_prec_min, prior_prec_max],
                        "log_scale": False,
                    },
                ],
                evaluation_function=nlpd_objective,
                objective_name="NLPD",
                minimize=True,
                total_trials=num_trials,
            )
            best_prior_prec = best_parameters["prior_prec"]
            best_nll = values[0]["NLPD"]
        else:
            raise NotImplementedError

        # If worse than original then reset
        if nll_before < best_nll:
            best_nll = nll_before
            best_prior_prec = prior_prec_before

        for x, y in val_loader:
            # TODO this is just here for debugging
            f_mean, f_var = self.predict_f(x.to(self.device), full_cov=False)
            logger.info(f"f_var after BO=: {f_var}")
            break
        logger.info(f"Best prior prec {best_prior_prec} with nll: {best_nll}")
        # self.update_pred_fn(best_prior_prec)
        self.prior_precision = best_prior_prec

    def nlpd(
        self,
        data_loader: DataLoader,
        pred_type: str = "gp",
        n_samples: int = 100,
        prior_prec: Optional[float] = None,
    ):
        if prior_prec:
            self.prior_precision = prior_prec
            # self.update_pred_fn(prior_prec)
        try:
            py, targets = [], []
            for idx, (x, y) in enumerate(data_loader):
                p = self(
                    x=x.to(self.device),
                    idx=idx,
                    pred_type=pred_type,
                    num_samples=n_samples,
                )[0]
                py.append(p)
                targets.append(y.to(self.device))
            targets = torch.cat(targets, dim=0).cpu().numpy()
            probs = torch.cat(py).cpu().numpy()

            if isinstance(self.likelihood, BernoulliLh):
                dist = dists.Bernoulli(torch.Tensor(probs[:, 0]))
            elif isinstance(self.likelihood, CategoricalLh):
                dist = dists.Categorical(torch.Tensor(probs))
            else:
                raise NotImplementedError
            nll = -dist.log_prob(torch.Tensor(targets)).mean().numpy()
        except RuntimeError:
            nll = torch.inf
        return nll

    @property
    def prior_precision(self):
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision):
        old_prior_precision = self._prior_precision
        self._prior_precision = prior_precision
        self.prior.prior_precision = prior_precision
        # Rebuild dual params with new prior precision
        # TODO probably needs to have if self.beta_u exists
        if self.beta_u is not None:
            self.beta_u = self.beta_u * old_prior_precision / prior_precision
            KzzplusBeta = (self.Kzz + self.beta_u) + self.Iz * self.jitter
            self.Lb = cholesky_add_jitter_until_psd(KzzplusBeta, jitter=self.jitter)
            self.alpha_u = calc_alpha_u(
                self.Kzz,
                beta_u=self.beta_u,
                y_tilde_u=self.y_tilde_u,
                output_dim=self.output_dim,
                jitter=self.jitter,
            )

    @property
    def num_data(self):
        return self.train_data[0].shape[0]


def project_dual_params_onto_inducing_points(
    Z,
    kernel: NTK,
    train_loader: DataLoader,
    # alpha: Alpha,
    beta_diag: BetaDiag,
    y_tilde: Lambda,
    output_dim: int,
    num_data: int,
    prior_precision: float,
    jitter: float = 1e-3,
    device: str = "cpu",
):
    num_inducing = Z.shape[0]
    alpha_u = torch.zeros((output_dim, num_inducing)).cpu()
    y_tilde_u = torch.zeros((output_dim, num_inducing)).cpu()
    beta_u = torch.zeros((output_dim, num_inducing, num_inducing)).cpu()

    for output_c in range(output_dim):
        start_idx, end_idx = 0, 0
        logging.info(f"Computing covariance for output dim {output_c}")
        for batch in train_loader:
            x_i, y_i = batch[0], batch[1]
            x_i, y_i = x_i.to(device), y_i.to(device)
            batch_size = x_i.shape[0]
            end_idx = start_idx + batch_size
            Kui_c = kernel(Z, x_i, index=output_c).cpu()
            print(f"Kui_c {Kui_c.shape}")

            # alpha_batch = alpha[start_idx:end_idx, output_c]
            # print(f"alpha_batch {alpha_batch.shape}")
            y_tilde_batch = y_tilde[start_idx:end_idx, output_c]
            print(f"y_tilde_batch {y_tilde_batch.shape}")
            beta_diag_batch = beta_diag[start_idx:end_idx, output_c]
            print(f"beta_diag_batch {beta_diag_batch.shape}")
            y_tilde_u_batch = torch.einsum("mb, b -> m", Kui_c, y_tilde_batch)
            print(f"y_tilde_u_batch {y_tilde_u_batch.shape}")
            beta_batch = torch.einsum("mb, b, nb -> mn", Kui_c, beta_diag_batch, Kui_c)
            print(f"beta_batch {beta_batch.shape}")

            # alpha_u_batch = calc_alpha_u(Kui_c, alpha=alpha_batch)

            y_tilde_u[output_c] += y_tilde_u_batch.cpu()
            # alpha_u[output_c] += alpha_u_batch.cpu()
            beta_u[output_c] += beta_batch.cpu() / (prior_precision * num_data)

            start_idx = end_idx
            del Kui_c
        torch.cuda.empty_cache()
        Kzz_c = (
            kernel(Z, Z, index=output_c).cpu()
            + torch.eye(num_inducing, device="cpu") * jitter
        )

        torch.cuda.empty_cache()
        # beta_u = beta_u / (prior_precision * num_data)
        alpha_u[output_c] = torch.linalg.solve(
            (Kzz_c + beta_u[output_c]), y_tilde_u[output_c]
        )
        # alpha_u[output_c] = torch.linalg.solve(Kzz_c, alpha_u[output_c])
    torch.cuda.empty_cache()
    return alpha_u.to(device), beta_u.to(device), y_tilde_u.to(device)


def calc_dual_params(
    network: nn.Module,
    likelihood: Likelihood,
    train_loader: DataLoader,
    output_dim: int,
    device: str = "cpu",
):
    num_data = len(train_loader.dataset)
    items_shape = (num_data, output_dim)

    # rename lambda_1 is Lambda, lamba2 is beta
    y_tilde = torch.zeros(items_shape).cpu()
    beta_diag = torch.zeros(items_shape).cpu()
    alpha = torch.zeros(items_shape).cpu()

    # Calculate dual params at data. Actually calc beta_diag and y_tilde
    start_idx, end_idx = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        f = network(x)
        if f.ndim == 1:
            f = f.unsqueeze(-1)
        beta_batch = calc_beta(likelihood=likelihood, F=f)
        print(f"beta_batch {beta_batch.shape}")
        alpha_batch = calc_alpha(likelihood=likelihood, Y=y, F=f)
        print(f"alpha_batch {alpha_batch.shape}")
        y_tilde_batch = calc_y_tilde(F=f, alpha=alpha_batch, beta=beta_batch)
        # y_tilde_batch = y - y_tilde_batch
        print(f"y_tilde_batch {y_tilde_batch.shape}")

        beta_diag_batch = torch.vmap(torch.diag)(beta_batch)
        print(f"beta_diag_batch {beta_diag_batch.shape}")

        end_idx = start_idx + batch_size
        y_tilde[start_idx:end_idx] = y_tilde_batch
        beta_diag[start_idx:end_idx] = beta_diag_batch
        alpha[start_idx:end_idx] = alpha_batch
        start_idx = end_idx
    return alpha, beta_diag, y_tilde


@torch.no_grad()
def calc_y_tilde_u(Kzx, y_tilde: Lambda):
    return torch.matmul(Kzx, torch.transpose(y_tilde, -1, -2)[..., None])[..., 0]


@torch.no_grad()
def calc_alpha_u(
    Kzz, beta_u: BetaInducing, y_tilde_u: Lambda, output_dim: int, jitter: float = 1e-3
) -> AlphaInducing:
    Iz = (
        torch.eye(Kzz.shape[-1], dtype=torch.float64)
        .to(Kzz.device)[None, ...]
        .repeat(output_dim, 1, 1)
    )
    KzzplusBeta = (Kzz + beta_u) + Iz * jitter
    alpha_u = torch.linalg.solve(KzzplusBeta, y_tilde_u[..., None])[..., 0]
    # alpha_u = torch.linalg.solve((Kzz + beta_u), y_tilde_u)
    return alpha_u


@torch.no_grad()
def calc_beta_u(kernel: NTK, Z, X: InputData, beta_diag: Beta) -> BetaInducing:
    Kzi = kernel(Z, X)
    return torch.einsum("mb, b, nb -> mn", Kzi, beta_diag, Kzi)


@torch.no_grad()
def calc_alpha(likelihood: Likelihood, Y: OutputData, F: FuncData) -> Alpha:
    assert F.ndim == 2
    assert Y.shape[0] == F.shape[0]
    # nll = likelihood.nn_loss
    # nll_jacobian_fn = jacrev(nll)
    # return -1 * nll_jacobian_fn(F, Y)  # [num_data, output_dim]
    # TODO put this back to using Jacobian
    return likelihood.residual(f=F, y=Y)  # [num_data, output_dim]


@torch.no_grad()
def calc_beta(likelihood: Likelihood, F: FuncData) -> Beta:
    assert F.ndim == 2
    return likelihood.Hessian(f=F)


@torch.no_grad()
def calc_y_tilde(F: FuncData, alpha: Alpha, beta: Beta) -> Lambda:
    beta_diag = torch.diagonal(beta, dim1=-2, dim2=-1)  # [num_data, output_dim]
    return alpha + F * beta_diag


@torch.no_grad()
def build_ntk(
    network: nn.Module,
    num_data: int,
    output_dim: int,
    prior_precision: float = 1.0,
    scaled: bool = True,
) -> NTK:
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
                return 1 / (prior_precision * num_data) * result[..., 0, 0]
            else:
                return result[..., 0, 0]
        else:
            result = torch.diagonal(result[..., 0], dim1=-1, dim2=-2)
            if scaled:
                return 1 / (prior_precision * num_data) * result
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
        return 1 / (prior_precision * num_data) * vmap(get_ntk_slice)(basis)

    @torch.no_grad()
    def ntk(X1: InputData, X2: Optional[InputData], full_cov: Optional[bool] = True):
        if X2 is None:
            X2 = X1
        if full_cov:
            K = torch.empty(output_dim, X1.shape[0], X2.shape[0]).to(X1.device)
        else:
            K = torch.empty(output_dim, X1.shape[0]).to(X1.device)
        for i in range(output_dim):
            K[i, ...] = single_output_ntk_contraction(X1, X2, i=i, full_cov=full_cov)
        return K

    @torch.no_grad()
    def kernel(
        X1: InputData,
        X2: Optional[InputData],
        full_cov: Optional[bool] = True,
        index: Optional[int] = None,
    ):
        if index is not None:
            return single_output_ntk_contraction(
                x1=X1, x2=X2, i=index, full_cov=full_cov
            )
        else:
            return ntk(X1=X1, X2=X2, full_cov=full_cov)

    return kernel


def cholesky_add_jitter_until_psd(x, jitter: float = 1e-5, jitter_factor=4):
    try:
        L = torch.linalg.cholesky(x, upper=True)
        return L
    except:
        logger.info(f"Cholesky failed so adding more jitter={jitter}")
        Iz = torch.eye(x.shape[-1]).to(x.device)
        jitter = jitter_factor * jitter
        x += Iz * jitter
        return cholesky_add_jitter_until_psd(x, jitter=jitter)
