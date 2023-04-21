#!/usr/bin/env python3
import logging
from typing import Optional, Union
from copy import deepcopy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from gpytorch.lazy import DiagLazyTensor, CholLazyTensor, TriangularLazyTensor
from torchtyping import TensorType
import gpytorch
import torch
import torch.distributions as td
import wandb
from functorch import jacrev
from gpytorch.module import _validate_module_outputs
from src.custom_types import Prediction
from src.utils import EarlyStopper
from torch.utils.data import DataLoader
import numpy as np


class SVGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        learn_inducing_locations: bool = True,
        jitter: float = 0.0,
        device="cuda",
    ):
        self.learn_inducing_locations = learn_inducing_locations
        if isinstance(covar_module, gpytorch.kernels.MultitaskKernel):
            self.out_size = covar_module.num_tasks
            self.is_multi_output = True
            print("out_size: {}".format(self.out_size))
            print("inducing_points {}".format(inducing_points.shape))
            assert inducing_points.ndim == 3
            assert (
                inducing_points.shape[0] == self.out_size
                or inducing_points.shape[0] == 1
            )
        elif (
            isinstance(covar_module, gpytorch.kernels.Kernel)
            and len(covar_module.batch_shape) > 0
        ):
            self.out_size = covar_module.batch_shape[0]
            self.is_multi_output = True
            print("out_size yo: {}".format(self.out_size))
            print("inducing_points yo {}".format(inducing_points.shape))
            assert inducing_points.ndim == 3
            assert (
                inducing_points.shape[0] == self.out_size
                or inducing_points.shape[0] == 1
            )
            num_inducing = inducing_points.shape[-2]
        elif isinstance(covar_module, gpytorch.kernels.Kernel):
            self.out_size = 1
            self.is_multi_output = False
            assert inducing_points.ndim == 2
            num_inducing = inducing_points.shape[0]
            print("out_size yo yo you: {}".format(self.out_size))
            print("inducing_points yo yo you {}".format(inducing_points.shape))
        else:
            raise NotImplementedError(
                "covar_module should be an instance of gpytorch.kernels.Kernel"
            )

        task_idxs = np.arange(self.out_size).reshape(-1, 1)
        self.task_indices = torch.Tensor(task_idxs).to(torch.int64).to(device)
        # self.task_indices_i = torch.Tensor([task_indices_i], device=X.device).to(
        #     torch.long
        # )
        # out_size = inducing_points.shape[0]
        # inducing_points = torch.rand(out_size, num_inducing, in_size)
        # print("out_size: {}".format(out_size))
        if self.out_size > 1:
            # Learn a variational distribution for each output dim
            variational_distribution = (
                gpytorch.variational.CholeskyVariationalDistribution(
                    num_inducing_points=num_inducing,
                    batch_shape=torch.Size([self.out_size]),
                )
            )
            variational_strategy = (
                gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    gpytorch.variational.VariationalStrategy(
                        self,
                        inducing_points,
                        variational_distribution,
                        learn_inducing_locations=learn_inducing_locations,
                    ),
                    num_tasks=self.out_size,
                )
            )
        else:
            variational_distribution = (
                gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
            )
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x, data_new: Optional = None):
        if data_new is None:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
        else:
            raise NotImplementedError("# TODO Paul implement fast update here")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def predict(
    svgp: SVGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    # jitter: float = 0.0,
    # jitter: float = 1e-5,
    jitter: float = 1e-9,
    # learning_rate: float = 0.8,
    # learning_rate: float = 1.0,
):
    @torch.no_grad()
    def predict_fn(x, data_new: Optional = None, full_cov: bool = False) -> Prediction:
        # if svgp.is_multi_output:
        #     print("out_size>0")
        # else:
        #     print("out_size=0")
        white: bool = False
        svgp.eval()
        likelihood.eval()
        if data_new != None:
            X, Y = data_new
            assert Y.ndim == 1 or Y.ndim == 2

            # # learning_rate = 1.0
            # if svgp.is_multi_output:
            #     # vmap(predict_fn)
            #     return predict_fn
            # else:
            #     return predict_fn

            # TODO how to handle diff Z on each output_dim??
            # var_dist = svgp.variational_strategy.variational_distribution
            if svgp.is_multi_output:
                # print("is mo")
                Z = svgp.variational_strategy.base_variational_strategy.inducing_points
                # if Z.ndim == 3 and Z.shape[0] > 1:
                #     raise NotImplementedError(
                #         "Currently we only support Z.shape=(1,num_inducing,input_dim)"
                #     )
                # elif Z.ndim == 3 and Z.shape[0] == 1:
                #     Z = Z[0, ...]
                # var_dist = (
                #     svgp.variational_strategy.base_variational_strategy.variational_distribution
                # )
            else:
                Z = svgp.variational_strategy.inducing_points
                # var_dist = svgp.variational_strategy.variational_distribution
            num_inducing = Z.size(-2)
            # print("var_dist.mean {}".format(var_dist.mean.shape))
            # print("var_dist.cov {}".format(var_dist.lazy_covariance_matrix.shape))

            # GPyTorch's way of computing Kuf:
            if Z.ndim == 3:
                X = X[None, ...].tile((Z.shape[0], 1, 1))
            # print("Z: {}".format(Z.shape))
            # print("X: {}".format(X.shape))
            # X = X[0, ...]
            # Z = Z[0, ...]
            full_inputs = torch.cat([Z, X], dim=-2)

            # else:
            #     full_inputs = torch.cat([Z, X], dim=-2)
            full_covar = svgp.covar_module(full_inputs)
            Kuu = full_covar[..., :num_inducing, :num_inducing].add_jitter()
            Kuf = full_covar[..., :num_inducing, num_inducing:].evaluate()
            if not svgp.is_multi_output:
                full_covar = full_covar[None, ...]
                Kuu = Kuu.unsqueeze(0)
                Kuf = Kuf[None, ...]
            # print("full_inputs {}".format(full_inputs.shape))
            # print("Kuu {}".format(Kuu.shape))
            # print("Kuf {}".format(Kuf.shape))
            # print("full_covar {}".format(full_covar.shape))

            # task_indices = torch.LongTensor([0])
            # f1 = svgp(Z, task_indices=task_indices)
            # print("f1 {}".format(f1))
            # task_indices = torch.LongTensor([1])
            # f2 = svgp(Z, task_indices=task_indices)
            # print("f {}".format(f2))
            # print("diff {}".format(f1.mean - f2.mean))
            # print("diff {}".format(f1.mean - f1.mean))

            # cov = f.to_data_independent_dist().covariance_matrix
            # cov = CholLazyTensor(torch.linalg.cholesky(deepcopy(f.covariance_matrix)))
            # if svgp.is_multi_output:
            #     f = svgp(Z)
            #     mean = f.mean
            #     covs = []
            #     print("f.covariance_matrix {}".format(f.covariance_matrix.shape))
            #     # cov = f.covariance_matrix
            #     for i in range(svgp.out_size):
            #         print("i {}".format(i))
            #         covs.append(
            #             f.covariance_matrix[
            #                 i * num_inducing : (i + 1) * num_inducing,
            #                 i * num_inducing : (i + 1) * num_inducing,
            #             ]
            #         )
            #     cov = torch.stack(covs, 0)
            #     cov = CholLazyTensor(cov)
            # cov = torch.linalg.cholesky(cov)
            # cov = LazyTensor(torch.linalg.cholesky(cov))
            # cov = CholLazyTensor(torch.linalg.cholesky(cov))

            if svgp.is_multi_output:

                def single_gp(task_indices):
                    # svgp.eval()
                    f = svgp(Z, task_indices=task_indices)
                    print("f: {}".format(f))
                    mean = f.mean
                    print("m: {}".format(mean.shape))
                    cov = f.covariance_matrix
                    print("c: {}".format(cov.shape))
                    return mean, cov
                    # return f.mean, f.covariance_matrix

                # task_indices = torch.LongTensor([[0], [1]])
                # print("task_indices.shape")
                # print(task_indices.shape)
                # print(Z.shape)
                means, covs = [], []
                # print("svgp.out_size {}".format(svgp.out_size))
                #
                # print("task_indices {}".format(task_indices.shape))
                # task_indices = torch.Tensor([task_indices_i], device=X.device).to(
                # for task_indices_i in range(svgp.out_size):
                for task_indices_i in svgp.task_indices:
                    # task_indices_i = torch.LongTensor([task_indices_i], device=X.device)
                    # task_indices_i = torch.LongTensor([task_indices_i], device=X.device)
                    # task_indices_i = torch.Tensor([task_indices_i], device=X.device).to(
                    #     torch.long
                    # )
                    mean, cov = single_gp(task_indices_i)
                    # mean, cov = single_gp(-1)
                    means.append(mean)
                    covs.append(cov)

                # task_indices = torch.LongTensor(0, device=X.device)
                # mean, cov = svgp(Z, task_indices=task_indices)
                # f = svgp(Z)
                # mean = f.mean
                # cov = f.variance
                # cov = f.covariance_matrix
                # print("mean {}".format(mean.shape))
                # print("cov you {}".format(cov.shape))
                # means.append(mean)
                # covs.append(cov)
                mean = torch.stack(means, -1)
                # mean = torch.stack(means, 0)
                cov = torch.stack(covs, 0)

            else:
                # print("single svgp.out_size {}".format(svgp.out_size))
                f = svgp(Z)
                cov = f.covariance_matrix[None, ...]  # [1, M, M]
                mean = f.mean[..., None]  # [M, 1]
            print("mean {}".format(mean.shape))
            print("cov you {}".format(cov.shape))
            cov = CholLazyTensor(torch.linalg.cholesky(deepcopy(cov)))

            # lambda_1, lambda_2 = mean_cov_to_natural_param(
            #     var_dist.mean, var_dist.lazy_covariance_matrix, Kuu
            # )
            lambda_1, lambda_2 = mean_cov_to_natural_param(mean, cov, Kuu)
            # print("lambda_1 {}".format(lambda_1.shape))
            # print("lambda_2 {}".format(lambda_2.shape))

            # new_mean, new_cov = conditional_from_precision_sites_white_full(
            #     Kuu, lambda_1, lambda_2, jitter=jitter
            # )

            # online_update
            with torch.no_grad():
                pred = svgp(X)
                mean = pred.mean  # [num_new]
                var = pred.variance  # [num_new]
            if not svgp.is_multi_output:
                mean = mean[..., None]
                var = var[..., None]
            # print("mean 1 {}".format(mean.shape))
            # print("var 1 {}".format(var.shape))
            # print("Y {}".format(Y.shape))
            # if Y.ndim == 1:
            # Y = Y[..., None]
            # print("Y {}".format(Y.shape))

            def predict_ve(mean, var):
                # print("mean you {}".format(mean.shape))
                # print("var you {}".format(var.shape))
                # f_dist_b = gpytorch.distributions.MultivariateNormal(
                #     mean.T, torch.diag_embed(var.T)
                # )  # Mean: num_new x output_dim Cov: num_new x output_dim x output_dim

                if svgp.is_multi_output:
                    f_dist_b = gpytorch.distributions.MultivariateNormal(
                        mean.T, torch.diag_embed(var.T)
                    )  # Mean: num_new x output_dim Cov: num_new x output_dim x output_dim
                    # print("f_dist_b {}".format(f_dist_b))
                    f_dist = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                        f_dist_b
                    )
                    ve_terms = likelihood.expected_log_prob(
                        Y, f_dist
                    )  # TODO: Is this right?
                else:
                    f_dist_b = gpytorch.distributions.MultivariateNormal(
                        mean[..., 0], torch.diag_embed(var[..., 0])
                    )  # Mean: num_new x output_dim Cov: num_new x output_dim x output_dim
                    # print("f_dist_b {}".format(f_dist_b))
                    ve_terms = likelihood.expected_log_prob(
                        Y, f_dist_b
                    )  # TODO: Is this right?
                # print("ve_terms {}".format(ve_terms.shape))
                # ve = ve_terms
                ve = (
                    ve_terms.sum()
                )  # TODO: CHECK: divide by num_data ? but only one point at a time so probably fine
                # print("ve {}".format(ve.shape))
                return ve

            jac_fn_mean = jacrev(predict_ve, argnums=0)
            jac_fn_var = jacrev(predict_ve, argnums=1)
            d_exp_dm = jac_fn_mean(mean, var)  # [num_new, output_dim]
            d_exp_dv = jac_fn_var(mean, var)  # [num_new, output_dim]
            # print("d_exp_dm {}".format(d_exp_dm.shape))
            # print("d_exp_dv {}".format(d_exp_dv.shape))

            eps = 1e-8
            d_exp_dv.clamp_(max=-eps)

            grad_nat_1 = d_exp_dm - 2.0 * (d_exp_dv * mean)
            grad_nat_2 = d_exp_dv
            # print("grad_nat_1 {}".format(grad_nat_1.shape))
            # print("grad_nat_2 {}".format(grad_nat_2.shape))

            grad_mu_1 = torch.einsum("bmc, cb -> mb", Kuf, grad_nat_1)
            grad_mu_2 = torch.einsum("bmc, cb, bnc -> bmn", Kuf, grad_nat_2, Kuf)
            # if svgp.is_multi_output:
            #     grad_mu_1 = torch.einsum("bmc, cb -> mb", Kuf, grad_nat_1)
            #     grad_mu_2 = torch.einsum("bmc, cb, bnc -> bmn", Kuf, grad_nat_2, Kuf)
            #     # grad_mu_2 = torch.einsum("nmf, nf, ncf -> nmc", K_uf, grad_nat_2, K_uf)
            # else:
            #     grad_mu_1 = torch.einsum("mc, c -> m", Kuf, grad_nat_1)
            #     grad_mu_2 = torch.einsum("mc, c, nc -> mn", Kuf, grad_nat_2, Kuf)
            # grad_mu_1 = Kuf.matmul(grad_nat_1)
            # grad_mu_2 = Kuf @ torch.diag_embed(grad_nat_2.T) @ Kuf.T
            # print("grad_mu_1 {}".format(grad_mu_1.shape))
            # print("grad_mu_2 {}".format(grad_mu_2.shape))

            # lambda_1_t_new = grad_mu_1[..., None]
            lambda_1_t_new = grad_mu_1
            lambda_2_t_new = (-2) * grad_mu_2
            # print("lambda_1_t_new {}".format(lambda_1_t_new.shape))
            # print("lambda_2_t_new {}".format(lambda_2_t_new.shape))

            # lambda_1_new = lambda_1 + lambda_1_t_new
            # lambda_2_new = lambda_2 + lambda_2_t_new
            lambda_1 = lambda_1 + lambda_1_t_new
            lambda_2 = lambda_2 + lambda_2_t_new
            # print("lambda_1_new {}".format(lambda_1_new.shape))
            # print("lambda_2_new {}".format(lambda_2_new.shape))

            new_mean, new_cov = conditional_from_precision_sites_white_full(
                Kuu,
                lambda_1,
                lambda_2,
                jitter=jitter
                # Kuu, lambda_1_new, lambda_2_new, jitter=jitter
            )
            # print("new_mean {}".format(new_mean.shape))
            # print("new_cov {}".format(new_cov.shape))
            # new_mean = new_mean[..., 0]
            # print("new_mean {}".format(new_mean.shape))

            if full_cov:
                Knn = svgp.covar_module(x, x).evaluate()
            else:
                Knn = svgp.covar_module(x, diag=True)
            # Knn += torch.eye(Knn.shape[-1]) * jitter
            # print("Knn {}".format(Knn.shape))
            Kmn = svgp.covar_module(Z, x).evaluate()
            # print("Kmn {}".format(Kmn.shape))
            Kmm = svgp.covar_module(Z).evaluate()
            Kmm += torch.eye(Kmm.shape[-1], device=Kmm.device) * jitter
            # print("Kmm {}".format(Kmm.shape))
            Lm = torch.linalg.cholesky(Kmm, upper=False)
            # print("Lm {}".format(Lm.shape))
            q_sqrt = torch.linalg.cholesky(new_cov, upper=False)

            if svgp.is_multi_output:
                f_means, f_vars = [], []
                for kmn, lm, knn, new_mean_, q_sqrt_ in zip(
                    Kmn, Lm, Knn, new_mean, q_sqrt
                ):
                    f_mean, f_var = base_conditional_with_lm(
                        Kmn=kmn,
                        Lm=lm,
                        Knn=knn,
                        f=new_mean_,
                        full_cov=full_cov,
                        q_sqrt=q_sqrt_,
                        white=white,
                    )
                    f_means.append(f_mean)
                    f_vars.append(f_var)
                f_mean = torch.stack(f_means, 0)
                f_var = torch.stack(f_vars, 0)
            else:
                f_mean, f_var = base_conditional_with_lm(
                    Kmn=Kmn,
                    Lm=Lm,
                    Knn=Knn,
                    f=new_mean[0, ...],
                    full_cov=full_cov,
                    q_sqrt=q_sqrt[0, ...],
                    white=white,
                )
            # print("f_mean {}".format(f_mean.shape))
            # print("f_var {}".format(f_var.shape))

            # TODO implement likelihood to get noise_var
            return f_mean.T, f_var.T, 0

        else:
            f = svgp(x)

        # print("x {}".format(x.shape))
        # print("f {}".format(f))
        # print("before fail")
        # print(x.shape)
        # f = svgp(x)
        # print(f)
        output = likelihood(f)
        noise_var = output.variance - f.variance
        return f.mean, f.variance, noise_var

    return predict_fn

    # def predict_fn(x, data_new: Optional = None) -> Prediction:
    #     print("out_size>0")
    #     svgp.eval()
    #     likelihood.eval()
    #     if data_new != None:
    #         X, Y = data_new
    #         # print("data new X={}, Y={}".format(X.shape, Y.shape))

    #         # # make copy of self
    #         # TODO how to make copy??
    #         # svgp_new = svgp.make_copy()
    #         # var_strat = svgp.variational_strategy.base_variational_strategy
    #         # TODO how to handle diff Z on each output_dim??
    #         var_dist = (
    #             svgp.variational_strategy.base_variational_strategy.variational_distribution
    #         )
    #         Z = svgp.variational_strategy.base_variational_strategy.inducing_points[
    #             0, ...
    #         ]
    #         print("var_dist.mean {}".format(var_dist.mean.shape))
    #         print("var_dist.mean {}".format(var_dist.lazy_covariance_matrix.shape))
    #         # print("Z={}".format(Z.shape))

    #         # GPyTorch's way of computing Kuf:
    #         # full_inputs = torch.cat([inducing_points, X], dim=-2)
    #         # full_inputs = torch.cat([torch.squeeze(Z).T, X], dim=-2)
    #         full_inputs = torch.cat([Z, X], dim=-2)
    #         # full_covar = svgp.covar_module(full_inputs)
    #         full_covar = svgp.covar_module(full_inputs)

    #         # Covariance terms
    #         num_induc = Z.size(-2)
    #         induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
    #         induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()

    #         K_uf = induc_data_covar

    #         # Kuu = self.covar_module(inducing_points)
    #         Kuu = induc_induc_covar
    #         # Kuu_root = Kuu.cholesky()

    #         # lambda_1, lambda_2 = mean_cov_to_natural_param(var_mean, var_cov, Kuu)
    #         lambda_1, lambda_2 = mean_cov_to_natural_param(
    #             var_dist.mean, var_dist.lazy_covariance_matrix, Kuu
    #         )

    #         lambda_1_t = torch.zeros_like(lambda_1)
    #         lambda_2_t = torch.zeros_like(lambda_2)
    #         #
    #         # online_update
    #         for _ in range(1):  # TODO: make parameter
    #             # grad_varexp_natural_params
    #             with torch.no_grad():
    #                 # Xt = torch.tile(X, Y.shape[:-2] + (1, 1, 1))
    #                 #                 if Y.shape[-1] == 1:
    #                 #                     Xt.unsqueeze_(-1)
    #                 # pred = svgp.forward(X)
    #                 pred = svgp(X)
    #                 # print("pred {}".format(pred))
    #                 mean = pred.mean  # [output_dim, num_new]
    #                 var = pred.variance  # [output_dim, num_new]
    #                 # print("pred_mean {}".format(mean))
    #                 # print("pred_var {}".format(var))
    #                 # mean = pred.latent_dist.mean
    #                 # var = pred.latent_dist.variance
    #             mean.requires_grad_()
    #             var.requires_grad_()
    #             # mean = pred.mean
    #             # var = pred.variance
    #             # mean.requires_grad_()
    #             # var.requires_grad_()

    #             # mean = mean[:, None, ...]
    #             # var = var[:, None, ...]
    #             # print("pred_mean {}".format(mean.shape))
    #             # print("pred_var {}".format(var.shape))
    #             # variational expectations
    #             # TODO gpytroch MVN or torch?
    #             # f_dist_b = gpytorch.distributions.MultivariateNormal(
    #             #     mean.T, torch.diag_embed(var.T)
    #             # )  # Mean: num_new x output_dim Cov: num_new x output_dim x output_dim
    #             # f_dist = (
    #             #     gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
    #             #         f_dist_b
    #             #     )
    #             # )
    #             # f_dist = gpytorch.distributions.MultitaskMultivariateNormal.from_independent_mvns(
    #             #     f_dist_b
    #             # )

    #             # f_dist = gpytorch.distributions.MultitaskMultivariateNormal(f_dist_b)
    #             # f_dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    #             # f_dist = gpytorch.distributions.MultivariateNormal(
    #             #     mean.T, torch.diag_embed(var.T)
    #             # )  # Mean: B x N Cov: B x N x N
    #             # print("f_dist_b {}".format(f_dist_b))
    #             # print("f_dist_b.event_shape {}".format(f_dist_b.event_shape))
    #             # print("f_dist_b {}".format(f_dist_b))
    #             # # print("f_dist {}".format(f_dist))
    #             # # f_dist = MultivariateNormal(mean, DiagLazyTensor(var))
    #             # # f_dist = gpytorch.distributions.MultitaskMultivariateNormal(mean, var)'
    #             # print("likelihood {}".format(likelihood))
    #             # print("likelihood.noise {}".format(likelihood.noise))

    #             # # print("likelihood.noise_covar {}".format(likelihood.noise_covar))
    #             # def expected_log_prob(target, input):
    #             #     mean, variance = input.mean, input.variance
    #             #     num_event_dim = len(input.event_shape)

    #             #     print("target.shape {}".format(target.shape))
    #             #     noise = likelihood._shaped_noise_covar(mean.shape).diagonal(
    #             #         dim1=-1, dim2=-2
    #             #     )
    #             #     print("noise {}".format(noise.shape))
    #             #     print("input.event_shape {}".format(input.event_shape))
    #             #     # Potentially reshape the noise to deal with the multitask case
    #             #     # noise = noise.view(*noise.shape[:-1], *input.event_shape)

    #             #     res = (
    #             #         ((target - mean).square() + variance) / noise
    #             #         + noise.log()
    #             #         + math.log(2 * math.pi)
    #             #     )
    #             #     res = res.mul(-0.5)
    #             #     if (
    #             #         num_event_dim > 1
    #             #     ):  # Do appropriate summation for multitask Gaussian likelihoods
    #             #         res = res.sum(list(range(-1, -num_event_dim, -1)))
    #             #     return res

    #             def predict_ve(mean, var):
    #                 # print("mean {}".format(mean.shape))
    #                 # print("var {}".format(var.shape))
    #                 f_dist_b = gpytorch.distributions.MultivariateNormal(
    #                     mean.T, torch.diag_embed(var.T)
    #                 )  # Mean: num_new x output_dim Cov: num_new x output_dim x output_dim
    #                 # print("f_3ist_b {}".format(f_dist_b))
    #                 f_dist = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
    #                     f_dist_b
    #                 )
    #                 # print("f_dist {}".format(f_dist))
    #                 ve_terms = likelihood.expected_log_prob(
    #                     Y, f_dist
    #                 )  # TODO: Is this right?
    #                 # print("ve_terms {}".format(ve_terms.shape))
    #                 ve = (
    #                     ve_terms.sum()
    #                 )  # TODO: CHECK: divide by num_data ? but only one point at a time so probably fine
    #                 # print("ve {}".format(ve.shape))
    #                 return ve

    #             # ve_terms = expected_log_prob(Y.T, f_dist)  # TODO: Is this right?
    #             # ve_terms = likelihood.expected_log_prob(
    #             #     # Y, pred,
    #             #     Y,
    #             #     f_dist,
    #             # )  # TODO: Is this right?
    #             # print("ve_terms {}".format(ve_terms.shape))
    #             # ve = (
    #             #     ve_terms.sum()
    #             # )  # TODO: CHECK: divide by num_data ? but only one point at a time so probably fine
    #             from functorch import jacrev

    #             jac_fn_mean = jacrev(predict_ve)
    #             jac_fn_var = jacrev(predict_ve, argnums=1)
    #             d_exp_dm = jac_fn_mean(mean, var)  # [num_new, output_dim]
    #             d_exp_dv = jac_fn_var(mean, var)  # [num_new, output_dim]
    #             # print(d_exp_dm.shape)
    #             # print(d_exp_dv.shape)
    #             # print(d_exp_dm)
    #             # print(d_exp_dv)

    #             # ve.backward(inputs=[mean, var])
    #             # ve.backward(inputs=[mean, var])
    #             # d_exp_dm = mean.grad  # [batch, N]
    #             # d_exp_dv = var.grad  # [batch, N]

    #             eps = 1e-8
    #             d_exp_dv.clamp_(max=-eps)

    #             grad_nat_1 = d_exp_dm - 2.0 * (d_exp_dv * mean)
    #             grad_nat_2 = d_exp_dv

    #             grad_mu_1 = torch.einsum("bmc, cb -> bm", K_uf, grad_nat_1)

    #             # print("K_uf {}".format(K_uf.shape))
    #             # K_uf = K_uf.permute(2, 1, 0)
    #             K_fu = K_uf.permute(0, 2, 1)
    #             # print("K_uf {}".format(K_uf.shape))
    #             # print("K_fu {}".format(K_fu.shape))
    #             # print("grad_nat_2 {}".format(grad_nat_2.shape))
    #             # grad_mu_2 = torch.einsum("bmc, cb, bnc -> bmn", K_uf, grad_nat_2, K_uf)
    #             # grad_mu_2 = torch.einsum("nmf, nf, ncf -> nmc", K_uf, grad_nat_2, K_uf)
    #             # print(
    #             #     "torch.diag(grad_nat_2) {}".format(
    #             #         torch.diag_embed(grad_nat_2).shape
    #             #     )
    #             # )
    #             grad_mu_2 = K_uf @ torch.diag_embed(grad_nat_2.T) @ K_fu
    #             # print("grad_mu_2 {}".format(grad_mu_2.shape))
    #             # print("K_uf {}".format(K_uf))
    #             # grad_mu_2 = K_uf @ K_uf.T
    #             # L = torch.cholesky(grad_mu_2)
    #             # print("psd")
    #             # print("lambda_2_t {}".format(lambda_2_t.shape))

    #             scale = 1.0

    #             lambda_1_t_new = (
    #                 1.0 - learning_rate
    #             ) * lambda_1_t + learning_rate * scale * grad_mu_1[..., None]
    #             lambda_2_t_new = (
    #                 1.0 - learning_rate
    #             ) * lambda_2_t + learning_rate * scale * (-2) * grad_mu_2

    #             lambda_1_new = lambda_1 - lambda_1_t + lambda_1_t_new
    #             lambda_2_new = lambda_2 - lambda_2_t + lambda_2_t_new
    #             print("lambda_1_new {}".format(lambda_1_new.shape))
    #             print("lambda_2_new {}".format(lambda_2_new.shape))

    #             # print("jitter {}".format(jitter))
    #             new_mean, new_cov = conditional_from_precision_sites_white_full(
    #                 Kuu,
    #                 lambda_1_new,
    #                 lambda_2_new,
    #                 jitter=jitter,
    #                 # jitter=getattr(self, "tsvgp_jitter", 0.0),
    #             )
    #             print("new_mean {}".format(new_mean.shape))
    #             print("new_cov {}".format(new_cov.shape))
    #             new_mean = new_mean.squeeze(-1)
    #             print("new_mean {}".format(new_mean.shape))

    #             with torch.no_grad():
    #                 # var_dist
    #                 svgp.variational_strategy.base_variational_strategy.variational_distribution.mean.set_(
    #                     new_mean
    #                 )
    #                 svgp.variational_strategy.base_variational_strategy.variational_distribution.covariance_matrix.set_(
    #                     new_cov
    #                 )
    #                 # var_dist.mean.set_(new_mean)
    #                 # var_dist.covariance_matrix.set_(new_cov)

    #             lambda_1 = lambda_1_new
    #             lambda_2 = lambda_2_new
    #             lambda_1_t = lambda_1_t_new
    #             lambda_2_t = lambda_2_t_new

    #     # print("before fail")
    #     # print(x.shape)
    #     f = svgp(x)
    #     # print(f)
    #     output = likelihood(f)
    #     noise_var = output.variance - f.variance
    #     return f.mean, f.variance, noise_var


def base_conditional_with_lm(
    Kmn: TensorType["NumInducing", "NumData"],
    Lm: TensorType["NumInducing", "NumInducing"],
    Knn: Union[TensorType["NumData", "NumData"], TensorType["NumData"]],
    f: TensorType["NumInducing"],
    full_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[TensorType["NumInducing", "NumInducing"], TensorType["NumInducing"]]
    ] = None,
    white: Optional[bool] = False,
):
    """Same as base_conditional but expects the cholesky Lm instead of Kmm = Lm Lm.T

    Lm can be precomputed, improving performance.
    """
    A = torch.linalg.solve_triangular(Lm, Kmn, upper=False)  # [M, N]
    # print("A")
    # print(A.shape)
    # print(Knn.shape)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - torch.matmul(A.T, A)
    else:
        fvar = Knn - torch.sum(torch.square(A), 0)
    # print("fvar: {}".format(fvar.shape))

    # another backsubstitution in the unwhitened case
    if not white:
        A = torch.linalg.solve_triangular(Lm.T, A, upper=True)  # [M, N]
        # print("A: {}".format(A.shape))

    # conditional mean
    fmean = A.T @ f  # [N]
    # print("fmean: {}".format(fmean.shape))

    # covariance due to inducing variables
    if q_sqrt is not None:
        if q_sqrt.ndim == 1:
            LTA = q_sqrt[..., None] * A  # [M, N]
        elif q_sqrt.ndim == 2:
            LTA = q_sqrt.T @ A  # [M, N]
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))
        # print("LTA: {}".format(LTA.shape))

        if full_cov:
            fvar = fvar + LTA.T @ LTA  # [N, N]
        else:
            fvar = fvar + torch.sum(torch.square(LTA), 0)  # [N]
        # print("fvar yo you: {}".format(fvar.shape))

    return fmean, fvar


def train(
    svgp: SVGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    learning_rate: float = 0.1,
    num_data: int = None,
    wandb_loss_name: str = None,
    early_stopper: EarlyStopper = None,
):
    if early_stopper is not None:
        early_stopper.reset()
    svgp.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [{"params": svgp.parameters()}, {"params": likelihood.parameters()}],
        lr=learning_rate,
    )
    loss_fn_ = loss_fn(svgp=svgp, likelihood=likelihood, num_data=num_data)

    def train_(data_loader: DataLoader, num_epochs: int):
        stop_flag = False
        svgp.train()
        likelihood.train()
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(data_loader):
                optimizer.zero_grad()
                loss = loss_fn_(batch)
                loss.backward()
                optimizer.step()
                if wandb_loss_name is not None:
                    wandb.log({wandb_loss_name: loss})
                    # wandb.log({wandb_loss_name + ": sigma_n": likelihood.variance})
                    # print("svgp.covar_module.base_kernel.lengthscale.shape")
                    # print(svgp.covar_module.base_kernel.lengthscale.shape)
                    # if len(svgp.covar_module.base_kernel.lengthscale.shape) == 2:
                    #     wandb.log(
                    #         {
                    #             wandb_loss_name
                    #             + ": l_00": svgp.covar_module.base_kernel.lengthscale[
                    #                 0, 0
                    #             ]
                    #         }
                    #     )
                    # elif len(svgp.covar_module.base_kernel.lengthscale.shape) == 3:
                    #     # print("svgp.covar_module.base_kernel.lengthscale.shape")
                    #     # print(svgp.covar_module.base_kernel.lengthscale.shape)
                    #     wandb.log(
                    #         {
                    #             wandb_loss_name
                    #             + ": l_00": svgp.covar_module.base_kernel.lengthscale[
                    #                 0, 0, 0
                    #             ]
                    #         }
                    #     )
                # for i, param in enumerate(svgp.covar_module.base_kernel.lengthscale):
                #     print("i {}".format(i))
                #     print("param {}".format(param.shape))
                #     for j, param_ in enumerate(param[0, :]):
                #         print("param_ {}".format(param_))
                #         print("j {}".format(j))
                #         print(type(param_))
                #         wandb.log({"lengthscale " + str(i) + str(j): float(param_)})
                # wandb.log({"sigma_f": self.gp.covar_module.base_kernel.variance})
                logger.info(
                    "Epoch : {} | Batch: {} | Loss: {}".format(epoch, batch_idx, loss)
                )
                if early_stopper is not None:
                    stop_flag = early_stopper(loss)
                if stop_flag:
                    logger.info("Early stopping criteria met, stopping training")
                    break
            if stop_flag:
                logger.info("Breaking out loop")
                break

    return train_


def loss_fn(svgp, likelihood: gpytorch.likelihoods.Likelihood, num_data: int):
    mll = gpytorch.mlls.VariationalELBO(likelihood, svgp, num_data=num_data)

    def loss_fn_(batch):
        x, y = batch
        latent = svgp(x)
        loss = -mll(latent, y)
        return loss

    return loss_fn_


def mean_cov_to_natural_param(mu, Su, K_uu):
    """
    Transforms (m,S) to (λ₁,P) tsvgp_white parameterization
    """
    # print("mu {}".format(mu.shape))
    # mu = torch.unsqueeze(mu, dim=2)
    # Su = Su @ Su.T
    assert mu.ndim == 2
    mu = torch.unsqueeze(mu.T, dim=-1)
    # mu = mu.T
    # print("mu {}".format(mu.shape))
    # print("Su {}".format(Su.shape))
    # print("K_uu {}".format(K_uu.shape))
    lamb1 = K_uu.matmul(Su.inv_matmul(mu))[..., 0].permute(1, 0)
    lamb2 = K_uu.matmul(Su.inv_matmul(K_uu.evaluate())) - K_uu.evaluate()

    return lamb1, lamb2


def conditional_from_precision_sites_white_full(
    Kuu,
    lambda1,
    Lambda2,
    jitter=1e-9,
):
    """
    Given a g₁ and g2, and distribution p and q such that
      p(g₂) = N(g₂; 0, Kuu)
      p(g₁) = N(g₁; 0, Kff)
      p(g₁ | g₂) = N(g₁; Kfu (Kuu⁻¹) g₂, Kff - Kfu (Kuu⁻¹) Kuf)
    And  q(g₂) = N(g₂; μ, Σ) such that
        Σ⁻¹  = Kuu⁻¹  + Kuu⁻¹LLᵀKuu⁻¹
        Σ⁻¹μ = Kuu⁻¹l
    This method computes the mean and (co)variance of
      q(g₁) = ∫ q(g₂) p(g₁ | g₂) dg₂ = N(g₂; μ*, Σ**)
    with
    Σ** = k** - kfu Kuu⁻¹ kuf - kfu Kuu⁻¹ Σ Kuu⁻¹ kuf
        = k** - kfu Kuu⁻¹kuf + kfu (Kuu + LLᵀ)⁻¹ kuf
    μ* = k*u Kuu⁻¹ m
       = k*u Kuu⁻¹ Λ⁻¹ Kuu⁻¹ l
       = k*u (Kuu + LLᵀ)⁻¹ l
    Inputs:
    :param Kuu: tensor output_dim x M x M
    :param l: tensor M x output_dim
    :param L: tensor output_dim x M x M
    """
    # TODO: rewrite this
    # print("lambda1 {}".format(lambda1.shape))

    R = Lambda2 + Kuu.evaluate()
    # print("R {}".format(R.shape))
    if R.ndim == 3:
        R += torch.eye(R.shape[-1], device=R.device)[None, ...] * jitter
    else:
        R += torch.eye(R.shape[-1], device=R.device) * jitter
    R = (Lambda2 + Kuu).add_jitter(jitter)
    # print("R {}".format(R.shape))

    # U = torch.cholesky(Lambda2)
    # print("R {}".format(type(R)))
    # print("R {}".format(R))
    # print(R.shape)
    # L = torch.linalg.cholesky(R)
    # lambda1 = lambda1.permute(0, 2, 1)
    # print("lambda1 {}".format(lambda1.shape))
    # print("L {}".format(L.shape))
    # mean = torch.matmul(R.inverse(), lambda1)
    # mean = Kuu.matmul(torch.cholesky_solve(lambda1, L))
    # print("mean {}".format(mean.shape))
    # mean = Kuu.matmul(mean)
    # print("mean {}".format(mean.shape))
    # mean = Kuu.matmul(torch.cholesky_solve(lambda1, U))
    # mean = Kuu.matmul(torch.cholesky_solve(lambda1, U))
    # cov = Kuu.matmul(torch.cholesky_solve(Kuu.evaluate(), L))
    # print("cov {}".format(cov.shape))
    # cov = cov.matmul(Kuu)
    # print("cov {}".format(cov.shape))
    # cov = Kuu.matmul(torch.cholesky_solve(Kuu.evaluate(), U))
    # mean = A @ A.T
    lambda1 = lambda1[..., None].permute(1, 0, 2)
    # print("here lambda1 {}".format(lambda1.shape))
    mean = Kuu.matmul(R.inv_matmul(lambda1))[..., 0]
    cov = Kuu.matmul(R.inv_matmul(Kuu.evaluate()))  # TODO: symmetrise?
    return mean, cov


# # class SVGP(gpytorch.models.ApproximateGP):
# class SVGP:
#     def __init__(
#         self,
#         # mean_fn,
#         # kernel,
#         # inducing_inputs,
#         gp: gpytorch.models.ApproximateGP,
#         # variational_distribution: gpytorch.variational.CholeskyVariationalDistribution,
#         # likelihood: gpytorch.likelihoods.Likelihood = None,
#         # mean_module: gpytorch.means.Mean = None,
#         # covar_module: gpytorch.kernels.Kernel = None,
#         # num_inducing: int = 16,
#         # learning_rate: float = 0.1,
#         # batch_size: int = 16,
#         # num_iterations: int = 1000,
#         num_workers: int = 1,
#         learn_inducing_locations: bool = True,
#     ):
#         variational_strategy = (
#             gpytorch.variational.IndependentMultitaskVariationalStrategy(
#                 gpytorch.variational.VariationalStrategy(
#                     self,
#                     inducing_points,
#                     variational_distribution,
#                     learn_inducing_locations=learn_inducing_locations,
#                 ),
#                 num_tasks=out_size,
#             )
#         )
#         super().__init__(variational_strategy)
#         if mean_module is None:
#             mean_module = gpytorch.means.ConstantMean(
#                 batch_shape=torch.Size([out_size])
#             )
#         if covar_module is None:
#             covar_module = gpytorch.kernels.ScaleKernel(
#                 gpytorch.kernels.RBFKernel(batch_shape=torch.Size([out_size])),
#                 batch_shape=torch.Size([out_size]),
#             )

#         # self.gp = gp
#         if likelihood is None:
#             likelihood = gpytorch.likelihoods.GaussianLikelihood()

#         self.likelihood = likelihood

#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.num_iterations = num_iterations
#         self.num_workers = num_workers
#         self.learn_inducing_locations = learn_inducing_locations

#         print("GP params:")
#         for param in self.gp.parameters():
#             print(param)
#         print("Lik params:")
#         for param in self.likelihood.parameters():
#             print(param)
#         # self.gp = torch.compile(self.gp)
#         # self.likelihood = torch.compile(self.likelihood)

#     def forward(self, x, data_new: Optional = None) -> Prediction:
#         self.gp.eval()
#         self.likelihood.eval()

#         if data_new != None:
#             X, Y = data_new

#             # # make copy of self
#             model = self.make_copy()
#             var_strat = self.model.variational_strategy.base_variational_strategy
#             inducing_points = var_strat.inducing_points

#             var_dist = var_strat.variational_distribution
#             var_mean = var_dist.mean
#             var_cov = var_dist.lazy_covariance_matrix

#             # GPyTorch's way of computing Kuf:
#             # full_inputs = torch.cat([inducing_points, X], dim=-2)
#             full_inputs = torch.cat([torch.squeeze(inducing_points).T, X], dim=-2)
#             full_covar = self.model.covar_module(full_inputs)

#             # Covariance terms
#             num_induc = inducing_points.size(-2)
#             induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
#             induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()

#             K_uf = induc_data_covar

#             # Kuu = self.covar_module(inducing_points)
#             Kuu = induc_induc_covar
#             # Kuu_root = Kuu.cholesky()

#             lambda_1, lambda_2 = mean_cov_to_natural_param(var_mean, var_cov, Kuu)

#             lambda_1_t = torch.zeros_like(lambda_1)
#             lambda_2_t = torch.zeros_like(lambda_2)
#             #
#             # online_update
#             for _ in range(1):  # TODO: make parameter
#                 # grad_varexp_natural_params
#                 with torch.no_grad():
#                     # Xt = torch.tile(X, Y.shape[:-2] + (1, 1, 1))
#                     #                 if Y.shape[-1] == 1:
#                     #                     Xt.unsqueeze_(-1)
#                     pred = model.forward(X)
#                     mean = pred.latent_dist.mean
#                     var = pred.latent_dist.variance
#                 mean.requires_grad_()
#                 var.requires_grad_()

#                 # variational expectations
#                 f_dist_b = MultivariateNormal(
#                     mean.T, torch.diag_embed(var.T)
#                 )  # Mean: B x N Cov: B x N x N
#                 f_dist = (
#                     gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
#                         f_dist_b
#                     )
#                 )
#                 # f_dist = MultivariateNormal(mean, DiagLazyTensor(var))
#                 # f_dist = gpytorch.distributions.MultitaskMultivariateNormal(mean, var)'
#                 ve_terms = self.likelihood.expected_log_prob(
#                     Y, f_dist
#                 )  # TODO: Is this right?
#                 ve = (
#                     ve_terms.sum()
#                 )  # TODO: CHECK: divide by num_data ? but only one point at a time so probably fine

#                 ve.backward(inputs=[mean, var])
#                 d_exp_dm = mean.grad  # [batch, N]
#                 d_exp_dv = var.grad  # [batch, N]

#                 eps = 1e-8
#                 d_exp_dv.clamp_(max=-eps)

#                 grad_nat_1 = d_exp_dm - 2.0 * (d_exp_dv * mean)
#                 grad_nat_2 = d_exp_dv

#                 grad_mu_1 = torch.einsum("bmc, cb -> bm", K_uf, grad_nat_1)

#                 grad_mu_2 = torch.einsum("bmc, cb, bnc -> bmn", K_uf, grad_nat_2, K_uf)

#                 lr = 0.8  # TODO: set as a parameter
#                 scale = 1.0

#                 lambda_1_t_new = (1.0 - lr) * lambda_1_t + lr * scale * grad_mu_1[
#                     ..., None
#                 ]
#                 lambda_2_t_new = (1.0 - lr) * lambda_2_t + lr * scale * (-2) * grad_mu_2

#                 lambda_1_new = lambda_1 - lambda_1_t + lambda_1_t_new
#                 lambda_2_new = lambda_2 - lambda_2_t + lambda_2_t_new

#                 new_mean, new_cov = conditional_from_precision_sites_white_full(
#                     Kuu,
#                     lambda_1_new,
#                     lambda_2_new,
#                     jitter=getattr(self, "tsvgp_jitter", 0.0),
#                 )
#                 new_mean = new_mean.squeeze(-1)
#                 new_cov_root = new_cov.cholesky()

#                 # fantasy_var_dist = fantasy_model.variational_strategy._variational_distribution
#                 with torch.no_grad():
#                     var_dist = (
#                         self.model.variational_strategy.base_variational_strategy.variational_distribution
#                     )
#                     var_dist.mean.set_(new_mean)
#                     var_dist.covariance_matrix.set_(new_cov)

#                 lambda_1 = lambda_1_new
#                 lambda_2 = lambda_2_new
#                 lambda_1_t = lambda_1_t_new
#                 lambda_2_t = lambda_2_t_new

#         # self.gp_module.eval()
#         # f = self.gp_module.forward(x)
#         f = self.gp(x)
#         # print("latent {}".format(f.variance))

#         output = self.likelihood(f)
#         # print("output {}".format(output))
#         f_dist = td.Normal(loc=f.mean, scale=torch.sqrt(f.variance))
#         # print("f_dist {}".format(f_dist))
#         y_dist = td.Normal(loc=f.mean, scale=torch.sqrt(output.variance))
#         # print("y_dist {}".format(y_dist))
#         noise_var = output.variance - f.variance

#         # pred = Prediction(latent_dist=f_dist, output_dist=y_dist, noise_var=noise_var)
#         # return pred
#         return f.mean, f.variance, noise_var

#     def make_copy(self):
#         with torch.no_grad():
#             inducing_points = (
#                 self.gp.variational_strategy.base_variational_strategy.inducing_points.detach().clone()
#             )

#             if hasattr(self, "input_transform"):
#                 [p.detach_() for p in self.input_transform.buffers()]

#             # new_covar_module = deepcopy(self.gp_module.covar_module)

#             new_model = self.__class__(
#                 likelihood=deepcopy(self.likelihood),
#                 mean_module=deepcopy(self.gp.mean_module),
#                 covar_module=deepcopy(self.gp.covar_module),
#                 learning_rate=deepcopy(self.learning_rate),
#                 # TODO add other arguments
#                 num_inducing=self.gp.num_inducing,
#                 batch_size=self.batch_size,
#                 num_iterations=self.num_iterations,
#                 delta_state=self.delta_state,
#                 # num_workers: int = 1,
#                 learn_inducing_locations=self.gp.learn_inducing_locations,
#             )
#             #             new_model.mean_module = deepcopy(self.mean_module)
#             #             new_model.likelihood = deepcopy(self.likelihood)

#             var_dist = (
#                 self.gp.variational_strategy.base_variational_strategy.variational_distribution
#             )
#             mean = var_dist.mean.detach().clone()
#             cov = var_dist.covariance_matrix.detach().clone()

#             new_var_dist = (
#                 new_model.model.variational_strategy.base_variational_strategy.variational_distribution
#             )
#             with torch.no_grad():
#                 new_var_dist.mean.set_(mean)
#                 new_var_dist.covariance_matrix.set_(cov)
#                 new_model.model.variational_strategy.base_variational_strategy.inducing_points.set_(
#                     inducing_points
#                 )

#             new_model.model.variational_strategy.base_variational_strategy.variational_params_initialized.fill_(
#                 1
#             )

#         return new_model
