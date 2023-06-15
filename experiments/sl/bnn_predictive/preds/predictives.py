import logging

import src as ntksvgp
import torch
from experiments.sl.bnn_predictive.preds.gradients import Jacobians_naive
from experiments.sl.bnn_predictive.preds.kron import Kron
from experiments.sl.bnn_predictive.preds.likelihoods import GaussianLh, get_Lams_Vys
from experiments.sl.bnn_predictive.preds.optimizers import GGN
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset


def nn_sampling_predictive(
    X, model, likelihood, mu, Sigma_chol, mc_samples=100, no_link=False
):
    theta_star = parameters_to_vector(model.parameters())
    if type(Sigma_chol) is Kron:
        covs = Sigma_chol.sample(mc_samples)
    elif len(Sigma_chol.shape) == 2:
        covs = (Sigma_chol @ torch.randn(len(mu), mc_samples, device=mu.device)).t()
    elif len(Sigma_chol.shape) == 1:
        covs = (
            Sigma_chol.reshape(-1, 1)
            * torch.randn(len(mu), mc_samples, device=mu.device)
        ).t()
    samples = mu + covs
    predictions = list()
    link = (lambda x: x) if no_link else likelihood.inv_link
    for i in range(mc_samples):
        vector_to_parameters(samples[i], model.parameters())
        f = model(X)
        predictions.append(link(f).detach())
    vector_to_parameters(theta_star, model.parameters())
    return torch.stack(predictions)


def linear_sampling_predictive(
    X, model, likelihood, mu, Sigma_chol, mc_samples=100, no_link=False
):
    theta_star = parameters_to_vector(model.parameters())
    Js, f = Jacobians_naive(model, X)
    if len(Js.shape) > 2:
        Js = Js.transpose(1, 2)
    offset = f.squeeze() - Js @ theta_star
    if len(Sigma_chol.shape) == 2:
        covs = (Sigma_chol @ torch.randn(len(mu), mc_samples, device=mu.device)).t()
    elif len(Sigma_chol.shape) == 1:
        covs = (
            Sigma_chol.reshape(-1, 1)
            * torch.randn(len(mu), mc_samples, device=mu.device)
        ).t()
    samples = mu + covs
    predictions = list()
    link = (lambda x: x) if no_link else likelihood.inv_link
    for i in range(mc_samples):
        f = offset + Js @ samples[i]
        predictions.append(link(f).detach())
    return torch.stack(predictions)


def svgp_sampling_predictive(
    X,
    svgp,
    likelihood,
    mc_samples=100,
    no_link=False,
    batch_size=None,
    nn_mean=False,
    device="cpu",
):
    """Returns the sparse data used for convenience."""
    link = (lambda x: x) if no_link else likelihood.inv_link
    if batch_size is None:
        f_mu, f_var = svgp.predict_f(X)
        if nn_mean:
            f_mu = svgp.network(X)
        fs = Normal(f_mu, torch.sqrt(f_var.clamp(1e-5)))
        return link(fs.sample((mc_samples,)))
    else:
        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        ps = []
        for X in iter(data_loader):
            X = X[0]
            X = X.to(device)
            ps.append(
                sample_svgp(X, likelihood, svgp, n_samples=mc_samples, nn_mean=nn_mean)
            )
        ps = torch.cat(ps, axis=1)
        return ps


def sample_svgp(
    X,
    likelihood,
    svgp,
    n_samples: int,
    nn_mean=False,
):
    """Sample the SVGP, assumes a batched input."""
    n_data = X.shape[0]
    gp_means, gp_vars = svgp.predict_f(X)
    if nn_mean:
        gp_means = svgp.network(X)
    dist = Normal(gp_means, torch.sqrt(gp_vars.clamp(10 ** (-8))))
    logit_samples = dist.sample((n_samples,))
    out_dim = logit_samples.shape[-1]
    samples = likelihood.inv_link(logit_samples)
    samples = samples.reshape(n_samples, n_data, out_dim)
    return samples


def functional_sampling_predictive(
    X, model, likelihood, mu, Sigma, mc_samples=1000, no_link=False
):
    theta_star = parameters_to_vector(model.parameters())
    Js, f = Jacobians_naive(model, X)
    # reshape to batch x output x params
    if len(Js.shape) > 2:
        Js = Js.transpose(1, 2)
    else:
        Js = Js.unsqueeze(1)  # add the output dimension
    f_mu = f + Js @ (mu - theta_star)
    if type(Sigma) is Kron:
        # NOTE: Sigma is in this case not really cov but prec-kron and internally inverted
        f_var = Sigma.functional_variance(Js)
    elif len(Sigma.shape) == 2:
        f_var = torch.einsum("nkp,pq,ncq->nkc", Js, Sigma, Js)
    elif len(Sigma.shape) == 1:
        f_var = torch.einsum("nkp,p,ncp->nkc", Js, Sigma, Js)
    if type(likelihood) is GaussianLh:
        return f_mu, f_var
    else:
        link = (lambda x: x) if no_link else likelihood.inv_link
        try:
            fs = MultivariateNormal(f_mu, f_var)
            return link(fs.sample((mc_samples,)))
        except RuntimeError:
            logging.warning("functional sampling covariance indefinite - use diagonal")
            fs = Normal(f_mu, f_var.diagonal(dim1=1, dim2=2).clamp(1e-5))
            return link(fs.sample((mc_samples,)))


def linear_regression_predictive(X, model, likelihood, mu, Sigma_chol):
    theta_star = parameters_to_vector(model.parameters())
    if len(Sigma_chol.shape) == 2:
        Sigma = Sigma_chol @ Sigma_chol.t()
    elif len(Sigma_chol.shape) == 1:
        Sigma = torch.diag(Sigma_chol**2)
    Js, Hess, f = GGN(model, likelihood, X)
    Lams, Vys = get_Lams_Vys(likelihood, Hess)
    delta = mu - theta_star
    # Lam Js = Jacobians of inv link g (m x p x k)
    Jgs = torch.bmm(Js, Lams)
    lin_pred = torch.einsum("mpk,p->mk", Jgs, delta).reshape(*f.shape)
    mu_star = (likelihood.inv_link(f) + lin_pred).detach()
    var_f = torch.bmm(Jgs.transpose(1, 2) @ Sigma, Jgs).squeeze().detach()
    var_noise = Vys.squeeze().detach()
    return mu_star, var_f, var_noise
