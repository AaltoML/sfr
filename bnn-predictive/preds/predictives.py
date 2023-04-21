import torch
from torch.distributions import MultivariateNormal, Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import logging

from preds.gradients import Jacobians_naive
from preds.optimizers import GGN
from preds.likelihoods import get_Lams_Vys, GaussianLh
from preds.kron import Kron


def nn_sampling_predictive(X, model, likelihood, mu, Sigma_chol, mc_samples=100, no_link=False):
    theta_star = parameters_to_vector(model.parameters())
    if type(Sigma_chol) is Kron:
        covs = Sigma_chol.sample(mc_samples)
    elif len(Sigma_chol.shape) == 2:
        covs = (Sigma_chol @ torch.randn(len(mu), mc_samples, device=mu.device)).t()
    elif len(Sigma_chol.shape) == 1:
        covs = (Sigma_chol.reshape(-1, 1) * torch.randn(len(mu), mc_samples, device=mu.device)).t()
    samples = mu + covs
    predictions = list()
    link = (lambda x: x) if no_link else likelihood.inv_link
    for i in range(mc_samples):
        vector_to_parameters(samples[i], model.parameters())
        f = model(X)
        predictions.append(link(f).detach())
    vector_to_parameters(theta_star, model.parameters())
    return torch.stack(predictions)


def linear_sampling_predictive(X, model, likelihood, mu, Sigma_chol, mc_samples=100, no_link=False):
    theta_star = parameters_to_vector(model.parameters())
    Js, f = Jacobians_naive(model, X)
    if len(Js.shape) > 2:
        Js = Js.transpose(1, 2)
    offset = f - Js @ theta_star
    if len(Sigma_chol.shape) == 2:
        covs = (Sigma_chol @ torch.randn(len(mu), mc_samples, device=mu.device)).t()
    elif len(Sigma_chol.shape) == 1:
        covs = (Sigma_chol.reshape(-1, 1) * torch.randn(len(mu), mc_samples, device=mu.device)).t()
    samples = mu + covs
    predictions = list()
    link = (lambda x: x) if no_link else likelihood.inv_link
    for i in range(mc_samples):
        f = offset + Js @ samples[i]
        predictions.append(link(f).detach())
    return torch.stack(predictions)

def svgp_sampling_predictive(X, X_train, y_train, model, likelihood, mc_samples=100, no_link=False):
    link = (lambda x: x) if no_link else likelihood.inv_link
    lambdas = lambdas_fn(nll_fn, model, y_train, n_classes=likelihood)
    lambdas = lambdas.reshape(lambdas.shape[0], 1).squeeze()
    delta = 0.0001
    m = (lambdas, y, X, param, delta)
    a, b = get_dual(X_train, Y_train, m)
    du = (a, b)
    kernel_fn = nt.empirical_ntk_fn(vmap(apply_fn, in_axes=(None, 0)), trace_axes=(), diagonal_axes=(1,), vmap_axes=0)
    f_mu, f_var = pred_svgp(X_lin, kernel_fn, m, du)
    fs = MultivariateNormal(f_mu, f_var)
    return link(fs.sample((mc_samples,)))

def lambdas_fn(nll_fn, logits_train, y, n_classes, clip_lambda=True):
    """In case of multiclass, this returns array with shape (N,K,K)"""
    lambdas = jit(vmap(hessian(nll_fn)))(logits_train, y)
    # for multiclass, lambdas shp (K,K); only clip diagonal
    if clip_lambda:
        if n_classes > 1:
            diag_indices = jnp.index_exp[:,jnp.arange(lambdas.shape[1]),
                                         jnp.arange(lambdas.shape[2])]
        else:
            diag_indices = jnp.index_exp[:]
        lambdas_diag = jnp.clip(lambdas[diag_indices], EPS)
        lambdas = lambdas.at[diag_indices].set(lambdas_diag)
    return lambdas

def get_dual(X, y, model):
    lambdas, y, x, params, delta = model
    
    gram = torch.squeeze(kernel_fn(X, None, params))
    K = 1/(delta*X.shape[0]) * gram 
    
    A = lambdas**-1 * torch.eye(gram.shape[0]) + K
    
    alpha_f = torch.solve(A, y)
    beta_f = torch.solve(lambdas**-1 * torch.eye(gram.shape[0]) + K, torch.eye(K.shape[0]))
    
    return alpha_f, beta_f

def pred_svgp(x_p, kernel, model, dual_p):
    lambdas, y, x, params, delta = model
    alpha, beta = dual_p
    
    gram_pp = torch.squeeze(kernel(x_p, x_p, params))
    gram_px = torch.squeeze(kernel(x_p, x, params))
    K_pp = 1/(delta*X.shape[0]) * gram_pp
    K_px = 1/(delta*X.shape[0]) * gram_px
    
    mean_f = K_px @ alpha
    var_f = K_pp  - K_px @ beta @ K_px.T
    var_f = torch.diag(var_f)
    
    return mean_f, var_f


def functional_sampling_predictive(X, model, likelihood, mu, Sigma, mc_samples=1000, no_link=False):
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
        f_var = torch.einsum('nkp,pq,ncq->nkc', Js, Sigma, Js)
    elif len(Sigma.shape) == 1:
        f_var = torch.einsum('nkp,p,ncp->nkc', Js, Sigma, Js)
    if type(likelihood) is GaussianLh:
        return f_mu, f_var
    else:
        link = (lambda x: x) if no_link else likelihood.inv_link
        try:
            fs = MultivariateNormal(f_mu, f_var)
            return link(fs.sample((mc_samples,)))
        except RuntimeError:
            logging.warning('functional sampling covariance indefinite - use diagonal')
            fs = Normal(f_mu, f_var.diagonal(dim1=1, dim2=2).clamp(1e-5))
            return link(fs.sample((mc_samples,)))


def linear_regression_predictive(X, model, likelihood, mu, Sigma_chol):
    theta_star = parameters_to_vector(model.parameters())
    if len(Sigma_chol.shape) == 2:
        Sigma = Sigma_chol @ Sigma_chol.t()
    elif len(Sigma_chol.shape) == 1:
        Sigma = torch.diag(Sigma_chol ** 2)
    Js, Hess, f = GGN(model, likelihood, X)
    Lams, Vys = get_Lams_Vys(likelihood, Hess)
    delta = mu - theta_star
    # Lam Js = Jacobians of inv link g (m x p x k)
    Jgs = torch.bmm(Js, Lams)
    lin_pred = torch.einsum('mpk,p->mk', Jgs, delta).reshape(*f.shape)
    mu_star = (likelihood.inv_link(f) + lin_pred).detach()
    var_f = torch.bmm(Jgs.transpose(1, 2) @ Sigma, Jgs).squeeze().detach()
    var_noise = Vys.squeeze().detach()
    return mu_star, var_f, var_noise
