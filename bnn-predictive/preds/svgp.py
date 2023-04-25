"""SVGP class for neural networks.

Current implementation does not update the duals after init!
This class is best suited for supervised learning.
"""

import torch
import torch.distributions as dist
import gpytorch
import copy
from torch.func import vmap, jacrev, functional_call, hessian

class SVGPNTK():

    def __init__(self, nn_model, likelihood, data, prior_prec, n_sparse=0.25, sparse_data=None, device='cpu'):
        # data
        (y, x) = data
        self.n_classes = nn_model(x[0]).shape[-1]
        self.nn_model = nn_model

        # randomly select n_sparse points
        if sparse_data is None:
            self.n_sparse = int(y.shape[0]*n_sparse)
            indices = torch.randperm(y.shape[0])[:self.n_sparse]
            self.y = y[indices]
            self.x = x[indices]
        else:
            (sparse_y, sparse_x) = sparse_data
            self.y = sparse_y
            self.x = sparse_x

        self.n_classes = nn_model(self.x[0]).shape[-1]

        self.delta = prior_prec / y.shape[0]
        self.eps = 10**(-7)

        # trained model and kernel type
        self.kernel = NTK(nn_model, device)
        self.likelihood = likelihood
        self.device = device

        # precompute the duals
        alpha, beta = self.estimate_duals((self.y, self.x))
        self.alpha = alpha
        self.beta = beta

    def nll(self, logits, y):
        return -self.likelihood.log_likelihood(y, logits)

    def get_sparse_data(self):
        return (self.y, self.x)

    def estimate_lambdas(self, lambdas_data, clip_lambda=True):
        (y, logits_train) = lambdas_data
        lambdas_fn = hessian(self.nll)
        lambdas = lambdas_fn(logits_train, y)
        #torch.jit(vmap(self.likelihood.Hessian))(logits_train, y)
        # for multiclass, lambdas shp (K,K); only clip diagonal
        if clip_lambda:
            if self.n_classes > 1:
                diag = lambdas[:,torch.arange(lambdas.shape[1]),
                                         torch.arange(lambdas.shape[2])]
                diag = torch.clip(diag, self.eps)                         
            else:
                diag = torch.diag(lambdas)
                diag = torch.clip(diag, min=self.eps)
            mask = torch.diag(torch.ones_like(diag))
            lambdas = mask*torch.diag(diag) + (1. - mask)*lambdas
        lambdas = lambdas.reshape(lambdas.shape[0], -1).squeeze()
        return lambdas

    def estimate_duals(self, data):
        """Estimate the duals alpha and beta."""
        (y, x) = data
        lambdas_data = (y, self.nn_model(x))
        lambdas = self.estimate_lambdas(lambdas_data)

        gram = torch.squeeze(self.kernel.empirical_ntk(self.kernel.params, x, x))
        K = 1/(self.delta*x.shape[0]) * gram 
        A = torch.inverse(lambdas) * torch.eye(gram.shape[0]) + K
        alpha_f = torch.linalg.solve(A, y)
        beta_f = torch.linalg.solve(torch.inverse(lambdas) * torch.eye(gram.shape[0]) + K, torch.eye(K.shape[0]))
        return alpha_f, beta_f

    def predict(self, x_p):
        gram_pp = torch.squeeze(self.kernel.empirical_ntk(self.kernel.params, x_p, x_p))
        gram_px = torch.squeeze(self.kernel.empirical_ntk(self.kernel.params, x_p, self.x))
        K_pp = 1/(self.delta*self.x.shape[0]) * gram_pp
        K_px = 1/(self.delta*self.x.shape[0]) * gram_px
        mean_f = K_px @ self.alpha
        var_f = K_px @ self.beta @ K_px.T
        var_f = K_pp - var_f
        var_f = torch.diag(var_f)
        return mean_f, var_f

    def sample_pred(self, x_, n_samples):
        """Sample the predictive distribution."""
        pred_mu, pred_var = self.predict(x_)
        pred_dist = dist.MultivariateNormal(pred_mu, pred_var)
        samples = pred_dist.sample((n_samples, ), device=self.device)
        return samples


class NTK():

    def __init__(self, nn_model, device='cpu'):
        self.net = nn_model
        self.curr_net = copy.deepcopy(self.net)
        self.curr_net.eval()
        self.params = {k: v.detach() for k, v in self.net.named_parameters()}

        self.device = device


    def fnet_single(self, params, x):
        class_out = 0
        f = functional_call(self.curr_net, params, (x.unsqueeze(0)))#[:, class_out] # TODO: Why using self.net doesn't work?
        return f

    def empirical_ntk(self, params, x1, x2, compute='full'):
        """Assumes flat inputs."""
        # Compute J(x1)
        jac1 = vmap(jacrev(self.fnet_single), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1.values()]
     
        # Compute J(x2)
        jac2 = vmap(jacrev(self.fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2.values()]
        # Compute J(x1) @ J(x2).T
        einsum_expr = None
        if compute == 'full':
            einsum_expr = 'Naf,Mbf->NMab'
        elif compute == 'trace':
            einsum_expr = 'Naf,Maf->NM'
        elif compute == 'diagonal':
            einsum_expr = 'Naf,Maf->NMa'
        else:
            assert False

        result = torch.stack(
            [torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result

