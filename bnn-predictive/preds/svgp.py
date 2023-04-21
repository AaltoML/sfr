"""SVGP class for neural networks.

Current implementation does not update the duals after init!
This class is best suited for supervised learning.
"""

import torch
import torch.distributions as dist
import gpytorch
from functorch import make_functional, vmap, jacrev, jit, hessian

class SVGPNTK():

    def __init__(self, nn_model, data, likelihood, n_sparse=100, device='cpu'):
        # data
        (y, x) = data
        self.n_classes = nn_model(self.x[0]).shape[-1]

        # randomly select n_sparse points
        self.n_sparse = n_sparse
        indices = torch.randperm(y.shape[0])[:n_sparse]
        self.y = y[indices]
        self.x = x[indices]

        # precompute the duals
        self.estimate_duals(data)

        # trained model and kernel type
        self.kernel = NTK(nn_model, device)
        self.likelihood = likelihood
        self.device = device

    def estimate_lambdas(self, lambdas_data, clip_lambda=True):
        (y, logits_train) = lambdas_data
        lambdas = jit(vmap(hessian(mse)))(logits_train, y)
        # for multiclass, lambdas shp (K,K); only clip diagonal
        if clip_lambda:
            if self.n_classes > 1:
                diag_indices = torch.index_exp[:,torch.arange(lambdas.shape[1]),
                                         torch.arange(lambdas.shape[2])]
        else:
            diag_indices = torch.index_exp[:]
        lambdas_diag = torch.clip(lambdas[diag_indices], EPS)
        lambdas = lambdas.at[diag_indices].set(lambdas_diag)
    
    lambdas = lambdas.reshape(lambdas.shape[0], 1).squeeze()
    return lambdas

    def estimate_duals(self, data):
        """Estimate the duals alpha and beta."""
        (y, x) = data
        lambdas_data = (y, self.nn_model(x))
        lambdas = self.estimate_lambdas(lambdas_data, clip_lambda=True)
    
        gram = torch.squeeze(self.ntk.empirical_ntk(X, None, params))
        K = 1/(delta*X.shape[0]) * gram 
        A = lambdas**-1 * torch.eye(gram.shape[0]) + K
        alpha_f = torch.linalg.solve(A, y)
        beta_f = torch.linalg.solve(lambdas**-1 * torch.eye(gram.shape[0]) + K, torch.eye(K.shape[0]))
        return alpha_f, beta_f

    def predict(self, x_p):
        gram_pp = torch.squeeze(self.ntk.empirical_ntk(x_p, x_p))
        gram_px = torch.squeeze(self.ntk.empirical_ntk(x_p, x))
        K_pp = 1/(delta*X.shape[0]) * gram_pp
        K_px = 1/(delta*X.shape[0]) * gram_px
    
        mean_f = K_px @ self.alpha
        var_f = K_pp  - K_px @ self.beta @ K_px.T
        var_f = torch.diag(var_f)
        return mean_f, var_f

    def sample_pred(self, x_, n_samples):
        """Sample the predictive distribution."""
        pred_mu, pred_var = self.predict(x_)
        pred_dist = dist.MultivariateNormal(pred_mu, pred_var)
        samples = pred_dist.sample((n_samples, ), device=self.device)
        return samples


class NTK(torch.nn.Module):

    def __init__(self, nn_model, device='cpu'):
        self.fnet, self.params = make_functional(nn_model().to(device))
        self.device = device
         
    def fnet_single(self, x):
        return self.fnet(self.params, x.unsqueeze(0)).squeeze(0)

    def empirical_ntk(self, x1, x2):
        # Compute J(x1)
        jac1 = vmap(jacrev(self.fnet_single), (None, 0))(self.params, x1)
        jac1 = [j.flatten(2) for j in jac1]
    
        # Compute J(x2)
        jac2 = vmap(jacrev(self.fnet_single), (None, 0))(self.params, x2)
        jac2 = [j.flatten(2) for j in jac2]
    
        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result



def mse(logits, targets):
    return 0.5*torch.square(logits-targets).mean()


