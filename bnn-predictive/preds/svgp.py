"""SVGP class for neural networks.."""

import torch
import torch.distributions as dist
import gpytorch
from functorch import make_functional, vmap, vjp, jvp, jacrev

class SVGP(torch.nn.Module):

    def __init__(self, nn_model, likelihood,  data, alpha, beta, device='cpu'):
        # data
        (y, x) = data
        self.y = y  
        self.x = x

        # dual parameters
        self.alpha = alpha
        self.beta = beta

        # trained model and kernel type
        self.kernel = NTK(nn_model, device)
        self.likelihood = likelihood
        self.device = device

    def predict(self, x_):
        """Evaluate the SVGP predictive."""


    def sample(self, x_, n_samples):
        """Sample the predictive distribution."""
        pred_mu, pred_var = self.predict(x_)
        pred_dist = dist.MultivariateNormal(pred_mu, pred_var)
        samples = pred_dist.sample((n_samples, ), device=self.device)
        return samples


class NTK(torch.nn.Module):

    def __init__(self, nn_model, device='cpu'):
        self.fnet, self.params = make_functional(nn_model().to(device))
        self.device = device
         
    def fnet_single(self, params, x):
        return self.fnet(params, x.unsqueeze(0)).squeeze(0)

    def empirical_ntk_jacobian_contraction(self, params, x1, x2):
        # Compute J(x1)
        jac1 = vmap(jacrev(self.fnet_single), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1]
    
        # Compute J(x2)
        jac2 = vmap(jacrev(self.fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2]
    
        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result




