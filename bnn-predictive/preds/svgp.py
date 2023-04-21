"""SVGP class for neural networks.."""

import torch
import torch.distributions as dist
import gpytorch
import functorch

class SVGP(torch.nn.Module):

    def __init__(self, nn_model, likelihood,  data, alpha, beta,  kernel_type='ntk', device='cpu'):
        # data
        y, x = data
        self.y = y  
        self.x = x

        # dual parameters
        self.alpha = alpha
        self.beta = beta

        # trained model and kernel type
        self.nn_model = nn_model
        self.kernel_type = 'ntk'

        self.likelihood = likelihood

        self.device = device

    def predict(self, x_):
        """Evaluate the SVGP predictive."""

    def sample(self, x_, n_samples):
        """Sample the predictive distribution."""
        pred_mu, pred_var = self.predict(x_)
        pred_dist = dist.MultivariateNormal(pred_mu, pred_var)
        
        samples = pred_dist.sample((n_samples, ))
        return samples



