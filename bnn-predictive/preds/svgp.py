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

    def __init__(self, nn_model, likelihood, data, prior_prec, n_sparse=0.25, sparse_data=None, subset=False, device='cpu'):
        # data
        (y, x) = data
        self.n_classes = nn_model(x[0]).shape[-1]
        self.nn_model = nn_model

        self.y = y
        self.x = x
         
        # randomly select n_sparse points
        if sparse_data is None:
            self.n_sparse = int(y.shape[0]*n_sparse)
            indices = torch.randperm(y.shape[0])[:self.n_sparse]
            self.z = x[indices]
            sparse_y = y[indices]
            self.z_y = sparse_y
        else:
            (sparse_y, sparse_x) = sparse_data
            self.z = sparse_x
            self.z_y = sparse_y

        if subset:  # makes the subset
            self.x = self.z 
            self.y = sparse_y

        self.n_classes = nn_model(self.x[0]).shape[-1]
        self.delta = prior_prec
        self.eps = 10**(-6)

        # trained model and kernel type
        self.kernel = NTK(nn_model, device)
        self.likelihood = likelihood
        self.device = device

        # precompute the duals
        alpha, beta = self.estimate_duals()
        self.alpha = alpha
        self.beta = beta

    def nll(self, logits, y):
        return -torch.mean(self.likelihood.log_likelihood(logits, y))

    def get_sparse_data(self):
        return (self.z_y, self.z)

    def estimate_lambdas(self, clip_lambda=True):
        logits_train = self.nn_model(self.x)
        if logits_train.ndim == 1:
            logits_train = logits_train.unsqueeze(-1)
        hessian_fn = hessian(self.nll, argnums=0)
       # lambdas = hessian_fn(logits_train.detach(), self.y.detach())
        lambdas = []
        for i in range(self.y.shape[0]):
            lambdas.append(hessian_fn(logits_train[i], self.y[i]))
        lambdas = torch.stack(lambdas, axis=0)
        if lambdas.ndim == 1:
            lambdas = lambdas.unsqueeze(-1).unsqueeze(-1)
        lambdas = vmap(torch.diag)(lambdas)
        if clip_lambda:
            lambdas = torch.clip(lambdas, self.eps)
        print(lambdas.mean())
        return lambdas

    def estimate_duals(self):
        """Estimate the duals alpha and beta."""
        lambdas_data = (self.y, self.nn_model(self.x))
        lambdas = self.estimate_lambdas(lambdas_data)
        if self.n_classes == 2:
            n_class_idx = 1
        else:
            n_class_idx = self.n_classes
        alpha_f = torch.zeros(n_class_idx, self.z.shape[0])
        beta_f = torch.zeros(n_class_idx, self.z.shape[0], self.z.shape[0])

        for i_class in range(n_class_idx):
            lambdas_class = lambdas[:, i_class]
            gram = torch.squeeze(self.kernel.empirical_ntk(self.kernel.params, self.z, self.x, class_num=i_class))    #TODO: x by z
            K = 1/(self.delta) * gram # was x.shape
            K_t = torch.transpose(K, dim0=1, dim1=0)
            lambda_inv = lambdas_class**(-1) #*torch.eye(self.x.shape[0])
            A = K @ (lambda_inv * torch.eye(self.x.shape[0]) )@ K_t
            alpha_f[i_class] = torch.matmul(K, self.y.float()) #torch.linalg.solve(A, y.float())
            beta_f[i_class] = A #torch.linalg.solve(lambda_inv * torch.eye(gram.shape[0]) + K, torch.eye(K.shape[0]))
        return alpha_f, beta_f

    def predict(self, x_p):
        if self.n_classes == 2:
            n_class_idx = 1
        else:
            n_class_idx = self.n_classes
        mean_f = torch.zeros(x_p.shape[0], n_class_idx)
        var_f = torch.zeros(x_p.shape[0], n_class_idx)
        for i_class in range(n_class_idx):
            gram_pp = torch.squeeze(self.kernel.empirical_ntk(self.kernel.params, x_p, x_p, class_num=i_class))
            gram_pz = torch.squeeze(self.kernel.empirical_ntk(self.kernel.params, x_p, self.z, class_num=i_class))
            gram_zz = torch.squeeze(self.kernel.empirical_ntk(self.kernel.params, self.z, self.z, class_num=i_class))
            K_pp = 1/(self.delta) * gram_pp
            K_pz = 1/(self.delta) * gram_pz
            K_zz = 1/(self.delta) * gram_zz
            beta_z = torch.linalg.solve(K_zz, torch.eye(K_zz.shape[0])) - torch.linalg.solve(self.beta[i_class] + K_zz, torch.eye(K_zz.shape[0]))
            var_f_class = K_pp - K_pz @ beta_z @ K_pz.T
            V = K_zz @ torch.linalg.solve(self.beta[i_class] + K_zz, K_zz)
            m_u = V @ torch.linalg.solve(K_zz, torch.eye(K_zz.shape[0])) @ self.alpha[i_class]
            m_f = K_pz @ torch.linalg.solve(K_zz, torch.eye(K_zz.shape[0])) @ m_u
            mean_f[:, i_class] = m_f
            var_f[:, i_class] = torch.diag(var_f_class)
        return mean_f.squeeze(), var_f.squeeze()

    def sample_pred(self, x_, n_samples):
        """Sample the predictive distribution."""
        pred_mu, pred_var = self.predict(x_)
        pred_dist = dist.MultivariateNormal(pred_mu, pred_var)
        samples = pred_dist.sample((n_samples, ), device=self.device)
        return samples


class NTK():

    def __init__(self, nn_model,device='cpu'):
        self.net = nn_model
        self.curr_net = copy.deepcopy(self.net)
        self.curr_net.eval()
        self.params = {k: v.detach() for k, v in self.net.named_parameters()}

        self.device = device

    def fnet_single(self, params, x, class_num=0):
        class_out = 0
        f = functional_call(self.curr_net, params, (x.unsqueeze(0)))
        if f.ndim == 1:
            f = f.unsqueeze(-1)
        else:
            f = f[:, class_num] # TODO: Why using self.net doesn't work?
        return f

    def empirical_ntk(self, params, x1, x2, class_num=0, compute='full'):
        """Assumes flat inputs."""
        # Compute J(x1)
        def fnet_class(params, x):
            return self.fnet_single(params, x, class_num=class_num)

        jac1 = vmap(jacrev(fnet_class), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1.values()]
     
        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_class), (None, 0))(params, x2)
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

