#!/usr/bin/env python3
import logging
from functools import partial
from typing import Callable, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch.nn.functional as torch_F
import numpy as np
import torch
import torch.nn as nn
from src.custom_types import (
    Action,
    Data,
    InputData,
    OutputData,
    Prediction,
    State,
    OutputMean,
    OutputVar,
)
from src.utils import EarlyStopper
from torch.func import functional_call, jacrev, jvp, vjp, vmap, hessian

# from torch.nn.functional import hessian, jacobian
from torchrl.data import ReplayBuffer
from torchtyping import TensorType


Alpha = TensorType["num_data", "output_dim"]
Beta = TensorType["num_data", "num_data", "output_dim"]
AlphaInducing = TensorType["num_inducing", "output_dim"]
BetaInducing = TensorType["num_inducing", "num_inducing", "output_dim"]

FuncData = TensorType["num_data", "output_dim"]
InducingPoints = TensorType["num_inducing", "input_dim"]

Lambda_1 = TensorType["num_data", "num_inducing", "output_dim"]
Lambda_2 = TensorType["num_data", "output_dim", "output_dim"]
NTK = Callable[[InputData, InputData], TensorType[""]]

TestInput = TensorType["num_test", "input_dim"]


def nll(f: FuncData, y: OutputData):
    # return 0.5 * torch.nn.MSELoss(reduction="sum")(f, y)
    loss = torch.nn.MSELoss()(f, y)
    return 0.5 * loss * y.shape[-1]


class NTKSVGP:
    def __init__(
        self,
        network: torch.nn.Module,
        train_data: Data,
        num_inducing: int = 30,
        jitter: float = 1e-6,
        delta: float = 0.001,
        nll=nll,
    ):
        X_train, Y_train = train_data
        assert X_train.ndim == 2
        assert Y_train.ndim == 2
        assert X_train.shape[0] == Y_train.shape[0]
        num_data, output_dim = Y_train.shape
        print("Y_train.shape {}".format(Y_train.shape))
        num_data = X_train.shape[0]
        indices = torch.randperm(num_data)[:num_inducing]
        Z = X_train[indices]
        # num_inducing = 100
        # Z = torch.linspace(0, 3, num_inducing).reshape(-1, 1)
        # Z = torch.rand(num_inducing, 1) * 3
        assert Z.ndim == 2
        self.num_inducing, self.input_dim = Z.shape
        self.Z = Z
        self.jitter = jitter

        self.kernel = buil_ntk(
            network=network, num_data=num_data, output_dim=output_dim, delta=delta
        )

        self.alpha, self.beta = calc_sparse_dual_params(
            network=network, train_data=train_data, Z=Z, kernel=self.kernel, nll=nll
        )
        print("alpha {}".format(self.alpha.shape))
        print("beta {}".format(self.beta.shape))
        assert self.alpha.ndim == 2
        assert self.beta.ndim == 3
        assert self.alpha.shape[0] == output_dim
        assert self.alpha.shape[1] == num_inducing
        assert self.beta.shape[0] == output_dim
        assert self.beta.shape[1] == num_inducing
        assert self.beta.shape[2] == num_inducing

        self._predict_fn = predict_from_duals(
            alpha=self.alpha,
            beta=self.beta,
            kernel=self.kernel,
            Z=self.Z,
            jitter=self.jitter,
        )

    @torch.no_grad()
    def predict(self, x: TestInput):
        # TODO implement noise_var correctly
        f_mean, f_var = self._predict_fn(x, full_cov=False)
        return Prediction(mean=f_mean, var=f_var, noise_var=0.0)

    def update(self, x: InputData, y: OutputData):
        assert x.ndim == 2 and y.ndim == 2
        num_new_data, input_dim = x.shape
        Kui = self.kernel(self.Z, x)
        print("Kui {}".format(Kui.shape))
        print("alpha {}".format(self.alpha.shape))
        print("beta {}".format(self.beta.shape))
        print("x {}".format(x.shape))
        print("y {}".format(y.shape))

        # lambda_1, lambda_2 = calc_lambdas(Y=Y, F=F, nll=nll)

        self.alpha += (Kui @ y.T[..., None])[..., 0]
        self.beta += (
            Kui
            @ (1**-1 * torch.eye(num_new_data)[None, ...])
            @ torch.transpose(Kui, -1, -2)
        )
        print("ALPHA {}".format(self.alpha.shape))
        print("BETA {}".format(self.beta.shape))

        self._predict_fn = predict_from_duals(
            alpha=self.alpha,
            beta=self.beta,
            kernel=self.kernel,
            Z=self.Z,
            jitter=self.jitter,
        )


def buil_ntk(
    network: nn.Module, num_data: int, output_dim: int, delta: float = 1.0
) -> NTK:
    # Detaching the parameters because we won't be calling Tensor.backward().
    params = {k: v.detach() for k, v in network.named_parameters()}

    def fnet_single(params, x, i):
        return functional_call(network, params, (x.unsqueeze(0),))[0, ...][:, i]

    def single_output_ntk(x1: InputData, x2: InputData, i):
        # func_x1 = partial(fnet_single, x=x1, i=i)
        # func_x2 = partial(fnet_single, x=x2, i=i)
        def func_x1(params):
            return fnet_single(params, x1, i=i)

        def func_x2(params):
            return fnet_single(params, x2, i=i)

        output, vjp_fn = vjp(func_x1, params)
        # print("output {}".format(output))

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # print("vjps {}".format(vjps))
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            # print("jvps {}".format(jvps))
            return jvps

        # Here's our identity matrix
        basis = torch.eye(
            output.numel(), dtype=output.dtype, device=output.device
        ).view(output.numel(), -1)
        # print("basis {}".format(basis))
        return 1 / (delta * num_data) * vmap(get_ntk_slice)(basis)

    def ntk(x1: InputData, x2: InputData) -> TensorType[""]:
        K = torch.empty(output_dim, x1.shape[0], x2.shape[0])
        # print("K building {}".format(K.shape))
        for i in range(output_dim):
            # print("output dim {}".format(i))
            K[i, :, :] = single_output_ntk(x1, x2, i=i)
        # print("K {}".format(K.shape))
        return K

    return ntk


def predict_from_duals(
    alpha: Alpha, beta: Beta, kernel: NTK, Z: InducingPoints, jitter: float = 1e-3
):
    print("Z {}".format(Z.shape))
    Kuu = kernel(Z, Z)
    output_dim = Kuu.shape[0]
    print("Kuu {}".format(Kuu.shape))
    Iu = torch.eye(Kuu.shape[-1])[None, ...].repeat(output_dim, 1, 1)
    print("Iu {}".format(Iu.shape))
    Kuu += Iu * jitter
    # beta += I
    print("Kuu {}".format(Kuu.shape))

    assert beta.shape == Kuu.shape
    # iBKuu = torch.linalg.solve(beta + Kuu, torch.eye(Kuu.shape[-1]))
    # print("iBKuu {}".format(iBKuu.shape))
    # V = torch.matmul(torch.matmul(Kuu, iBKuu), Kuu)
    V = torch.matmul(Kuu, torch.linalg.solve(beta + Kuu, Kuu))
    print("V {}".format(V.shape))
    # iKuuViKuu = torch.linalg.solve(torch.linalg.solve(Kuu, V), Kuu, left=False)
    # print("iKuuViKuu {}".format(iKuuViKuu.shape))
    # iKuuViKuua = torch.matmul(iKuuViKuu, alpha[..., None])
    # print("iKuuVKuua {}".format(iKuuViKuua.shape))

    def predict(x: TestInput, full_cov: bool = False) -> Tuple[OutputMean, OutputVar]:
        Kxx = kernel(x, x)
        print("Kxx {}".format(Kxx.shape))
        Kxu = kernel(x, Z)
        print("Kxu {}".format(Kxu.shape))

        # f_mean = torch.matmul(Kxu, iKuuViKuua)
        # print("f_mean {}".format(f_mean.shape))
        # f_mean = f_mean[..., 0].T
        # print("f_mean {}".format(f_mean.shape))
        # Iu = torch.eye(Kuu.shape[-1])[None, ...].repeat(ouput_dim, 1, 1)
        print("Iu {}".format(Iu.shape))
        print("V {}".format(V.shape))
        # print("alpha[...,None] {}".format(alpha[..., None].shape))
        # print(
        #     "torch.eye(Kuu.shape[-1])[None, ...] {}".format(
        #         torch.eye(Kuu.shape[-1])[None, ...].shape
        #     )
        # )
        # print(
        #     "torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...]) {}".format(
        #         torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...]).shape
        #     )
        # )
        m_u = (
            V
            @ torch.linalg.solve(Kuu, Iu)
            # @ torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...])
            @ alpha[..., None]
        )
        print("m_u {}".format(m_u.shape))

        # print(
        #     "torch.linalg.solve(Kuu, torch.eye(Kuu.shape[-1])[None, ...] {}".format(
        #         torch.linalg.solve(Kuu, Iu).shape
        #     )
        # )
        f_mean = Kxu @ torch.linalg.solve(Kuu, Iu) @ m_u
        print("f_mean {}".format(f_mean.shape))
        f_mean = f_mean[..., 0].T
        print("f_mean {}".format(f_mean.shape))
        beta_u = torch.linalg.solve(Kuu, Iu) - torch.linalg.solve(beta + Kuu, Iu)
        print("beta_u {}".format(beta_u.shape))
        print("Kuu {}".format(Kuu.shape))
        print("Iu {}".format(Iu.shape))

        if full_cov:
            f_cov = Kxx - torch.matmul(
                torch.matmul(Kxu, iBKuu), torch.transpose(Kxu, -1, -2)
            )
            print("f_cov full_cov {}".format(f_cov.shape))
            return f_mean, f_cov
        else:
            # TODO implement more efficiently
            # f_cov = Kxx - torch.matmul(
            #     torch.matmul(Kxu, iBKuu), torch.transpose(Kxu, -1, -2)
            # )
            f_cov = Kxx - torch.matmul(
                torch.matmul(Kxu, beta_u), torch.transpose(Kxu, -1, -2)
            )
            print("f_cov {}".format(f_cov.shape))
            f_var = torch.diagonal(f_cov, dim1=-2, dim2=-1).T
            print("f_var {}".format(f_var.shape))
            return f_mean, f_var

    return predict


def calc_sparse_dual_params(
    network: torch.nn.Module,
    train_data: Tuple[InputData, OutputData],
    Z: InducingPoints,
    kernel: NTK,
    nll: Callable[[FuncData, OutputData], float],
) -> Tuple[AlphaInducing, BetaInducing]:
    num_inducing, input_dim = Z.shape
    X, Y = train_data
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == input_dim
    Kuf = kernel(Z, X)
    print("Kuf {}".format(Kuf.shape))
    F = network(X)
    print("F {}".format(F.shape))
    lambda_1, lambda_2 = calc_lambdas(Y=Y, F=F, nll=nll)
    print("lambda_1 {}".format(lambda_1.shape))
    print("lambda_2 {}".format(lambda_2.shape))
    alpha, beta = calc_sparse_dual_params_from_lambdas(
        lambda_1=lambda_1, lambda_2=lambda_2, Kuf=Kuf
    )
    print("alpha {}".format(alpha.shape))
    print("beta {}".format(beta.shape))
    return alpha, beta


def calc_sparse_dual_params_from_lambdas(
    lambda_1: Lambda_1,
    lambda_2: Lambda_2,
    Kuf: TensorType["output_dim", "num_inducing", "num_data"],
) -> Tuple[AlphaInducing, BetaInducing]:
    assert lambda_1.ndim == 2
    num_data, output_dim = lambda_1.shape
    assert lambda_2.ndim == 3
    assert lambda_2.shape[0] == num_data
    assert lambda_2.shape[1] == lambda_2.shape[2] == output_dim
    assert Kuf.ndim == 3
    assert Kuf.shape[0] == output_dim
    assert Kuf.shape[2] == num_data
    alpha_u = torch.matmul(Kuf, torch.transpose(lambda_1, -1, -2)[..., None])[..., 0]
    print("alpha_u {}".format(alpha_u.shape))
    lambda_2_diag = torch.diagonal(lambda_2, dim1=-2, dim2=-1)  # [num_data, output_dim]
    # TODO broadcast lambda_2 correctly for multiple output dims
    print("lambda_2_diag {}".format(lambda_2_diag.shape))
    # inv_lambda_2 = (
    #     torch.transpose(lambda_2_diag, -1, -2) ** -1 * torch.repeat(torch.eye(num_data)[None, ...]
    # )  # [output_dim, num_data, num_data]
    # print("inv_lambda_2 {}".format(inv_lambda_2.shape))
    inv_lambda_2 = torch.diag_embed(lambda_2_diag.T**-1)
    print("inv_lambda_2 {}".format(inv_lambda_2.shape))
    print("inv_lambda_2 {}".format(inv_lambda_2))
    beta_u = torch.matmul(
        torch.matmul(Kuf, inv_lambda_2),
        torch.transpose(Kuf, -1, -2),
    )
    print("beta_u {}".format(beta_u.shape))
    return alpha_u, beta_u


def calc_lambdas(
    Y: OutputData,  # [num_data, output_dim]
    F: FuncData,  # [num_data, output_dim]
    nll: Callable[[FuncData, OutputData], float],
) -> Tuple[Lambda_1, Lambda_2]:
    assert Y.ndim == 2
    assert F.ndim == 2
    assert Y.shape[0] == F.shape[0]
    assert Y.shape[1] == F.shape[1]
    nll_jacobian_fn = jacrev(nll)
    nll_hessian_fn = torch.vmap(hessian(nll))

    # nll_jacobian_fn = torch.gradient(nll)
    lambda_1 = nll_jacobian_fn(F, Y)
    lambda_2 = nll_hessian_fn(F, Y)
    # lambda_1, lambda_2 = [], []
    # TODO we can do better than a for loop...
    # for y, f in zip(Y, F):
    #     # lambda_1.append(nll_jacobian_fn(f, y))
    #     print("nll_hessian_fn(f, y) {}".format(nll_hessian_fn(f, y).shape))
    #     lambda_2.append(nll_hessian_fn(f, y))
    #     # TODO implement clipping for lambdas
    # lambda_1 = torch.stack(lambda_1, dim=0)  # [num_data, output_dim]
    # TODO should lambda_1 just be Y?
    lambda_1 = Y
    print("lambda_2 {}".format(lambda_2))
    # lambda_2 = torch.stack(lambda_2, dim=0)  # [num_data, output_dim, output_dim]
    return lambda_1, lambda_2


def train(
    network: nn.Module,
    # noise_var,
    data,
    num_epochs: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    loss_fn=torch.nn.MSELoss(),
    delta=0.001,
):
    # params = torch.nn.utils.parameters_to_vector(network)
    # print("params {}".format(params))
    # print("params {}".format(type(params)))
    # def regularised_loss_fn(x, y):
    def regularised_loss_fn(x, y):
        # dist = torch.distributions.Normal(loc=network(x), scale=torch.sqrt(noise_var))
        # log_prob = dist.log_prob(y)
        # print("log_prob {}".format(log_prob.shape))
        # params = torch.Tensor(list(network.parameters()))
        squared_params = torch.cat(
            [torch.square(param.view(-1)) for param in network.parameters()]
        )
        # l2r = 0.5 * torch.sum(torch.sum(jnp.square(p)) for p in jtu.tree_leaves(params))
        l2r = 0.5 * torch.sum(squared_params)
        # print("params {}".format(params))
        # num_params = params.shape[0]
        # # print("num_params {}".format(num_params))
        # log_prior = torch.distributions.Normal(
        #     loc=torch.zeros(num_params), scale=l2_decay**2 * torch.ones(num_params)
        # ).log_prob(params)
        # # print("log_prior {}".format(log_prior.shape))
        # return -log_prob.sum() - log_prior.sum()
        # l2_norm = torch.Tensor(
        #     [torch.square(param).sum() for param in network.parameters()]
        # )
        # print("l2_norm {}".format(1 / (2 * l2_decay**2) * l2_norm.sum()))
        # # return loss_fn(x, y) + 1 / (2 * l2_decay**2) * l2_norm.sum()
        f_pred = network(x)
        # return 0.5 * loss_fn(y_pred, y) + delta * l2r
        return 0.5 * loss_fn(f_pred, y) + delta * l2r

    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data),
        batch_size=batch_size,
        # collate_fn=collate_wrapper,
        # pin_memory=True,
    )

    network.train()
    optimizer = torch.optim.Adam([{"params": network.parameters()}], lr=learning_rate)
    loss_history = []
    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            x, y = batch
            # pred = network(x)
            # loss = loss_fn(pred, y)
            # loss = regularised_loss_fn(pred, y)
            loss = regularised_loss_fn(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.detach().numpy())

            print("Epoch: {} | Batch: {} | Loss: {}".format(epoch_idx, batch_idx, loss))
            # logger.info("Iteration : {} | Loss: {}".format(i, loss))
    return {"loss": loss_history}


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    def func(x, noise=True):
        # x = x + 1e-6
        f1 = torch.sin(x * 5) / x + torch.cos(
            x * 10
        )  # + x**2 +np.log(5*x + 0.00001)  + 0.5
        f2 = torch.sin(x) / x + torch.cos(x)  # + x**2 +np.log(5*x + 0.00001)  + 0.5
        f3 = torch.cos(x * 5) + torch.sin(x * 1)  # + x**2 +np.log(5*x + 0.00001)  + 0.5
        if noise == True:
            y1 = f1 + torch.randn(size=(x.shape)) * 0.2
            y2 = f2 + torch.randn(size=(x.shape)) * 0.2
            y3 = f3 + torch.randn(size=(x.shape)) * 0.2
            return torch.stack([y1[:, 0], y2[:, 0], y3[:, 0]], -1)
        else:
            return torch.stack([f1[:, 0], f2[:, 0], f3[:, 0]], -1)

    delta = 0.0001
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 3),
    )
    print("network: {}".format(network))
    # noise_var = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)

    # X_train = torch.rand((50, 1)) * 2 - 1
    # X_train = torch.rand((50, 1)) * 2
    X_train = torch.rand((100, 1)) * 2
    print("X_train {}".format(X_train.shape))
    X_train_clipped_1 = X_train[X_train < 1.5].reshape(-1, 1)
    X_train_clipped_2 = X_train[X_train > 1.9].reshape(-1, 1)
    # print("X_train {}".format(X_train.shape))
    X_train = torch.concat([X_train_clipped_1, X_train_clipped_2], 0)
    print("X_train {}".format(X_train.shape))
    # X_train = torch.linspace(-1, 1, 50, dtype=torch.float64).reshape(-1, 1)
    # Y_train = func(X_train, noise=True)
    Y_train = func(X_train, noise=True)
    data = (X_train, Y_train)
    print("X, Y: {}, {}".format(X_train.shape, Y_train.shape))
    # X_test = torch.linspace(-1.8, 1.8, 200, dtype=torch.float64).reshape(-1, 1)
    X_test_short = torch.linspace(0.0, 2.05, 110, dtype=torch.float64).reshape(-1, 1)
    X_test = torch.linspace(-0.2, 2.2, 200, dtype=torch.float64).reshape(-1, 1)
    X_test = torch.linspace(-1.0, 2.2, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-6.0, 2.2, 200, dtype=torch.float64).reshape(-1, 1)
    X_test = torch.linspace(-2.0, 3.5, 300, dtype=torch.float64).reshape(-1, 1)
    X_test = torch.linspace(-0.7, 3.5, 300, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-8, 8, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-2, 2, 100, dtype=torch.float64).reshape(-1, 1)
    print("X_test: {}".format(X_test.shape))
    print("f: {}".format(network(X_test).shape))

    X_new = torch.linspace(-0.5, -0.2, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new = func(X_new, noise=True)

    # X_new_2 = torch.linspace(3.0, 4.0, 20, dtype=torch.float64).reshape(-1, 1)
    # Y_new_2 = func(X_new_2, noise=True)
    X_new_2 = torch.linspace(1.6, 1.8, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new_2 = func(X_new_2, noise=True)

    X_new_3 = torch.linspace(-6.0, -5.0, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new_3 = func(X_new_3, noise=True)

    batch_size = X_train.shape[0]
    metrics = train(
        network=network,
        # noise_var=noise_var,
        data=data,
        num_epochs=2500,
        # num_epochs=25,
        batch_size=batch_size,
        learning_rate=1e-2,
        # loss_fn=torch.nn.MSELoss(),
        loss_fn=nll,
        delta=delta,
    )

    svgp = NTKSVGP(
        network=network,
        train_data=(X_train, Y_train),
        num_inducing=30,
        # jitter=1e-6,
        jitter=1e-4,
        delta=delta,
        nll=nll,
    )
    pred = svgp.predict(X_test_short)

    # pred = predict(network=network, train_data=(X_train, Y_train), delta=delta)(X_test)
    # print("pred {}".format(pred))
    print("MEAN {}".format(pred.mean.shape))
    print("VAR {}".format(pred.var.shape))
    print("X_test_short {}".format(X_test_short.shape))
    print(X_test_short.shape)

    svgp.update(x=X_new, y=Y_new)
    pred_new = svgp.predict(X_test)
    print("mean NEW {}".format(pred_new.mean.shape))
    print("var NEW {}".format(pred_new.var.shape))

    svgp.update(x=X_new_2, y=Y_new_2)
    pred_new_2 = svgp.predict(X_test)
    print("MEAN NEW_2 {}".format(pred_new_2.mean.shape))
    print("VAR NEW_2 {}".format(pred_new_2.var.shape))

    import matplotlib.pyplot as plt

    plot_var = False
    plot_var = True

    # fig = plt.subplots(1, 1)
    # plt.plot(np.arange(len(metrics["loss"])), metrics["loss"])
    # plt.savefig("loss.pdf", transparent=True)
    # fig = plt.subplots(1, 1)
    # plt.scatter(X_train, Y_train, color="k", marker="x", label="Data")
    # plt.legend()
    # # plt.savefig("data.pdf", transparent=True)
    # fig = plt.subplots(1, 1)
    # plt.scatter(X_train, Y_train, color="k", marker="x", alpha=0.6, label="Data")
    # plt.plot(
    #     X_test[:, 0],
    #     func(X_test, noise=False),
    #     color="b",
    #     label=r"$f_{true}(\cdot)$",
    # )
    # plt.plot(
    #     X_test[:, 0],
    #     network(X_test).detach()[:, i],
    #     color="m",
    #     linestyle="--",
    #     label=r"$f_{NN}(\cdot)$",
    # )
    # plt.savefig("nnyo.pdf", transparent=True)

    def plot_output(i):
        fig = plt.subplots(1, 1)
        plt.scatter(
            X_train, Y_train[:, i], color="k", marker="x", alpha=0.6, label="Data"
        )
        plt.plot(
            X_test[:, 0],
            func(X_test, noise=False)[:, i],
            color="b",
            label=r"$f_{true}(\cdot)$",
        )
        plt.plot(
            X_test[:, 0],
            network(X_test).detach()[:, i],
            color="m",
            linestyle="--",
            label=r"$f_{NN}(\cdot)$",
        )

        plt.plot(X_test_short[:, 0], pred.mean[:, i], color="c", label=r"$\mu(\cdot)$")
        if plot_var:
            plt.fill_between(
                X_test_short[:, 0],
                (pred.mean - 1.98 * torch.sqrt(pred.var))[:, i],
                # pred.mean[:, 0],
                (pred.mean + 1.98 * torch.sqrt(pred.var))[:, i],
                color="c",
                alpha=0.2,
                label=r"$\mu(\cdot) \pm 1.98\sigma$",
            )
        plt.scatter(
            svgp.Z, torch.ones_like(svgp.Z) * -5, marker="|", color="b", label="Z"
        )
        plt.legend()
        # plt.savefig("nn2svgp.pdf", transparent=True)
        plt.savefig("nn2svgp" + str(i) + ".pdf", transparent=True)

        plt.scatter(
            X_new, Y_new[:, i], color="m", marker="o", alpha=0.6, label="New data"
        )
        plt.plot(
            X_test[:, 0], pred_new.mean[:, i], color="m", label=r"$\mu_{new}(\cdot)$"
        )
        if plot_var:
            plt.fill_between(
                X_test[:, 0],
                (pred_new.mean - 1.98 * torch.sqrt(pred_new.var))[:, i],
                # pred.mean[:, 0],
                (pred_new.mean + 1.98 * torch.sqrt(pred_new.var))[:, i],
                color="m",
                alpha=0.2,
                label=r"$\mu_{new}(\cdot) \pm 1.98\sigma_{new}(\cdot)$",
            )

        plt.scatter(
            X_new_2, Y_new_2[:, i], color="y", marker="o", alpha=0.6, label="New data 2"
        )
        plt.plot(
            X_test[:, 0],
            pred_new_2.mean[:, i],
            color="y",
            linestyle="-",
            label=r"$\mu_{new,2}(\cdot)$",
        )
        if plot_var:
            plt.fill_between(
                X_test[:, 0],
                (pred_new_2.mean - 1.98 * torch.sqrt(pred_new_2.var))[:, i],
                # pred.mean[:, 0],
                (pred_new_2.mean + 1.98 * torch.sqrt(pred_new_2.var))[:, i],
                color="y",
                alpha=0.2,
                label=r"$\mu_{new,2}(\cdot) \pm 1.98\sigma_{new,2}(\cdot)$",
            )

        # svgp.update(x=X_new_3, y=Y_new_3)
        # pred_new_3 = svgp.predict(X_test)
        # print("mean NEW {}".format(pred_new_3.mean.shape))
        # print("var NEW {}".format(pred_new_3.var.shape))
        # plt.scatter(X_new_3, Y_new_3, color="g", marker="o", alpha=0.6, label="New data 3")
        # plt.plot(
        #     X_test[:, 0],
        #     pred_new_3.mean[:, 0],
        #     color="g",
        #     linestyle="-",
        #     label=r"$\mu_{new,3}(\cdot)$",
        # )
        # if plot_var:
        #     plt.fill_between(
        #         X_test[:, 0],
        #         (pred_new_3.mean - 1.98 * torch.sqrt(pred_new_3.var))[:, 0],
        #         # pred.mean[:, 0],
        #         (pred_new_3.mean + 1.98 * torch.sqrt(pred_new_3.var))[:, 0],
        #         color="y",
        #         alpha=0.2,
        #         label=r"$\mu_{new,3}(\cdot) \pm 1.98\sigma_{new,3}(\cdot)$",
        #     )

        plt.legend()
        plt.savefig("nn2svgp_new" + str(i) + ".pdf", transparent=True)

    for i in range(3):
        plot_output(i)
