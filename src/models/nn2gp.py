#!/usr/bin/env python3
import logging
from functools import partial


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from src.custom_types import Action, Data, Prediction, State
from src.utils import EarlyStopper
from torch.func import functional_call, jacrev, jvp, vjp, vmap
from torchrl.data import ReplayBuffer


@torch.no_grad()
def predict(
    network: torch.nn.Module,
    train_data: Data,
    jitter: float = 1e-6,
    delta=0.001,
    noise_var: float = 1.0,
):
    # Detaching the parameters because we won't be calling Tensor.backward().
    params = {k: v.detach() for k, v in network.named_parameters()}

    def fnet_single(params, x):
        print("inside fnet_single {}".format(x.shape))
        f = functional_call(network, params, (x.unsqueeze(0),)).squeeze(0)[:, 0]
        print("f: {}".format(f.shape))
        return f

    def kernel(x1, x2):
        def func_x1(params):
            return fnet_single(params, x1)

        def func_x2(params):
            return fnet_single(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(
            output.numel(), dtype=output.dtype, device=output.device
        ).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)

    # kernel = partial(
    #     empirical_ntk_ntk_vps, func=fnet_single, params=params, compute="full"
    # )

    X_train, Y_train = train_data

    num_data = X_train.shape[0]

    # Kxx = (1 / delta**2) * kernel(X_train, X_train)  # [num_train, num_train]
    Kxx = kernel(X_train, X_train)  # [num_train, num_train]
    # + torch.eye(X_train.shape[-2]) * noise_var
    Kxx *= 1 / (delta * num_data)
    Kxx += torch.eye(Kxx.shape[-1]) * jitter

    # Kxx *= 1 / (delta)
    # + jnp.eye(X.shape[-2], dtype=X.dtype) * default_jitter()
    print("Kxx {}".format(Kxx))
    print("Kxx {}".format(Kxx.shape))
    B = Kxx + 2 * torch.eye(Kxx.shape[-1])
    U = torch.linalg.cholesky(Kxx + 2*torch.eye(Kxx.shape[-1]))  # [num_train, num_train]
    #print("U {}".format(U.shape))

    def predict_fn(x, full_cov: bool = False) -> Prediction:
        # mean = network.forward(x)
        print("x {}".format(x.shape))

        alpha = torch.linalg.solve(B, Y_train)
        beta = torch.linalg.solve(B, torch.eye(B.shape[0]))

        Kss = kernel(x, x)  # [num_test, num_test]
        Kss *= 1 / (delta * num_data)
        # Kss *= 1 / (delta)
        print("Kss {}".format(Kss.shape))
        Kxs = kernel(X_train, x)  # [num_train, num_test]
        Kxs *= 1 / (delta * num_data)
        # Kxs *= 1 / (delta)
        print("Kxs {}".format(Kxs.shape))

        f_mean = Kxs.T @ alpha
        A = torch.cholesky_solve(Kxs, U)  # [M, N]
        print("A {}".format(A.shape))

        # conditional mean
        #f_mean = A.T @ Y_train  # [N]
        print("f_mean {}".format(f_mean.shape))

        # compute the covariance due to the conditioning
        if full_cov:
            f_var = Kss - Kxs.T @ beta @ Kxs
        else:
            Kss = torch.diag(Kss)
            f_var = Kss - torch.diag(Kxs.T @ beta @ Kxs)
        print("f_var {}".format(f_var.shape))

        return Prediction(mean=f_mean, var=f_var, noise_var=0)

    return predict_fn


def empirical_ntk_ntk_vps(func, params, x1, x2, compute="full"):
    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(
            output.numel(), dtype=output.dtype, device=output.device
        ).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)

    # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to empirical_ntk_ntk_vps are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the vmaps here do.
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)

    if compute == "full":
        return result
    if compute == "trace":
        return torch.einsum("NMKK->NM", result)
    if compute == "diagonal":
        return torch.einsum("NMKK->NMK", result)


def train(
    network: nn.Module,
    noise_var,
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
        y_pred = network(x)
        return 0.5 * loss_fn(y_pred, y) + delta * l2r

    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data),
        batch_size=batch_size,
        # collate_fn=collate_wrapper,
        # pin_memory=True,
    )

    network.train()
    optimizer = torch.optim.Adam(
        [{"params": network.parameters()}, {"params": noise_var}], lr=learning_rate
    )
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
        f = torch.sin(x * 5) / x + torch.cos(
            x * 10
        )  # + x**2 +np.log(5*x + 0.00001)  + 0.5
        if noise == True:
            y = f + torch.randn(size=(x.shape)) * 0.4
            return y
        else:
            return f

    # delta = 0.1
    delta = 0.0001
    # delta = 0.00001
    # delta = 1000
    # delta = 0.1
    # delta = 1.0
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 25),
        torch.nn.Tanh(),
        torch.nn.Linear(25, 25),
        torch.nn.Tanh(),
        torch.nn.Linear(25, 1),
    )
    print("network: {}".format(network))
    noise_var = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)

    # X_train = torch.rand((50, 1)) * 2 - 1
    # X_train = torch.rand((50, 1)) * 2
    X_train = torch.rand((100, 1)) * 2
    # X_train = torch.linspace(-1, 1, 50, dtype=torch.float64).reshape(-1, 1)
    # Y_train = func(X_train, noise=True)
    Y_train = func(X_train, noise=True)
    data = (X_train, Y_train)
    print("X, Y: {}, {}".format(X_train.shape, Y_train.shape))
    # X_test = torch.linspace(-1.8, 1.8, 200, dtype=torch.float64).reshape(-1, 1)
    X_test = torch.linspace(0.0, 2.5, 110, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-0.2, 2.2, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-8, 8, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-2, 2, 100, dtype=torch.float64).reshape(-1, 1)
    print("X_test: {}".format(X_test.shape))
    print("f: {}".format(network(X_test).shape))

    batch_size = X_train.shape[0]
    metrics = train(
        network=network,
        noise_var=noise_var,
        data=data,
        num_epochs=2500,
        batch_size=batch_size,
        learning_rate=1e-2,
        loss_fn=torch.nn.MSELoss(),
        delta=delta,
    )

    pred = predict(
        network=network, train_data=(X_train, Y_train), delta=delta, noise_var=0
    )(X_test)
    # print("pred {}".format(pred))
    import matplotlib.pyplot as plt

    fig = plt.subplots(1, 1)
    plt.plot(np.arange(len(metrics["loss"])), metrics["loss"])
    plt.savefig("loss.pdf", transparent=True)
    fig = plt.subplots(1, 1)
    plt.scatter(X_train, Y_train, color="k", marker="x", label="Data")
    plt.legend()
    plt.savefig("data.pdf", transparent=True)
    fig = plt.subplots(1, 1)
    plt.scatter(X_train, Y_train, color="k", marker="x", alpha=0.6, label="Data")
    plt.plot(
        X_test,
        network(X_test).detach().numpy(),
        color="m",
        linestyle="--",
        label=r"$f_{\theta}(\cdot)$",
    )
    plt.savefig("nn.pdf", transparent=True)
    plt.plot(X_test, pred.mean, color="c", label=r"$\mu(\cdot)$")
    plt.fill_between(
        X_test[:, 0],
        (pred.mean - 1.98 * torch.sqrt(pred.var))[:, 0],
        # pred.mean[:, 0],
        (pred.mean + 1.98 * torch.sqrt(pred.var))[:, 0],
        color="c",
        alpha=0.2,
        label=r"$\mu(\cdot) \pm 1.98\sigma$",
    )
    plt.legend()
    plt.savefig("nn2gp.pdf", transparent=True)
    plt.plot(X_train, Y_train)
