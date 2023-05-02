#!/usr/bin/env python3
import logging
import random
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import numpy as np
import omegaconf
import torch


torch.set_default_dtype(torch.float64)
import torchrl
import utils
import wandb
from dm_env import specs, StepType
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
import src

# from src.utils.buffer import ReplayBuffer
from utils import EarlyStopper, set_seed_everywhere


@hydra.main(version_base="1.3", config_path="../configs", config_name="svgp_mo")
def train(cfg: DictConfig):
    set_seed_everywhere(42)
    # try:  # Make experiment reproducible
    #     set_seed_everywhere(cfg.random_seed)
    # except:
    #     random_seed = random.randint(0, 10000)
    #     set_seed_everywhere(random_seed)
    # set_seed_everywhere(cfg.random_seed)

    # cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    svgp = hydra.utils.instantiate(cfg.svgp)
    # likelihood = hydra.utils.instantiate(cfg.likelihood)

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

    # X_train = torch.rand((50, 1)) * 2 - 1
    # X_train = torch.rand((50, 1)) * 2
    X_train = torch.rand((100, 1)) * 2
    # X_train = torch.rand((15, 1)) * 2
    # X_train = torch.linspace(-1, 1, 50, dtype=torch.float64).reshape(-1, 1)
    Y_train = func(X_train, noise=True)
    # Y_train = func(X_train, noise=False)
    data = (X_train, Y_train)
    print("X, Y: {}, {}".format(X_train.shape, Y_train.shape))
    # X_test = torch.linspace(-1.8, 1.8, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(0.0, 2.5, 110, dtype=torch.float64)
    # X_test = torch.linspace(0.0, 7.5, 110, dtype=torch.float64)
    X_test = torch.linspace(0.0, 7.5, 110, dtype=torch.float64).reshape(-1, 1)

    # Z = torch.linspace(
    #     0.0,
    #     7.5,
    #     svgp.variational_strategy.base_variational_strategy.inducing_points.shape[-2],
    #     dtype=torch.float64,
    #     # )
    # ).reshape(-1, 1)
    Z1 = torch.linspace(
        0.0,
        # 4.0,
        # 2.5,
        7.5,
        svgp.variational_strategy.base_variational_strategy.inducing_points.shape[-2],
        dtype=torch.float64,
    ).reshape(1, -1, 1)
    Z2 = torch.linspace(
        0.0,
        # 4.0,
        # 2.5,
        7.5,
        svgp.variational_strategy.base_variational_strategy.inducing_points.shape[-2],
        dtype=torch.float64,
    ).reshape(1, -1, 1)
    Z3 = torch.linspace(
        0.0,
        # 4.0,
        # 2.5,
        7.5,
        svgp.variational_strategy.base_variational_strategy.inducing_points.shape[-2],
        dtype=torch.float64,
    ).reshape(1, -1, 1)
    Z = torch.concat([Z1, Z2, Z3], 0)
    Z = Z1
    print("Z: {}".format(Z.shape))
    svgp.variational_strategy.base_variational_strategy.inducing_points = Z

    # X_new = torch.linspace(5, 6, 10, dtype=torch.float64).reshape(-1, 1)
    # X_new = torch.linspace(2, 3.5, 10, dtype=torch.float64).reshape(-1, 1)
    X_new = torch.linspace(4.0, 5.0, 30, dtype=torch.float64).reshape(-1, 1)
    X_new_2 = torch.linspace(6.0, 6.5, 30, dtype=torch.float64).reshape(-1, 1)
    # X_new = torch.linspace(2.0, 2.5, 10, dtype=torch.float64).reshape(-1, 1)
    # X_new = torch.linspace(2, 3.5, 100, dtype=torch.float64)
    Y_new = func(X_new, noise=True)
    data_new = (X_new, Y_new)
    Y_new_2 = func(X_new_2, noise=True)
    data_new_2 = (X_new_2, Y_new_2)
    print("X_new, Y_new: {}, {}".format(X_new.shape, Y_new.shape))

    # X_test = torch.linspace(-0.2, 2.2, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-8, 8, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-2, 2, 100, dtype=torch.float64).reshape(-1, 1)
    print("X_test: {}".format(X_test.shape))
    print("f: {}".format(svgp(X_test)))
    # print("y: {}".format(likelihood(svgp(X_test))))

    batch_size = X_train.shape[0]
    batch_size = 64
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data),
        batch_size=batch_size,
        # collate_fn=collate_wrapper,
        # pin_memory=True,
    )

    # predict = src.models.svgp.predict(
    #     svgp=svgp,
    #     likelihood=likelihood,
    # )
    num_epochs = 1000
    # num_epochs = 1
    print(svgp.likelihood.noise)
    src.models.svgp.train(
        svgp=svgp,
        # likelihood=svgp.likelihood,
        learning_rate=0.1,
        num_data=X_train.shape[0],
        # early_stopper=early_stopper,
    )(data_loader, num_epochs)
    print(svgp.likelihood.noise)

    import matplotlib.pyplot as plt

    # plt.scatter(X_train[:, 0], Y_train, color="k", marker="x", alpha=0.6, label="Data")
    # plt.scatter(
    #     Z[:, 0],
    #     torch.ones_like(Z[:, 0]) * -2.3,
    #     color="b",
    #     marker="|",
    #     label="Z"
    #     # Z[:, 0], 0 * torch.ones_like(Z[:, 0]) * -2.5, color="k", marker="x", label="Z"
    # )
    # svgp.update(data_new=data_new)
    # mean, var, noise_var = svgp.predict(X_test)
    mean, var, noise_var = svgp.predict(X_test)
    # pred = likelihood(svgp(X_test))
    print("pred mean {}".format(mean.shape))
    print("pred var {}".format(var.shape))
    # plt.plot(
    #     X_test[:, 0], mean.detach().numpy(), color="c", label=r"$\mu_{old}(\cdot)$"
    # )
    # plt.fill_between(
    #     X_test[:, 0],
    #     (mean.detach() - 1.98 * torch.sqrt(var.detach())),
    #     # pred.mean[:, 0],
    #     (mean.detach() + 1.98 * torch.sqrt(var.detach())),
    #     color="c",
    #     alpha=0.2,
    #     label=r"$\mu_{old}(\cdot) \pm 1.98\sigma_{old}$",
    # )

    # meanp_new, meanZ, varZ = predict(X_test, data_new=data_new)
    # pred = predict(X_test, data_new=data_new)
    svgp.update(data_new=data_new)
    mean_new, var_new, noise_var = svgp.predict(X_test)
    # pred = likelihood(svgp(X_test))

    svgp.update(data_new=data_new_2)
    mean_new_2, var_new_2, noise_var = svgp.predict(X_test)

    # plt.plot(Z[:, 0], meanZ)
    print("pred mean_new {}".format(mean_new.shape))
    print("pred var_new {}".format(var_new.shape))
    print("X_train {}".format(X_train.shape))
    print("Y_train {}".format(Y_train.shape))

    def plot(i):
        plt.scatter(
            # Z[i, :, 0], # TODO uncomment this when using Z for each output_dim
            # torch.ones_like(Z[i, :, 0]) * -2.5,
            Z[0, :, 0],
            torch.ones_like(Z[0, :, 0]) * -2.5,
            color="k",
            marker="|",
            label="Z",
        )
        plt.scatter(
            X_train[:, 0],
            Y_train[:, i],
            color="k",
            marker="x",
            alpha=0.6,
            label="Old data",
        )
        plt.scatter(
            X_new, Y_new[:, i], color="c", marker="o", alpha=0.6, label="New data"
        )
        plt.scatter(
            X_new_2, Y_new_2[:, i], color="r", marker="o", alpha=0.6, label="New data"
        )

        plt.plot(
            X_test[:, 0],
            mean.detach()[:, i],
            color="m",
            label=r"$\mu_{old}(\cdot)$",
        )
        plt.fill_between(
            X_test[:, 0],
            mean[:, i] - 1.98 * torch.sqrt(var[:, i]),
            # pred.mean[:, 0],
            mean[:, i] + 1.98 * torch.sqrt(var[:, i]),
            color="m",
            alpha=0.2,
            label=r"$\mu_{old}(\cdot) \pm 1.98\sigma_{old}(\cdot)$",
        )

        plt.plot(
            X_test[:, 0],
            mean_new.detach()[:, i],
            color="c",
            label=r"$\mu_{new}(\cdot)$",
        )
        plt.fill_between(
            X_test[:, 0],
            mean_new[:, i] - 1.98 * torch.sqrt(var_new[:, i]),
            # pred.mean[:, 0],
            mean_new[:, i] + 1.98 * torch.sqrt(var_new[:, i]),
            color="c",
            alpha=0.2,
            label=r"$\mu_{new}(\cdot) \pm 1.98\sigma_{new}(\cdot)$",
        )

        plt.plot(
            X_test[:, 0],
            mean_new_2.detach()[:, i],
            color="r",
            label=r"$\mu_{2}(\cdot)$",
        )
        plt.fill_between(
            X_test[:, 0],
            mean_new_2[:, i] - 1.98 * torch.sqrt(var_new_2[:, i]),
            # pred.mean[:, 0],
            mean_new_2[:, i] + 1.98 * torch.sqrt(var_new_2[:, i]),
            color="r",
            alpha=0.2,
            label=r"$\mu_{2}(\cdot) \pm 1.98\sigma_{2}(\cdot)$",
        )

        # mean, var, noise_var = predict(X_test, data_new=data_new)
        plt.legend()
        plt.savefig("mo_gp" + str(i) + ".pdf", transparent=True)

    for i in [0, 1, 2]:
        plt.figure()
        plot(i)


if __name__ == "__main__":
    train()  # pyright: ignore
