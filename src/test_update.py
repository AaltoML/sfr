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


@hydra.main(version_base="1.3", config_path="../configs", config_name="svgp")
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
        f = (
            torch.sin(x * 5) / x + torch.cos(x * 10) + 3
        )  # + x**2 +np.log(5*x + 0.00001)  + 0.5
        if noise == True:
            y = f + torch.randn(size=(x.shape)) * 0.2
            return y[:, 0]
        else:
            return f[:, 0]

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

    Z = torch.linspace(
        0.0,
        7.5,
        svgp.variational_strategy.inducing_points.shape[0],
        dtype=torch.float64,
        # )
    ).reshape(-1, 1)
    svgp.variational_strategy.inducing_points = Z

    # X_new = torch.linspace(5, 6, 10, dtype=torch.float64).reshape(-1, 1)
    # X_new = torch.linspace(2, 3.5, 10, dtype=torch.float64).reshape(-1, 1)
    # X_new = torch.linspace(4.0, 5.0, 10, dtype=torch.float64).reshape(-1, 1)
    X_new = torch.linspace(2.0, 2.5, 10, dtype=torch.float64).reshape(-1, 1)
    X_new_2 = torch.linspace(5.0, 6.0, 20, dtype=torch.float64).reshape(-1, 1)
    # X_new = torch.linspace(2, 3.5, 100, dtype=torch.float64)
    Y_new = func(X_new, noise=True)
    Y_new_2 = func(X_new_2, noise=True)
    data_new = (X_new, Y_new)
    data_new_2 = (X_new_2, Y_new_2)
    print("X_new, Y_new: {}, {}".format(X_new.shape, Y_new.shape))

    # X_test = torch.linspace(-0.2, 2.2, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-8, 8, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-2, 2, 100, dtype=torch.float64).reshape(-1, 1)
    print("X_test: {}".format(X_test.shape))
    print("f: {}".format(svgp(X_test)))
    # print("y: {}".format(likelihood(svgp(X_test))))

    batch_size = X_train.shape[0]
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
    num_epochs = 500
    print(svgp.likelihood.noise)
    src.models.svgp.train(
        svgp=svgp,
        # likelihood=likelihood,
        learning_rate=0.01,
        num_data=X_train.shape[0],
        # early_stopper=early_stopper,
    )(data_loader, num_epochs)
    print(svgp.likelihood.noise)

    import matplotlib.pyplot as plt

    # fig = plt.subplots(1, 1)
    # plt.plot(np.arange(len(metrics["loss"])), metrics["loss"])
    # plt.savefig("loss.pdf", transparent=True)
    fig = plt.subplots(1, 1)
    plt.scatter(X_train, Y_train, color="k", marker="x", label="Data")
    # plt.scatter(Z, torch.ones_like(Z) * -2.5, color="k", marker="|", label="Z")
    plt.legend()
    plt.savefig("data_gp.pdf", transparent=True)

    # mean, var, noise_var = predict(X_test, data_new=None)
    # svgp.update(data_new=(X_train, Y_train))
    # mean, var, noise_var = svgp.predict(X_test)
    f_dist = svgp(X_test)
    mean_g = f_dist.mean
    var_g = f_dist.variance
    # mean, var, noise_var = svgp.predict(X_test, full_cov=True, white=True)
    # mean, var, noise_var = svgp.predict(X_test, white=True)
    #
    # svgp.set_dual()
    # mean, var, noise_var = svgp.predict(X_test, white=False)
    mean, var, noise_var = svgp.predict(X_test, white=True)
    # pred = likelihood(svgp(X_test))
    print("pred mean {}".format(mean.shape))
    print("pred var {}".format(var.shape))
    # var = torch.diag(var)
    print("pred var {}".format(var.shape))

    fig, ax = plt.subplots(1, 1)
    plt.scatter(X_train[:, 0], Y_train, color="k", marker="x", alpha=0.6, label="Data")
    plt.scatter(
        Z[:, 0],
        torch.ones_like(Z[:, 0]) * -2.3,
        color="b",
        marker="|",
        label="Z"
        # Z[:, 0], 0 * torch.ones_like(Z[:, 0]) * -2.5, color="k", marker="x", label="Z"
    )
    plt.plot(
        X_test[:, 0],
        mean.detach().numpy(),
        color="c",
        label=r"$\mu_{old}(\cdot)$",
    )
    print("mean.detach() {}".format(mean.detach().shape))
    plt.fill_between(
        X_test[:, 0],
        mean.detach() - 1.98 * torch.sqrt(var.detach()),
        mean.detach() + 1.98 * torch.sqrt(var.detach()),
        color="c",
        alpha=0.2,
        label=r"$\mu_{old}(\cdot) \pm 1.98\sigma_{old}(\cdot)$",
    )
    plt.savefig("test_pred.pdf")

    def plot(save_name="gp.pdf", data_new=None):
        fig, ax = plt.subplots(1, 1)
        plt.scatter(
            X_train[:, 0], Y_train, color="k", marker="x", alpha=0.6, label="Data"
        )
        plt.scatter(
            Z[:, 0],
            torch.ones_like(Z[:, 0]) * -2.3,
            color="b",
            marker="|",
            label="Z"
            # Z[:, 0], 0 * torch.ones_like(Z[:, 0]) * -2.5, color="k", marker="x", label="Z"
        )
        plt.plot(
            X_test[:, 0],
            mean.detach().numpy(),
            color="c",
            label=r"$\mu_{old}(\cdot)$",
        )
        print("mean.detach() {}".format(mean.detach().shape))
        plt.fill_between(
            X_test[:, 0],
            mean.detach() - 1.98 * torch.sqrt(var.detach()),
            mean.detach() + 1.98 * torch.sqrt(var.detach()),
            color="c",
            alpha=0.2,
            label=r"$\mu_{old}(\cdot) \pm 1.98\sigma_{old}(\cdot)$",
        )

        # plt.plot(
        #     X_test[:, 0],
        #     mean_g.detach().numpy(),
        #     color="r",
        #     label=r"$\mu_{g}(\cdot)$",
        #     linestyle="--",
        # )
        # print("mean_g.detach() {}".format(mean_g.detach().shape))
        # plt.fill_between(
        #     X_test[:, 0],
        #     mean_g.detach() - 1.98 * torch.sqrt(var_g.detach()),
        #     mean_g.detach() + 1.98 * torch.sqrt(var_g.detach()),
        #     color="r",
        #     linestyle="--",
        #     alpha=0.2,
        #     label=r"$\mu_{old}(\cdot) \pm 1.98\sigma_{g}(\cdot)$",
        # )

        # meanp_new, meanZ, varZ = predict(X_test, data_new=data_new)
        # pred = predict(X_test, data_new=data_new)
        plt.scatter(Z, torch.ones_like(Z) * -2.5, color="k", marker="|", label="Z")
        plt.scatter(
            data_new[0], data_new[1], color="r", marker="o", alpha=0.6, label="New data"
        )
        plt.plot(
            X_test[:, 0],
            mean_new.detach().numpy(),
            color="m",
            label=r"$\mu_{new}(\cdot)$",
        )
        plt.fill_between(
            X_test.squeeze(),
            mean_new.squeeze() - 1.98 * torch.sqrt(var_new),
            # pred.mean[:, 0],
            mean_new.squeeze() + 1.98 * torch.sqrt(var_new),
            color="m",
            alpha=0.2,
            label=r"$\mu_{new}(\cdot) \pm 1.98\sigma_{new}(\cdot)$",
        )

        # mean, var, noise_var = predict(X_test, data_new=data_new)
        return fig

    svgp.update(data_new=data_new)
    mean_new, var_new, noise_var = svgp.predict(X_test, white=False)
    print("pred mean_new {}".format(mean_new.shape))
    print("pred var_new {}".format(var_new.shape))
    fig = plot("gp_new_1.pdf", data_new=data_new)
    fig.legend()
    fig.savefig("gp_new_1.pdf", transparent=True)

    # svgp.update(data_new=data_new)
    # svgp.update(data_new=data_new)
    svgp.update(data_new=data_new_2)
    mean_new_2, var_new_2, noise_var = svgp.predict(X_test, white=False)
    print("pred mean_new {}".format(mean_new_2.shape))
    print("pred var_new {}".format(var_new_2.shape))
    plt.scatter(
        data_new_2[0], data_new_2[1], color="r", marker="o", alpha=0.6, label="New data"
    )
    plt.plot(
        X_test[:, 0],
        mean_new_2.detach().numpy(),
        color="b",
        label=r"$\mu_{new}(\cdot)$",
    )
    plt.fill_between(
        X_test.squeeze(),
        mean_new_2.squeeze() - 1.98 * torch.sqrt(var_new_2),
        # pred.mean[:, 0],
        mean_new_2.squeeze() + 1.98 * torch.sqrt(var_new_2),
        color="b",
        alpha=0.2,
        label=r"$\mu_{new,2}(\cdot) \pm 1.98\sigma_{new,2}(\cdot)$",
    )
    fig.legend()
    fig.savefig("gp_new_2.pdf", transparent=True)
    # plot("gp_new_2.pdf", data_new=data_new_2)


if __name__ == "__main__":
    train()  # pyright: ignore
