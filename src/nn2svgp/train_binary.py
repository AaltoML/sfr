#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import torch.nn as nn
from src.nn2svgp import NTKSVGP
from src.nn2svgp.custom_types import Data


def train(
    ntksvgp: NTKSVGP,
    data: Data,
    num_epochs: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
):
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data),
        batch_size=batch_size,
        # collate_fn=collate_wrapper,
        # pin_memory=True,
    )

    ntksvgp.train()
    optimizer = torch.optim.Adam([{"params": ntksvgp.parameters()}], lr=learning_rate)
    loss_history = []
    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            x, y = batch
            loss = ntksvgp.loss(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.detach().numpy())

            logger.info(
                "Epoch: {} | Batch: {} | Loss: {}".format(epoch_idx, batch_idx, loss)
            )

    ntksvgp.build_dual_svgp()
    return {"loss": loss_history}


if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(45)
    torch.set_default_dtype(torch.float64)

    plot_var = False
    plot_var = True
    save_dir = "./figs/binary"

    torch.set_default_dtype(torch.float64)

    def func(x, noise=True):
        ys, y2s = [], []
        for x_i in x:
            # if x_i > 0.2 and x_i < 1.2:
            if x_i > 0.6 and x_i < 1.2:
                y = 1
                y2 = 0
            else:
                y = 0
                y2 = 1
            ys.append(y)
            y2s.append(y2)
        ys = np.stack(ys, 0).reshape(-1, 1)
        y2s = np.stack(y2s, 0).reshape(-1, 1)
        y = np.concatenate([ys, y2s], -1)
        return torch.Tensor(ys)

    delta = 0.0002
    delta = 0.002
    # delta = 0.005
    # delta = 0.01
    # delta = 1.0
    # delta = 0.00001

    # delta = 1.0
    # network = torch.nn.Sequential(
    #     torch.nn.Linear(1, 64),
    #     # torch.nn.ReLU(),
    #     torch.nn.Sigmoid(),
    #     # torch.nn.Tanh(),
    #     torch.nn.Linear(64, 64),
    #     # torch.nn.Tanh(),
    #     # torch.nn.ReLU(),
    #     torch.nn.Sigmoid(),
    #     torch.nn.Linear(64, 1),
    # )
    class Sin(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    ## takes in a module and applies the specified weight initialization
    def weights_init_normal(m):
        """Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution."""

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find("Linear") != -1:
            y = m.in_features
            # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, 1 / np.sqrt(y))
            # m.bias.data should be 0
            m.bias.data.fill_(0)

    width = 512
    width = 64
    network = torch.nn.Sequential(
        torch.nn.Linear(1, width),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        # torch.nn.Tanh(),
        # torch.nn.Linear(width, width),
        # torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        # Sin(),
        torch.nn.Linear(width, 1),
    )
    network.apply(weights_init_normal)
    print("network: {}".format(network))
    # noise_var = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)

    X_train = torch.rand((100, 1)) * 2
    print("X_train {}".format(X_train.shape))
    # X_train_clipped_1 = X_train[X_train < 1.5].reshape(-1, 1)
    # X_train_clipped_2 = X_train[X_train > 1.9].reshape(-1, 1)
    # print("X_train {}".format(X_train.shape))
    # X_train = torch.concat([X_train_clipped_1, X_train_clipped_2], 0)
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
    X_test = torch.linspace(-0.05, 2.05, 300, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-8, 8, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-2, 2, 100, dtype=torch.float64).reshape(-1, 1)
    print("X_test: {}".format(X_test.shape))
    print("f: {}".format(network(X_test).shape))

    X_new = torch.linspace(-0.5, -0.2, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new = func(X_new, noise=True)
    plt.scatter(X_train, Y_train[:, 0])
    plt.savefig(os.path.join(save_dir, "classification_data.pdf"))

    # X_new_2 = torch.linspace(3.0, 4.0, 20, dtype=torch.float64).reshape(-1, 1)
    # Y_new_2 = func(X_new_2, noise=True)
    X_new_2 = torch.linspace(1.6, 1.8, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new_2 = func(X_new_2, noise=True)

    X_new_3 = torch.linspace(-6.0, -5.0, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new_3 = func(X_new_3, noise=True)

    batch_size = X_train.shape[0]

    likelihood = src.nn2svgp.likelihoods.BernoulliLh()
    # likelihood = src.nn2svgp.likelihoods.CategoricalLh()
    # likelihood = src.nn2svgp.likelihoods.Gaussian()
    prior = src.nn2svgp.priors.Gaussian(params=network.parameters, delta=delta)
    ntksvgp = NTKSVGP(
        network=network,
        # train_data=(X_train, Y_train),
        prior=prior,
        likelihood=likelihood,
        output_dim=1,
        num_inducing=X_train.shape[0],
        # num_inducing=50,
        # jitter=1e-6,
        jitter=1e-4,
    )

    print("setting data")
    ntksvgp.set_data((X_train, Y_train))
    print("FINISHED setting data")
    metrics = train(
        ntksvgp=ntksvgp,
        data=data,
        num_epochs=5000,
        batch_size=batch_size,
        learning_rate=1e-2,
    )

    # f_mean, f_var = ntksvgp.predict_f(X_test_short)
    f_mean, f_var = ntksvgp.predict_f(X_test)
    print("MEAN {}".format(f_mean.shape))
    print("VAR {}".format(f_var.shape))
    print("X_test_short {}".format(X_test_short.shape))
    print(X_test_short.shape)

    ntksvgp.update(x=X_new, y=Y_new)
    f_mean_new, f_var_new = ntksvgp.predict_f(X_test)
    print("MEAN NEW_2 {}".format(f_mean_new.shape))
    print("VAR NEW_2 {}".format(f_var_new.shape))

    ntksvgp.update(x=X_new_2, y=Y_new_2)
    f_mean_new_2, f_var_new_2 = ntksvgp.predict_f(X_test)
    print("MEAN NEW_2 {}".format(f_mean_new_2.shape))
    print("VAR NEW_2 {}".format(f_var_new_2.shape))

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
            network(X_test).detach()[:, 0],
            color="m",
            linestyle="--",
            label=r"$f_{NN,1}(\cdot)$",
        )
        # plt.plot(
        #     X_test[:, 0],
        #     network(X_test).detach()[:, 1],
        #     color="c",
        #     linestyle="--",
        #     label=r"$f_{NN,2}(\cdot)$",
        # )

        # nn_probs = likelihood.inv_link(network(X_test)).detach()
        # print("nn_probs {}".format(nn_probs.shape))
        # plt.plot(
        #     X_test[:, 0],
        #     nn_probs[:, i],
        #     color="m",
        #     linestyle="--",
        #     label=r"$\Pr_{NN}(y=1 \mid x)$",
        # )
        # probs = likelihood.prob(f_mean=f_mean, f_var=f_var)
        # print("probs {}".format(probs.shape))
        # plt.plot(X_test[:, 0], probs[:, i], color="c", label=r"$\Pr(y=1 \mid x)$")
        #
        # plt.plot(X_test_short[:, 0], probs[:, i], color="c", label=r"$\Pr(y=1 \mid x)$")
        plt.plot(X_test[:, 0], f_mean[:, 0], color="m", label=r"$\mu_1(\cdot)$")
        # # plt.plot(X_test[:, 0], f_mean[:, 1], color="c", label=r"$\mu_2(\cdot)$")
        if plot_var:
            plt.fill_between(
                # X_test_short[:, 0],
                X_test[:, 0],
                (f_mean - 1.98 * torch.sqrt(f_var))[:, 0],
                # pred.mean[:, 0],
                (f_mean + 1.98 * torch.sqrt(f_var))[:, 0],
                color="m",
                alpha=0.2,
                label=r"$\mu_1(\cdot) \pm 1.98\sigma_1(\cdot)$",
            )
        plt.scatter(
            ntksvgp.Z,
            torch.ones_like(ntksvgp.Z) * 0.5,
            marker="|",
            color="b",
            label="Z",
        )
        plt.legend()
        # plt.savefig("nn2svgp.pdf", transparent=True)
        plt.savefig(
            os.path.join(save_dir, "nn2svgp-classification" + str(i) + ".pdf"),
            transparent=True,
        )

        plt.scatter(
            X_new, Y_new[:, i], color="m", marker="o", alpha=0.6, label="New data"
        )
        plt.plot(X_test[:, 0], f_mean_new[:, i], color="m", label=r"$\mu_{new}(\cdot)$")
        if plot_var:
            plt.fill_between(
                X_test[:, 0],
                (f_mean_new - 1.98 * torch.sqrt(f_var_new))[:, i],
                # pred.mean[:, 0],
                (f_mean_new + 1.98 * torch.sqrt(f_var_new))[:, i],
                color="m",
                alpha=0.2,
                label=r"$\mu_{new}(\cdot) \pm 1.98\sigma_{new}(\cdot)$",
            )

        plt.scatter(
            X_new_2, Y_new_2[:, i], color="y", marker="o", alpha=0.6, label="New data 2"
        )
        plt.plot(
            X_test[:, 0],
            f_mean_new_2[:, i],
            color="y",
            linestyle="-",
            label=r"$\mu_{new,2}(\cdot)$",
        )
        if plot_var:
            plt.fill_between(
                X_test[:, 0],
                (f_mean_new_2 - 1.98 * torch.sqrt(f_var_new_2))[:, i],
                (f_mean_new_2 + 1.98 * torch.sqrt(f_var_new_2))[:, i],
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
        plt.savefig(
            os.path.join(save_dir, "nn2svgp_new_classification" + str(i) + ".pdf"),
            transparent=True,
        )

    # for i in range(3):
    plot_output(0)
