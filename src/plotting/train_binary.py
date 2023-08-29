#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import torch.nn as nn
from src.sfr import SFR
from src.custom_types import Data


def train(
    sfr: SFR,
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

    sfr.train()
    optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=learning_rate)
    loss_history = []
    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            x, y = batch
            loss = sfr.loss(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.detach().numpy())

            logger.info(
                "Epoch: {} | Batch: {} | Loss: {}".format(epoch_idx, batch_idx, loss)
            )

    sfr.set_data((data[0], data[1]))
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
    plot_update = False
    plot_update = True

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
        return torch.Tensor(ys)[..., 0]

    prior_precision = 0.0006
    prior_precision = 0.006
    prior_precision = 0.00008
    # prior_precision = 0.00002
    # prior_precision = 0.02
    # prior_precision = 0.000002
    # prior_precision = 0.002
    # prior_precision = 0.005
    # prior_precision = 0.01
    # prior_precision = 1.0
    # prior_precision = 0.00001

    # prior_precision = 1.0
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
        torch.nn.Tanh(),
        torch.nn.Linear(width, width),
        # torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        # Sin(),
        torch.nn.Linear(width, 1),
    )

    class Sin(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x)

    network = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 16),
        # torch.nn.Tanh(),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        # torch.nn.Linear(64, 8),
        Sin(),
        # torch.nn.Tanh(),
        # torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
        # torch.nn.Linear(8, 1),
    )

    network.apply(weights_init_normal)
    print("network: {}".format(network))
    # noise_var = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)

    X_train = torch.rand((100, 1)) * 4
    X_train = torch.linspace(0.0, 2.0, 100, dtype=torch.float64).reshape(-1, 1)
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
    # X_test = torch.linspace(-0.05, 2.05, 300, dtype=torch.float64).reshape(-1, 1)
    X_test = torch.linspace(-8, 8, 200, dtype=torch.float64).reshape(-1, 1)
    # X_test = torch.linspace(-2, 2, 100, dtype=torch.float64).reshape(-1, 1)
    print("X_test: {}".format(X_test.shape))
    print("f: {}".format(network(X_test).shape))

    X_new = torch.linspace(-0.5, -0.2, 20, dtype=torch.float64).reshape(-1, 1)
    X_new = torch.linspace(-4.0, -3.0, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new = func(X_new, noise=True)
    Y_new = torch.ones(Y_new.shape)
    plt.scatter(X_train, Y_train)
    plt.savefig(os.path.join(save_dir, "classification_data.pdf"))

    # X_new_2 = torch.linspace(3.0, 4.0, 20, dtype=torch.float64).reshape(-1, 1)
    # Y_new_2 = func(X_new_2, noise=True)
    X_new_2 = torch.linspace(1.6, 1.8, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new_2 = func(X_new_2, noise=True)

    X_new_3 = torch.linspace(-6.0, -5.0, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new_3 = func(X_new_3, noise=True)

    batch_size = X_train.shape[0]

    # likelihood = src.likelihoods.BernoulliLh(EPS=0.01)
    likelihood = src.likelihoods.BernoulliLh(EPS=0.1)
    likelihood = src.likelihoods.BernoulliLh(EPS=0.0005)
    likelihood = src.likelihoods.BernoulliLh(EPS=0.000)
    # likelihood = src.likelihoods.CategoricalLh()
    # likelihood = src.likelihoods.Gaussian()
    prior = src.priors.Gaussian(
        params=network.parameters, prior_precision=prior_precision
    )
    # sfr = src.NN2GPSubset(
    #     network=network,
    #     prior=prior,
    #     likelihood=likelihood,
    #     output_dim=1,
    #     # dual_batch_size=100,
    #     jitter=1e-4,
    # )

    sfr = SFR(
        network=network,
        # train_data=(X_train, Y_train),
        prior=prior,
        likelihood=likelihood,
        output_dim=1,
        # dual_batch_size=None,
        dual_batch_size=100,
        # num_inducing=X_train.shape[0],
        num_inducing=50,
        jitter=1e-10,
        # jitter=1e-6,
        # jitter=0.0,
        # jitter=1e-4,
    )

    # sfr = src.NN2GPSubset(
    #     network=network,
    #     # train_data=(X_train, Y_train),
    #     prior=prior,
    #     likelihood=likelihood,
    #     output_dim=1,
    #     # dual_batch_size=100,
    #     dual_batch_size=None,
    #     # num_inducing=X_train.shape[0],
    #     subset_size=50,
    #     # num_inducing=50,
    #     # jitter=1e-6,
    #     jitter=1e-4,
    # )

    metrics = train(
        sfr=sfr,
        data=data,
        # num_epochs=5000,
        num_epochs=15000,
        batch_size=batch_size,
        learning_rate=1e-3,
    )

    sfr.Z = torch.linspace(-6, 6, sfr.num_inducing)[..., None]
    sfr._build_sfr()

    # def rbf(x1, x2, full_cov=False):
    #     variance = 1.0
    #     k = torch.zeros((x1.shape[0], x2.shape[0]))
    #     print(f"k {k.shape}")
    #     for i, x1_ in enumerate(x1):
    #         for j, x2_ in enumerate(x2):
    #             dist = (x1_ - x2_) ** 2
    #             k[i, j] = torch.exp(-(dist) / variance**2)
    #     return k[None, ...]
    #     # return torch.exp(-(dist).sum() / variance ^ 2)

    # def rbf_single(x1, x2, full_cov=False):
    #     variance = 1.0
    #     k = torch.zeros((x1.shape[0], x2.shape[0]))
    #     print(f"k {k.shape}")
    #     for i, x1_ in enumerate(x1):
    #         for j, x2_ in enumerate(x2):
    #             dist = (x1_ - x2_) ** 2
    #             k[i, j] = torch.exp(-(dist) / variance**2)
    #     return k

    # sfr.kernel = rbf
    # sfr.kernel_single = rbf_single

    # f_mean, f_var = sfr.predict_f(X_test_short)
    f_mean, f_var = sfr.predict_f(X_test)
    print("MEAN {}".format(f_mean.shape))
    print("VAR {}".format(f_var.shape))
    print("X_test_short {}".format(X_test_short.shape))
    print(X_test_short.shape)

    print(f"Y_train {Y_new}")
    print(f"Y_train {Y_train.shape}")
    print(f"Y_new {Y_new}")
    print(f"Y_new {Y_new.shape}")

    # sfr.update(x=X_new_2, y=Y_new_2)
    # f_mean_new_2, f_var_new_2 = sfr.predict_f(X_test)
    # print("MEAN NEW_2 {}".format(f_mean_new_2.shape))
    # print("VAR NEW_2 {}".format(f_var_new_2.shape))

    def plot_output(i):
        fig = plt.subplots(1, 1)
        plt.scatter(X_train, Y_train, color="k", marker="x", alpha=0.6, label="Data")

        # nn_probs = likelihood.inv_link(network(X_test)).detach()
        nn_probs = likelihood(network(X_test))[0].detach()
        print("nn_probs {}".format(nn_probs.shape))
        plt.plot(
            X_test[:, 0],
            nn_probs[:, i],
            color="m",
            linestyle="--",
            label=r"$\Pr_{NN}(y=1 \mid x)$",
        )
        probs = likelihood(f_mean=f_mean, f_var=f_var)[0]
        plt.plot(X_test[:, 0], probs[:, i], color="c", label=r"$\Pr(y=1 \mid x)$")
        plt.legend()
        plt.savefig(
            os.path.join(save_dir, "sfr_classification_probs" + str(i) + ".pdf"),
            transparent=True,
        )

        fig = plt.subplots(1, 1)
        plt.scatter(X_train, Y_train, color="k", marker="x", alpha=0.6, label="Data")
        plt.plot(
            X_test[:, 0],
            func(X_test, noise=False),
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
        # print("y_tilde {}".format(y_tilde.shape))
        # print("X_tset {}".format(X_test.shape))
        # plt.scatter(
        #     X_train[:, 0],
        #     y_tilde[:, 0],
        #     marker="o",
        #     color="r",
        #     label="y_tilde",
        # )
        plt.scatter(
            sfr.Z,
            torch.ones_like(sfr.Z) * -1.0,
            marker="|",
            color="b",
            label="Z",
        )
        plt.legend()
        # plt.savefig("sfr.pdf", transparent=True)
        plt.savefig(
            os.path.join(save_dir, "sfr-classification" + str(i) + ".pdf"),
            transparent=True,
        )

        if plot_update:
            sfr.update(x=X_new, y=Y_new)
            f_mean_new, f_var_new = sfr.predict_f(X_test)
            print("MEAN NEW {}".format(f_mean_new.shape))
            print("VAR NEW {}".format(f_var_new.shape))
            plt.scatter(
                X_new, Y_new, color="m", marker="o", alpha=0.6, label="New data"
            )
            plt.plot(
                X_test[:, 0],
                f_mean_new[:, i],
                color="c",
                linestyle="--",
                label=r"$\mu_{new}(\cdot)$",
            )
            if plot_var:
                plt.fill_between(
                    X_test[:, 0],
                    (f_mean_new - 1.98 * torch.sqrt(f_var_new))[:, i],
                    # pred.mean[:, 0],
                    (f_mean_new + 1.98 * torch.sqrt(f_var_new))[:, i],
                    color="c",
                    alpha=0.2,
                    label=r"$\mu_{new}(\cdot) \pm 1.98\sigma_{new}(\cdot)$",
                )

            # plt.scatter(
            #     X_new_2, Y_new_2[:, i], color="y", marker="o", alpha=0.6, label="New data 2"
            # )
            # plt.plot(
            #     X_test[:, 0],
            #     f_mean_new_2[:, i],
            #     color="y",
            #     linestyle="-",
            #     label=r"$\mu_{new,2}(\cdot)$",
            # )
            # if plot_var:
            #     plt.fill_between(
            #         X_test[:, 0],
            #         (f_mean_new_2 - 1.98 * torch.sqrt(f_var_new_2))[:, i],
            #         (f_mean_new_2 + 1.98 * torch.sqrt(f_var_new_2))[:, i],
            #         color="y",
            #         alpha=0.2,
            #         label=r"$\mu_{new,2}(\cdot) \pm 1.98\sigma_{new,2}(\cdot)$",
            #     )

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
                os.path.join(save_dir, "sfr_new_classification" + str(i) + ".pdf"),
                transparent=True,
            )
            fig = plt.subplots(1, 1)
            plt.scatter(
                X_train, Y_train, color="k", marker="x", alpha=0.6, label="Data"
            )
            plt.scatter(
                X_new, Y_new, color="m", marker="o", alpha=0.6, label="New data"
            )

            # nn_probs = likelihood.inv_link(network(X_test)).detach()
            nn_probs = likelihood(network(X_test))[0].detach()
            print("nn_probs {}".format(nn_probs.shape))
            plt.plot(
                X_test[:, 0],
                nn_probs[:, i],
                color="m",
                linestyle="--",
                label=r"$\Pr_{NN}(y=1 \mid x)$",
            )
            probs_new = likelihood(f_mean=f_mean_new, f_var=f_var_new)[0]
            print("probs_new {}".format(probs_new.shape))
            plt.plot(X_test[:, 0], probs[:, i], color="c", label=r"$\Pr(y=1 \mid x)$")
            plt.plot(
                X_test[:, 0],
                probs_new[:, i],
                color="r",
                label=r"$\Pr_{new}(y=1 \mid x)$",
            )
            plt.legend()
            plt.savefig(
                os.path.join(
                    save_dir, "sfr_new_classification_probs" + str(i) + ".pdf"
                ),
                transparent=True,
            )

    # for i in range(3):
    plot_output(0)
