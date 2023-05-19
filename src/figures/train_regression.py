#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
from src import SFR
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

    sfr.set_data(data)
    return {"loss": loss_history}


if __name__ == "__main__":
    import os

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    updates = False
    updates = True
    plot_var = False
    plot_var = True
    save_dir = "figs"

    def func(x, noise=True):
        # x = x + 1e-6
        f1 = torch.sin(x * 5) / x + torch.cos(
            x * 10
        )  # + x**2 +np.log(5*x + 0.00001)  + 0.5
        f2 = torch.sin(x) / x + torch.cos(x)  # + x**2 +np.log(5*x + 0.00001)  + 0.5
        f3 = torch.cos(x * 5) + torch.sin(x * 1)  # + x**2 +np.log(5*x + 0.00001)  + 0.5
        if noise == True:
            # y1 = f1 + torch.randn(size=(x.shape)) * 0.2
            # y2 = f2 + torch.randn(size=(x.shape)) * 0.2
            # y3 = f3 + torch.randn(size=(x.shape)) * 0.2
            # y1 = f1 + torch.randn(size=(x.shape)) * 0.1
            # y2 = f2 + torch.randn(size=(x.shape)) * 0.1
            # y3 = f3 + torch.randn(size=(x.shape)) * 0.1
            y1 = f1 + torch.randn(size=(x.shape)) * 0.0
            y2 = f2 + torch.randn(size=(x.shape)) * 0.0
            y3 = f3 + torch.randn(size=(x.shape)) * 0.0
            return torch.stack([y1[:, 0], y2[:, 0], y3[:, 0]], -1)
        else:
            return torch.stack([f1[:, 0], f2[:, 0], f3[:, 0]], -1)

    delta = 0.00005
    # delta = 0.005
    # delta = 0.00001
    # delta = 0.000005
    # delta = 0.001
    # delta = 1.0
    # delta = 0.01
    # network = torch.nn.Sequential(
    #     torch.nn.Linear(1, 64),
    #     # torch.nn.Sigmoid(),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(64, 3),
    # )
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        torch.nn.Linear(64, 3),
    )
    print("network: {}".format(network))
    # noise_var = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)

    # X_train = torch.rand((50, 1)) * 2 - 1
    # X_train = torch.rand((50, 1)) * 2
    X_train = torch.rand((100, 1)) * 2
    X_train = torch.rand((200, 1)) * 2
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
    X_new = torch.linspace(-5.0, -2.0, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new = func(X_new, noise=True)

    # X_new_2 = torch.linspace(3.0, 4.0, 20, dtype=torch.float64).reshape(-1, 1)
    # Y_new_2 = func(X_new_2, noise=True)
    X_new_2 = torch.linspace(1.6, 1.8, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new_2 = func(X_new_2, noise=True)

    X_new_3 = torch.linspace(-6.0, -5.0, 20, dtype=torch.float64).reshape(-1, 1)
    Y_new_3 = func(X_new_3, noise=True)

    batch_size = X_train.shape[0]

    num_inducing = 32
    # likelihood = src.likelihoods.Gaussian(sigma_noise=1)
    likelihood = src.likelihoods.Gaussian(sigma_noise=2)
    # likelihood = src.likelihoods.Gaussian(sigma_noise=0.1)
    # likelihood = src.likelihoods.Gaussian(sigma_noise=2)
    # likelihood = src.likelihoods.Gaussian(sigma_noise=0.8)
    prior = src.priors.Gaussian(params=network.parameters, delta=delta)
    sfr = SFR(
        network=network,
        # train_data=(X_train, Y_train),
        prior=prior,
        likelihood=likelihood,
        output_dim=3,
        # num_inducing=500,
        num_inducing=num_inducing,
        dual_batch_size=None,
        # dual_batch_size=32,
        # num_inducing=20,
        # jitter=1e-6,
        jitter=1e-4,
    )
    metrics = train(
        sfr=sfr,
        data=data,
        num_epochs=2500,
        # num_epochs=1,
        batch_size=batch_size,
        learning_rate=1e-2,
    )

    f_mean, f_var = sfr.predict_f(X_test_short)
    print("MEAN {}".format(f_mean.shape))
    print("VAR {}".format(f_var.shape))
    print("X_test_short {}".format(X_test_short.shape))
    print(X_test_short.shape)

    if updates:
        sfr.update(x=X_new, y=Y_new)
        f_mean_new, f_var_new = sfr.predict_f(X_test)
        print("MEAN NEW_2 {}".format(f_mean_new.shape))
        print("VAR NEW_2 {}".format(f_var_new.shape))

        sfr.update(x=X_new_2, y=Y_new_2)
        f_mean_new_2, f_var_new_2 = sfr.predict_f(X_test)
        print("MEAN NEW_2 {}".format(f_mean_new_2.shape))
        print("VAR NEW_2 {}".format(f_var_new_2.shape))

    import matplotlib.pyplot as plt

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

        plt.plot(X_test_short[:, 0], f_mean[:, i], color="c", label=r"$\mu(\cdot)$")
        if plot_var:
            plt.fill_between(
                X_test_short[:, 0],
                (f_mean - 1.98 * torch.sqrt(f_var))[:, i],
                # pred.mean[:, 0],
                (f_mean + 1.98 * torch.sqrt(f_var))[:, i],
                color="c",
                alpha=0.2,
                label=r"$\mu(\cdot) \pm 1.98\sigma(\cdot)$",
            )
        plt.scatter(
            sfr.Z, torch.ones_like(sfr.Z) * -5, marker="|", color="b", label="Z"
        )
        plt.legend()
        plt.savefig(os.path.join(save_dir, "sfr" + str(i) + ".pdf"), transparent=True)

        if updates:
            plt.scatter(
                X_new, Y_new[:, i], color="m", marker="o", alpha=0.6, label="New data"
            )
            plt.plot(
                X_test[:, 0], f_mean_new[:, i], color="m", label=r"$\mu_{new}(\cdot)$"
            )
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
                X_new_2,
                Y_new_2[:, i],
                color="y",
                marker="o",
                alpha=0.6,
                label="New data 2",
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
                os.path.join(save_dir, "sfr_new" + str(i) + ".pdf"),
                transparent=True,
            )

    for i in range(3):
        plot_output(i)
