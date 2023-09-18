#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import torch.nn as nn
from experiments.sl.utils import compute_metrics
from src import SFR
from src.custom_types import Data


def train(
    sfr: SFR,
    data: Data,
    num_epochs: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
):
    print("data {}".format(data[0].shape))
    print("data {}".format(data[1].shape))
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
            # print("batch {} {}".format(x.shape, y.shape))
            loss = sfr.loss(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.detach().numpy())

            if epoch_idx % 100 == 0:
                logger.info(
                    "Epoch: {} | Batch: {} | Loss: {}".format(
                        epoch_idx, batch_idx, loss
                    )
                )

    print("setting data")
    torch.set_default_dtype(torch.float64)
    sfr.double()
    sfr.eval()
    sfr.fit(data_loader)
    # sfr.set_data((X_train, Y_train))
    print("FINISHED setting data")
    return {"loss": loss_history}


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    # import torch._dynamo as dynamo
    # torch._dynamo.config.verbose = True
    # torch.backends.cudnn.benchmark = True
    # torch._dynamo.config.verbose = True
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    plot_var = False
    plot_var = True
    save_dir = "./figs"
    # plot_updates = False
    plot_updates = True

    torch.set_default_dtype(torch.float64)
    # torch.set_default_dtype(torch.float)

    def func(x, noise=True):
        ys, y2s = [], []
        for x_i in x:
            if x_i > 0.2 and x_i < 1.2:
                y = 1
                y2 = 0
            elif x_i < 0.0 and x_i > -0.3:
                y = 1
                y2 = 0
            else:
                y = 0
                y2 = 1
            if x_i > 1.75:
                y = 1
            ys.append(y)
            y2s.append(y2)
        ys = np.stack(ys, 0)
        # ys = np.stack(ys, 0).reshape(-1, 1)
        # y2s = np.stack(y2s, 0).reshape(-1, 1)
        # y = np.concatenate([ys, y2s], -1)
        return torch.Tensor(ys).long()

    prior_precision = 0.0001
    prior_precision = 0.0002
    prior_precision = 0.001
    # prior_precision = 0.05
    width = 64
    network = torch.nn.Sequential(
        torch.nn.Linear(1, width),
        # torch.nn.ReLU(),
        torch.nn.Tanh(),
        torch.nn.Linear(width, width),
        torch.nn.Tanh(),
        # torch.nn.Sigmoid(),
        torch.nn.Linear(width, 2),
    )
    # network = torch.nn.Sequential(
    #     torch.nn.Linear(1, 64),
    #     # torch.nn.ReLU(),
    #     torch.nn.Sigmoid(),
    #     # torch.nn.Tanh(),
    #     torch.nn.Linear(64, 64),
    #     # torch.nn.Tanh(),
    #     # torch.nn.ReLU(),
    #     torch.nn.Sigmoid(),
    #     torch.nn.Linear(64, 2),
    # )

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
        torch.nn.Linear(16, 2),
        # torch.nn.Linear(8, 1),
    )

    print("network: {}".format(network))
    # noise_var = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)

    X_train = torch.rand((100, 1)) * 2
    print("X_train {}".format(X_train.shape))
    # X_train_clipped_1 = X_train[X_train < 1.5].reshape(-1, 1)
    # X_train_clipped_2 = X_train[X_train > 1.9].reshape(-1, 1)
    # print("X_train {}".format(X_train.shape))
    # X_train = torch.concat([X_train_clipped_1, X_train_clipped_2], 0)
    print("X_train {}".format(X_train.shape))
    # X_train = torch.linspace(-1, 1, 50).reshape(-1, 1)
    # Y_train = func(X_train, noise=True)
    Y_train = func(X_train, noise=True)
    print("Y_train {}".format(Y_train.shape))
    data = (X_train, Y_train)
    print("X, Y: {}, {}".format(X_train.shape, Y_train.shape))
    # X_test = torch.linspace(-1.8, 1.8, 200).reshape(-1, 1)
    X_test_short = torch.linspace(0.0, 2.05, 110).reshape(-1, 1)
    X_test = torch.linspace(-0.2, 2.2, 200).reshape(-1, 1)
    X_test = torch.linspace(-1.0, 2.2, 200).reshape(-1, 1)
    # X_test = torch.linspace(-6.0, 2.2, 200).reshape(-1, 1)
    X_test = torch.linspace(-2.0, 3.5, 300).reshape(-1, 1)
    X_test = torch.linspace(-0.7, 3.5, 300).reshape(-1, 1)
    X_test = X_test.to(torch.double)
    # X_test = torch.linspace(-0.05, 2.2, 300).reshape(-1, 1)
    # X_test = torch.linspace(-8, 8, 200).reshape(-1, 1)
    # X_test = torch.linspace(-2, 2, 100).reshape(-1, 1)
    print("X_test: {}".format(X_test.shape))
    # print("f: {}".format(network(X_test).shape))

    X_new = torch.linspace(-0.5, -0.2, 20).reshape(-1, 1)
    X_new = X_new.to(torch.double)
    Y_new = func(X_new, noise=True)
    # plt.scatter(X_train, Y_train[:, 0])
    plt.scatter(X_train, Y_train)
    plt.savefig(os.path.join(save_dir, "classification_data.pdf"))

    # X_new_2 = torch.linspace(3.0, 4.0, 20).reshape(-1, 1)
    # Y_new_2 = func(X_new_2, noise=True)
    X_new_2 = torch.linspace(1.6, 1.8, 20).reshape(-1, 1)
    X_new_2 = X_new_2.to(torch.double)
    Y_new_2 = func(X_new_2, noise=True)

    X_new_3 = torch.linspace(-6.0, -5.0, 20).reshape(-1, 1)
    X_new_3 = X_new_3.to(torch.double)
    Y_new_3 = func(X_new_3, noise=True)

    batch_size = X_train.shape[0]

    # likelihood = src.likelihoods.BernoulliLh()
    likelihood = src.likelihoods.CategoricalLh()
    likelihood = src.likelihoods.CategoricalLh(EPS=0.01)
    likelihood = src.likelihoods.CategoricalLh(EPS=0.001)
    # likelihood = src.likelihoods.Gaussian()
    prior = src.priors.Gaussian(
        params=network.parameters, prior_precision=prior_precision
    )
    sfr = SFR(
        network=network,
        # train_data=(X_train, Y_train),
        prior=prior,
        likelihood=likelihood,
        output_dim=2,
        # num_inducing=X_train.shape[0],
        # num_inducing=X_train.shape[0] - 10,
        num_inducing=50,
        # num_inducing=30,
        jitter=1e-6,
        # jitter=1e-4,
    )

    metrics = train(
        sfr=sfr,
        data=data,
        num_epochs=10000,
        # num_epochs=3500,
        batch_size=batch_size,
        learning_rate=1e-2,
    )

    data_float = (X_train.to(torch.float), Y_train.to(torch.long))

    @torch.no_grad()
    def map_pred_fn_float(x, idx=None):
        sfr.float()
        f = sfr.network(x)
        return sfr.likelihood.inv_link(f)

    @torch.no_grad()
    def map_pred_fn_double(x, idx=None):
        sfr.double()
        f = sfr.network(x)
        return sfr.likelihood.inv_link(f)

    train_loader_float = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data_float), batch_size=batch_size
    )
    device = "cpu"
    map_metrics = compute_metrics(
        pred_fn=map_pred_fn_float, data_loader=train_loader_float, device=device
    )
    print(f"map_metrics float {map_metrics}")
    data_double = (X_train.to(torch.double), Y_train.to(torch.long))
    train_loader_double = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data_double), batch_size=batch_size
    )
    map_metrics = compute_metrics(
        pred_fn=map_pred_fn_double, data_loader=train_loader_double, device=device
    )
    print(f"map_metrics double {map_metrics}")

    def sfr_pred(
        model: src.SFR,
        pred_type: str = "gp",  # "gp" or "nn"
        num_samples: int = 100,
        device: str = "cuda",
    ):
        @torch.no_grad()
        def pred_fn(x, idx=None):
            return model(
                x.to(device), idx=idx, pred_type=pred_type, num_samples=num_samples
            )[0]

        return pred_fn

    sfr_metrics = compute_metrics(
        pred_fn=sfr_pred(model=sfr, pred_type="gp", num_samples=100, device=device),
        data_loader=train_loader_double,
        device=device,
    )
    print(f"sfr_metrics {sfr_metrics}")

    # f_mean, f_var = sfr.predict_f(X_test_short)
    f_mean, f_var = sfr.predict_f(X_test)
    print("MEAN {}".format(f_mean.shape))
    print("VAR {}".format(f_var.shape))
    print("X_test_short {}".format(X_test_short.shape))
    print(X_test_short.shape)

    if plot_updates:
        sfr.update(x=X_new, y=Y_new)
        f_mean_new, f_var_new = sfr.predict_f(X_test)
        print("MEAN NEW_2 {}".format(f_mean_new.shape))
        print("VAR NEW_2 {}".format(f_var_new.shape))

        sfr_metrics = compute_metrics(
            pred_fn=sfr_pred(model=sfr, pred_type="gp", num_samples=100, device=device),
            data_loader=train_loader_double,
            device=device,
        )
        print(f"sfr_metrics {sfr_metrics}")

        sfr.update(x=X_new_2, y=Y_new_2)
        f_mean_new_2, f_var_new_2 = sfr.predict_f(X_test)
        print("MEAN NEW_2 {}".format(f_mean_new_2.shape))
        print("VAR NEW_2 {}".format(f_var_new_2.shape))

        sfr_metrics = compute_metrics(
            pred_fn=sfr_pred(model=sfr, pred_type="gp", num_samples=100, device=device),
            data_loader=train_loader_double,
            device=device,
        )
        print(f"sfr_metrics {sfr_metrics}")

    def plot_output(i):
        fig = plt.subplots(1, 1)
        plt.scatter(
            X_train,
            Y_train,
            color="k",
            marker="x",
            alpha=0.6,
            label="Data"
            # X_train, Y_train[:, i], color="k", marker="x", alpha=0.6, label="Data"
        )
        plt.plot(
            X_test[:, 0],
            # func(X_test, noise=False)[:, i],
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
        plt.plot(
            X_test[:, 0],
            network(X_test).detach()[:, 1],
            color="c",
            linestyle="--",
            label=r"$f_{NN,2}(\cdot)$",
        )

        plt.plot(X_test[:, 0], f_mean[:, 0], color="m", label=r"$\mu_1(\cdot)$")
        plt.plot(X_test[:, 0], f_mean[:, 1], color="c", label=r"$\mu_2(\cdot)$")
        if plot_var:
            plt.fill_between(
                # X_test_short[:, 0],
                X_test[:, 0],
                (f_mean - 1.96 * torch.sqrt(f_var))[:, 0],
                # pred.mean[:, 0],
                (f_mean + 1.96 * torch.sqrt(f_var))[:, 0],
                color="m",
                alpha=0.2,
                label=r"$\mu_1(\cdot) \pm 1.96\sigma_1(\cdot)$",
            )
            plt.fill_between(
                # X_test_short[:, 0],
                X_test[:, 0],
                (f_mean - 1.96 * torch.sqrt(f_var))[:, 1],
                # pred.mean[:, 0],
                (f_mean + 1.96 * torch.sqrt(f_var))[:, 1],
                color="c",
                alpha=0.2,
                label=r"$\mu_2(\cdot) \pm 1.96\sigma_2(\cdot)$",
            )
        plt.scatter(
            sfr.Z,
            torch.ones_like(sfr.Z) * 0.5,
            marker="|",
            color="b",
            label="Z",
        )
        plt.legend()
        plt.savefig(
            os.path.join(save_dir, "sfr-classification" + str(i) + ".pdf"),
            transparent=True,
        )

        if plot_updates:
            plt.scatter(
                X_new, Y_new, color="m", marker="o", alpha=0.6, label="New data"
            )
            plt.plot(
                X_test[:, 0], f_mean_new[:, i], color="y", label=r"$\mu_{new}(\cdot)$"
            )
            if plot_var:
                plt.fill_between(
                    X_test[:, 0],
                    (f_mean_new - 1.96 * torch.sqrt(f_var_new))[:, i],
                    # pred.mean[:, 0],
                    (f_mean_new + 1.96 * torch.sqrt(f_var_new))[:, i],
                    color="y",
                    alpha=0.2,
                    label=r"$\mu_{new}(\cdot) \pm 1.96\sigma_{new}(\cdot)$",
                )

            # plt.scatter(
            #     X_new_2,
            #     Y_new_2,
            #     color="y",
            #     marker="o",
            #     alpha=0.6,
            #     label="New data 2"
            #     # X_new_2, Y_new_2[:, i], color="y", marker="o", alpha=0.6, label="New data 2"
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
            #         (f_mean_new_2 - 1.96 * torch.sqrt(f_var_new_2))[:, i],
            #         (f_mean_new_2 + 1.96 * torch.sqrt(f_var_new_2))[:, i],
            #         color="y",
            #         alpha=0.2,
            #         label=r"$\mu_{new,2}(\cdot) \pm 1.96\sigma_{new,2}(\cdot)$",
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
            X_train,
            Y_train,
            color="k",
            marker="x",
            alpha=0.6,
            label="Data"
            # X_train, Y_train[:, i], color="k", marker="x", alpha=0.6, label="Data"
        )
        plt.plot(
            X_test[:, 0],
            # func(X_test, noise=False)[:, i],
            func(X_test, noise=False),
            color="b",
            label=r"$f_{true}(\cdot)$",
        )
        plt.scatter(X_new, Y_new, color="y", marker="o", alpha=0.6, label="New data")

        # plt.plot(X_test[:, 0], f_mean[:, 0], color="m", label=r"$\mu_1(\cdot)$")
        # plt.plot(X_test[:, 0], f_mean[:, 1], color="c", label=r"$\mu_2(\cdot)$")

        probs = likelihood(f_mean=f_mean, f_var=f_var)[0]
        print("probs {}".format(probs.shape))
        plt.plot(X_test[:, 0], probs[:, i], color="c", label=r"$\Pr(y=1 \mid x)$")
        # plt.plot(X_test_short[:, 0], probs[:, i], color="c", label=r"$\Pr(y=1 \mid x)$")
        plt.legend()
        plt.savefig(
            os.path.join(save_dir, "sfr_classification_probs_" + str(i) + ".pdf"),
            transparent=True,
        )
        if plot_updates:
            # fig = plt.subplots(1, 1)
            probs = likelihood(f_mean=f_mean_new, f_var=f_var_new)[0]
            print("probs {}".format(probs.shape))
            plt.plot(
                X_test[:, 0],
                probs[:, i],
                color="r",
                label=r"$\Pr_{new}(y=1 \mid x)$",
                linestyle="--",
            )
            # plt.plot(X_test_short[:, 0], probs[:, i], color="c", label=r"$\Pr(y=1 \mid x)$")
            plt.legend()
            plt.savefig(
                os.path.join(
                    save_dir, "sfr_new_classification_probs_" + str(i) + ".pdf"
                ),
                transparent=True,
            )

    # for i in range(3):
    plot_output(0)
