#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
from experiments.sl.train import checkpoint
from experiments.sl.utils import compute_metrics_regression, EarlyStopper
from src import SFR
from src.custom_types import Data
from torch.utils.data import DataLoader, Dataset


def train(
    sfr: SFR,
    data: Data,
    num_epochs: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
):
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data), batch_size=batch_size
    )

    sfr.train()
    optimizer = torch.optim.Adam(
        # [{"params": sfr.parameters()}],
        [
            {"params": sfr.parameters()},
            {"params": sfr.likelihood.log_sigma_noise},
            # {"params": sfr.prior.prior_precision},
        ],
        lr=learning_rate,
    )

    early_stopper = EarlyStopper(patience=100, min_prior_precision=0)

    loss_history = []
    best_nll = float("inf")
    best_loss = float("inf")
    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            x, y = batch
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

    #         if val_loss < best_loss:
    #             best_ckpt_fname = checkpoint(
    #                 sfr=sfr, optimizer=optimizer, save_dir=run.dir
    #             )
    #             best_loss = val_loss
    #             # wandb.log({"best_test/": test_metrics})
    #             # wandb.log({"best_val/": val_metrics})
    #         if early_stopper(val_loss):  # (val_loss):
    #             logger.info("Early stopping criteria met, stopping training...")
    #             break
    # # Load checkpoint
    # ckpt = torch.load(best_ckpt_fname)
    # print(f"ckpt {ckpt}")
    # print(f"sfr {[p for p in sfr.parameters()]}")
    # sfr.load_state_dict(ckpt["model"])
    # print(f"sfr loaded {[p for p in sfr.parameters()]}")

    torch.set_default_dtype(torch.float64)
    sfr.double()
    sfr.eval()
    sfr.fit(data_loader)
    # sfr.set_data(data)
    return {"loss": loss_history}


if __name__ == "__main__":
    import os

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)
    # torch.set_default_dtype(torch.float)

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
            y1 = f1 + torch.randn(size=(x.shape)) * 0.2
            y2 = f2 + torch.randn(size=(x.shape)) * 0.2
            y3 = f3 + torch.randn(size=(x.shape)) * 0.2
            # y1 = f1 + torch.randn(size=(x.shape)) * 0.1
            # y2 = f2 + torch.randn(size=(x.shape)) * 0.1
            # y3 = f3 + torch.randn(size=(x.shape)) * 0.1
            # y1 = f1 + torch.randn(size=(x.shape)) * 0.0
            # y2 = f2 + torch.randn(size=(x.shape)) * 0.0
            # y3 = f3 + torch.randn(size=(x.shape)) * 0.0
            return torch.stack([y1[:, 0], y2[:, 0], y3[:, 0]], -1)
        else:
            return torch.stack([f1[:, 0], f2[:, 0], f3[:, 0]], -1)

    # prior_precision = 0.00005
    # prior_precision = 0.005
    # prior_precision = 0.00001
    # prior_precision = 0.000005
    prior_precision = 0.001
    prior_precision = 0.002
    # prior_precision = 1.0
    # prior_precision = 0.01
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

    class Sin(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x)

    network = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        # torch.nn.Tanh(),
        # torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        torch.nn.Linear(64, 16),
        Sin(),
        # torch.nn.Tanh(),
        # torch.nn.Tanh(),
        torch.nn.Linear(16, 3),
    )

    print("network: {}".format(network))
    # noise_var = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)

    # X_train = torch.rand((50, 1)) * 2 - 1
    # X_train = torch.rand((50, 1)) * 2
    X_train = torch.rand((100, 1)) * 2
    X_train = torch.rand((200, 1)) * 6
    print("X_train {}".format(X_train.shape))
    X_train_clipped_1 = X_train[X_train < 1.5].reshape(-1, 1)
    X_train_clipped_2 = X_train[X_train > 1.9].reshape(-1, 1)
    # print("X_train {}".format(X_train.shape))
    X_train = torch.concat([X_train_clipped_1, X_train_clipped_2], 0)
    print("X_train {}".format(X_train.shape))
    # X_train = torch.linspace(-1, 1, 50).reshape(-1, 1)
    # Y_train = func(X_train, noise=True)
    Y_train = func(X_train, noise=True)
    # data = (X_train[0:150, :], Y_train[0:150, :])
    data = (X_train, Y_train)

    # split_dataset(dataset=data, random_seed=42, double=True, data_split=[70, 30])

    print("X, Y: {}, {}".format(X_train.shape, Y_train.shape))
    print("X, Y: {}, {}".format(X_train.dtype, Y_train.dtype))
    # X_test = torch.linspace(-1.8, 1.8, 200).reshape(-1, 1)
    X_test_short = torch.linspace(0.0, 2.05, 110).reshape(-1, 1)
    X_test = torch.linspace(-0.2, 2.2, 200).reshape(-1, 1)
    X_test = torch.linspace(-1.0, 2.2, 200).reshape(-1, 1)
    # X_test = torch.linspace(-6.0, 2.2, 200).reshape(-1, 1)
    X_test = torch.linspace(-2.0, 3.5, 300).reshape(-1, 1)
    X_test = torch.linspace(-0.7, 3.5, 300).reshape(-1, 1)
    X_test = torch.linspace(-8, 8, 200).reshape(-1, 1)
    # X_test = torch.linspace(-2, 2, 100).reshape(-1, 1)
    # X_test.float()
    print("X_test: {}".format(X_test.shape))
    # print("f: {}".format(network(X_test).shape))
    X_test_short = X_test
    X_test_short = X_test_short.to(torch.double)
    X_test = X_test.to(torch.double)

    # X_new = torch.linspace(-0.5, -0.2, 20).reshape(-1, 1)
    X_new = torch.linspace(-5.0, -2.0, 20).reshape(-1, 1)
    # X_new.float()
    # X_new.double()
    X_new = X_new.to(torch.double)
    Y_new = func(X_new, noise=True)

    # X_new_2 = torch.linspace(3.0, 4.0, 20).reshape(-1, 1)
    # Y_new_2 = func(X_new_2, noise=True)
    X_new_2 = torch.linspace(1.6, 1.8, 20).reshape(-1, 1)
    # X_new_2.float()
    # X_new_2.double()
    X_new_2 = X_new_2.to(torch.double)
    Y_new_2 = func(X_new_2, noise=True)

    X_new_3 = torch.linspace(-6.0, -5.0, 20).reshape(-1, 1)
    # X_new_3.float()
    # X_new_3.double()
    X_new_3 = X_new_3.to(torch.double)
    Y_new_3 = func(X_new_3, noise=True)

    batch_size = X_train.shape[0]

    num_inducing = 32
    num_inducing = 100
    likelihood = src.likelihoods.Gaussian(sigma_noise=1)
    # likelihood = src.likelihoods.Gaussian(sigma_noise=2)
    # likelihood = src.likelihoods.Gaussian(sigma_noise=0.1)
    likelihood = src.likelihoods.Gaussian(
        sigma_noise=torch.tensor([0.2], requires_grad=True)
    )
    # likelihood = src.likelihoods.Gaussian(sigma_noise=2)
    # likelihood = src.likelihoods.Gaussian(sigma_noise=0.8)
    prior_precision = torch.tensor(prior_precision, requires_grad=False)
    prior = src.priors.Gaussian(
        params=network.parameters, prior_precision=prior_precision
    )
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
        jitter=1e-6,
        # jitter=1e-4,
    )
    sfr.train()
    metrics = train(
        sfr=sfr,
        data=data,
        # num_epochs=2500,
        num_epochs=30000,
        # num_epochs=1,
        batch_size=batch_size,
        learning_rate=1e-2,
    )
    sfr.eval()
    sfr.double()
    print(f"X_test_short {X_test_short.dtype}")
    print(f"SIGMA NOISE: {sfr.likelihood.sigma_noise}")

    sfr.Z = torch.linspace(-6, 6, num_inducing).reshape(-1, 1)
    sfr._build_sfr()

    # ds_train = torch.utils.data.TensorDataset(X_train.double(), Y_train.double())
    ds_test = torch.utils.data.TensorDataset(
        # torch.concat([X_train.double(), X_new.double()], 0),
        # torch.concat([Y_train.double(), Y_new.double()], 0),
        X_new.double(),
        Y_new.double(),
    )
    batch_size = X_train.shape[0]
    device = "cpu"
    train_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    data_float = torch.utils.data.TensorDataset(
        # torch.concat([X_train.float(), X_new.float()], 0),
        # torch.concat([Y_train.float(), Y_new.float()], 0),
        X_new.float(),
        Y_new.float(),
    )
    # data_float = (X_train.to(torch.float), Y_train.to(torch.float))
    train_loader_float = torch.utils.data.DataLoader(data_float, batch_size=batch_size)
    map_metrics = compute_metrics_regression(
        model=sfr.float(), data_loader=train_loader_float, device=device, map=True
    )
    print(f"map_metrics float {map_metrics}")

    # data_double = (X_train.to(torch.double), Y_train.to(torch.double))
    data_double = torch.utils.data.TensorDataset(
        # torch.concat([X_train.double(), X_new.double()], 0),
        # torch.concat([Y_train.double(), Y_new.double()], 0),
        X_new.double(),
        Y_new.double(),
    )
    train_loader_double = torch.utils.data.DataLoader(
        data_double, batch_size=batch_size
    )
    map_metrics_double = compute_metrics_regression(
        model=sfr.double(), data_loader=train_loader_double, device=device, map=True
    )
    print(f"map_metrics double {map_metrics_double}")
    sfr_metrics = compute_metrics_regression(
        model=sfr,
        pred_type="gp",
        # pred_type="nn",
        data_loader=train_loader,
        device=device,
    )
    print(f"sfr_metrics {sfr_metrics}")

    f_mean, f_var = sfr.predict_f(X_test_short)
    print("MEAN {}".format(f_mean.shape))
    print("VAR {}".format(f_var.shape))
    print("X_test_short {}".format(X_test_short.shape))
    print(X_test_short.shape)

    if updates:
        sfr.update(x=X_new, y=Y_new)
        # sfr.update_full(x=X_new, y=Y_new)
        f_mean_new, f_var_new = sfr.predict_f(X_test)
        print("MEAN NEW_2 {}".format(f_mean_new.shape))
        print("VAR NEW_2 {}".format(f_var_new.shape))
        sfr_metrics = compute_metrics_regression(
            model=sfr,
            pred_type="gp",
            data_loader=train_loader,
            device=device,
        )
        print(f"sfr_metrics_update {sfr_metrics}")

        # sfr.update_full(x=X_new_2, y=Y_new_2)
        sfr.update(x=X_new_2, y=Y_new_2)
        f_mean_new_2, f_var_new_2 = sfr.predict_f(X_test)
        print("MEAN NEW_2 {}".format(f_mean_new_2.shape))
        print("VAR NEW_2 {}".format(f_var_new_2.shape))
        sfr_metrics = compute_metrics_regression(
            model=sfr,
            pred_type="gp",
            data_loader=train_loader,
            device=device,
        )
        print(f"sfr_metrics_update_2 {sfr_metrics}")

    import matplotlib.pyplot as plt

    def plot_output(i):
        fig = plt.subplots(1, 1)
        plt.scatter(
            X_train, Y_train[:, i], color="k", marker="x", alpha=0.6, label="Data"
        )
        plt.scatter(
            sfr.Z, torch.ones_like(sfr.Z) * -5, marker="|", color="b", label="Z"
        )
        # plt.plot(
        #     X_test[:, 0],
        #     func(X_test, noise=False)[:, i],
        #     color="b",
        #     label=r"$f_{true}(\cdot)$",
        # )
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
        plt.legend()
        plt.savefig(os.path.join(save_dir, "sfr" + str(i) + ".pdf"), transparent=True)

        if updates:
            fig = plt.subplots(1, 1)
            plt.scatter(
                X_train, Y_train[:, i], color="k", marker="x", alpha=0.6, label="Data"
            )
            plt.scatter(
                sfr.Z, torch.ones_like(sfr.Z) * -5, marker="|", color="b", label="Z"
            )
            # plt.plot(
            #     X_test[:, 0],
            #     func(X_test, noise=False)[:, i],
            #     color="b",
            #     label=r"$f_{true}(\cdot)$",
            # )
            plt.plot(
                X_test[:, 0],
                network(X_test).detach()[:, i],
                color="m",
                linestyle="--",
                label=r"$f_{NN}(\cdot)$",
            )
            plt.scatter(
                X_new,
                Y_new[:, i],
                color="r",
                marker="o",
                alpha=0.6,
                label="New data",
            )
            plt.plot(
                X_test[:, 0],
                f_mean_new[:, i],
                color="c",
                # color="m",
                label=r"$\mu_{new}(\cdot)$",
            )
            if plot_var:
                plt.fill_between(
                    X_test[:, 0],
                    (f_mean_new - 1.98 * torch.sqrt(f_var_new))[:, i],
                    # pred.mean[:, 0],
                    (f_mean_new + 1.98 * torch.sqrt(f_var_new))[:, i],
                    color="c",
                    # color="m",
                    alpha=0.2,
                    label=r"$\mu_{new}(\cdot) \pm 1.98\sigma_{new}(\cdot)$",
                )

            plt.legend()
            plt.savefig(
                os.path.join(save_dir, "sfr_new" + str(i) + ".pdf"),
                transparent=True,
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
                os.path.join(save_dir, "sfr_new_2_" + str(i) + ".pdf"),
                transparent=True,
            )

    for i in range(3):
        plot_output(i)
