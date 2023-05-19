## REGRESSION EXAMPLE IN 1D FOR THE PAPER

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

            if epoch_idx % 500 == 0:
                logger.info(
                    "Epoch: {} | Batch: {} | Loss: {}".format(
                        epoch_idx, batch_idx, loss
                    )
                )

    sfr.set_data(data)
    return {"loss": loss_history}


if __name__ == "__main__":
    import os

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
            y1 = f1 + torch.randn(size=(x.shape)) * 0.5
            y2 = f2 + torch.randn(size=(x.shape)) * 0.5
            y3 = f3 + torch.randn(size=(x.shape)) * 0.5
            return torch.stack([y1[:, 0], y2[:, 0], y3[:, 0]], -1)
        else:
            return torch.stack([f1[:, 0], f2[:, 0], f3[:, 0]], -1)

    # delta = 0.00005 # works with sigmoid
    delta = 0.0001  # works with tanh

    network = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        # torch.nn.ReLU(),
        # torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 3),
    )
    print("network: {}".format(network))
    # noise_var = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)

    X_train = torch.rand((200, 1)) * 2
    # print("X_train {}".format(X_train.shape))
    X_train_clipped_1 = X_train[X_train < 1.5].reshape(-1, 1)
    X_train_clipped_2 = X_train[X_train > 1.9].reshape(-1, 1)
    # print("X_train {}".format(X_train.shape))
    X_train = torch.concat([X_train_clipped_1, X_train_clipped_2], 0)
    print("X_train {}".format(X_train.shape))
    # X_train = torch.linspace(-1, 1, 50, dtype=torch.float64).reshape(-1, 1)

    Y_train = func(X_train, noise=True)
    data = (X_train, Y_train)
    print("X, Y: {}, {}".format(X_train.shape, Y_train.shape))

    batch_size = X_train.shape[0]

    likelihood = src.likelihoods.Gaussian(sigma_noise=0.5)
    prior = src.priors.Gaussian(params=network.parameters, delta=delta)
    sfr = SFR(
        network=network,
        # train_data=(X_train, Y_train),
        prior=prior,
        likelihood=likelihood,
        output_dim=3,
        # num_inducing=500,
        num_inducing=30,
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


import matplotlib.pyplot as plt
import tikzplotlib


# Set up plotting
def plot(
    sfr,
    ax=None,
    data=None,
    data_dimmed=None,
    network=None,
    showSVGP=True,
    save=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Test grid
    X_test = torch.linspace(-0.2, 2.2, 200, dtype=torch.float64).reshape(-1, 1)

    # Show data
    if data is not None:
        X_train, Y_train = data
        # ax.scatter(X_train[:,0], Y_train[:,0], s=30, alpha=.5, color='k', marker='+', linewidth=2,label='Data')
        ax.plot(
            X_train[:, 0],
            Y_train[:, 0],
            marker="+",
            color="k",
            markersize=3,
            alpha=0.5,
            linewidth=1.5,
            linestyle="none",
            label="Data",
            zorder=0,
        )

    # Show data (dimmed)
    if data_dimmed is not None:
        X_train, Y_train = data_dimmed
        # ax.scatter(X_train[:,0], Y_train[:,0], s=30, alpha=.2, color='k', marker='+')
        ax.plot(
            X_train[:, 0],
            Y_train[:, 0],
            marker="+",
            color="k",
            markersize=3,
            alpha=0.15,
            linewidth=1.5,
            linestyle="none",
        )

    # Visualize iducing points
    if hasattr(sfr, "Z") and showSVGP == True:
        ax.scatter(
            sfr.Z.numpy()[:, 0],
            torch.ones_like(sfr.Z) * -5,
            marker="|",
            color="k",
            linewidth=1.5,
            s=30,
        )
        ax.text(0.0, -4.3, r"\inducing")

    # Show data
    if network is not None:
        f_net = network(X_test).detach()[:, 0]
        if showSVGP == True:
            ax.plot(X_test, f_net, "-r")
        else:
            ax.plot(X_test, f_net, "-r", label="Neural net output")

    if showSVGP == True:
        # Predict y
        f_mean, f_var = sfr.predict(X_test)

        # Mean
        ax.plot(X_test, f_mean[:, 0], "-", color="C0", label="Mean", zorder=1)

        # 95% credible interval
        ax.fill_between(
            X_test[:, 0],
            (f_mean - 1.96 * torch.sqrt(f_var))[:, 0],
            (f_mean + 1.96 * torch.sqrt(f_var))[:, 0],
            color="C0",
            alpha=0.2,
            label="95\% interval",
            zorder=2,
        )

    # Set limits
    ax.set_xlim(X_test.numpy().min(), X_test.numpy().max())
    ax.set_ylim(-5.2, 7)
    ax.legend()
    plt.show()

    # Save
    if save is not None:
        tikzplotlib.save(
            save,
            axis_width="\\figurewidth",
            axis_height="\\figureheight",
            tex_relative_path_to_data="\\datapath",
        )

    return ax


# Plot original model
plot(
    sfr,
    data=(X_train, Y_train),
    network=network,
    showSVGP=False,
    save="./figs/regression-nn.tex",
)

# Plot original model
plot(
    sfr,
    data=None,
    data_dimmed=(X_train, Y_train),
    network=network,
    save="./figs/regression-nn2svgp.tex",
)

# New data #1
X_new = torch.linspace(1.5, 1.8, 10, dtype=torch.float64).reshape(-1, 1)
Y_new = func(X_new, noise=True) - 2

# Update with new data
sfr.update(x=X_new, y=Y_new)

# Plot updated model results
plot(
    sfr,
    data=(X_new, Y_new),
    data_dimmed=(X_train, Y_train),
    network=network,
    save="./figs/regression-update.tex",
)
