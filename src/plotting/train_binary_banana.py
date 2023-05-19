import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src
import torch
import torch.nn as nn
from src import SFR
from src.custom_types import Data


# Download the Banana data
X = np.loadtxt(
    "https://raw.githubusercontent.com/AaltoML/BayesNewton/main/data/banana_X_train",
    delimiter=",",
)
Y = np.loadtxt(
    "https://raw.githubusercontent.com/AaltoML/BayesNewton/main/data/banana_Y_train"
)[:, None]

X_train = torch.from_numpy(X)
Y_train = torch.from_numpy(Y)


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

            if epoch_idx % 100 == 0:
                logger.info(
                    "Epoch: {} | Batch: {} | Loss: {}".format(
                        epoch_idx, batch_idx, loss
                    )
                )

    sfr.set_data(data)
    return {"loss": loss_history}


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    torch.manual_seed(45)
    torch.set_default_dtype(torch.float64)

    plot_var = False
    plot_var = True
    save_dir = "./figs/binary"

    torch.set_default_dtype(torch.float64)

    delta = 0.0002

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

    # width = 512
    width = 64
    network = torch.nn.Sequential(
        torch.nn.Linear(2, width),
        torch.nn.Sigmoid(),
        # torch.nn.Tanh(),
        # torch.nn.ReLU(),
        # Sin(),
        torch.nn.Linear(width, width),
        torch.nn.Sigmoid(),
        # torch.nn.Tanh(),
        # torch.nn.ReLU(),
        # Sin(),
        torch.nn.Linear(width, 1),
    )
    network.apply(weights_init_normal)
    print("network: {}".format(network))

    print("X_train {}".format(X_train.shape))
    print("X_train {}".format(X_train.shape))
    data = (X_train, Y_train)
    print("X, Y: {}, {}".format(X_train.shape, Y_train.shape))

    batch_size = X_train.shape[0]

    likelihood = src.likelihoods.BernoulliLh()
    # likelihood = src.likelihoods.CategoricalLh()
    # likelihood = src.likelihoods.Gaussian()
    prior = src.priors.Gaussian(params=network.parameters, delta=delta)
    sfr = SFR(
        network=network,
        # train_data=(X_train, Y_train),
        prior=prior,
        likelihood=likelihood,
        output_dim=1,
        num_inducing=50,  # X_train.shape[0],
        # num_inducing=50,
        # jitter=1e-6,
        jitter=1e-4,
    )

    print("setting data")
    sfr.set_data((X_train, Y_train))
    print("FINISHED setting data")
    metrics = train(
        sfr=sfr,
        data=data,
        num_epochs=5000,
        batch_size=batch_size,
        learning_rate=1e-2,
    )


import matplotlib


# Custom colormap
cmap0 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "C1"])
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C0", "white"])
colors0 = cmap0(np.linspace(0, 1.0, 128))
colors1 = cmap1(np.linspace(0, 1.0, 128))
colors = np.append(colors0, colors1, axis=0)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)


# Set up plotting
def plot(sfr, ax=None, vmax=None):
    limits = [-2.8, 2.8, -2.8, 2.8]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    xtest, ytest = np.mgrid[limits[0] : limits[1] : 30j, limits[2] : limits[3] : 30j]
    Xtest = np.vstack((xtest.flatten(), ytest.flatten())).T
    for i, mark in [[1, "o"], [0, "s"]]:
        ind = Y[:, 0] == i
        ax.scatter(X[ind, 0], X[ind, 1], s=50, alpha=0.5, edgecolor="k", marker=mark)
    if hasattr(sfr, "Z"):
        ax.scatter(sfr.Z.numpy()[:, 0], sfr.Z.numpy()[:, 1], s=20, color="k")
    # if hasattr(m.kernel,'z'):
    #    ax.scatter(limits[0]*np.ones(m.kernel.z.value.shape[0]),m.kernel.z.value[:,0], s=40, color='k', marker='>')
    # if hasattr(m,'R'):
    #    mu, var = m.predict_y(X=Xtest[:,:1],R=Xtest[:,1:2])
    # else:
    #    mu, var = m.predict_y(X=Xtest)

    # Predict f
    X_test = torch.from_numpy(Xtest)
    Fmu, Fvar = sfr.predict_f(X_test)

    # Predict y
    p = src.likelihoods.inv_probit(Fmu / torch.sqrt(1 + Fvar))
    mu = p
    var = p - torch.square(p)

    mu = mu.numpy()
    var = var.numpy()

    # Scale background
    foo = mu > 0.5
    foo = foo.astype(float)
    foo = (2.0 * foo - 1.0) * np.sqrt(var)
    if vmax == None:
        vmax = np.max(np.sqrt(var))

    ax.imshow(
        foo.reshape(*xtest.shape).transpose(),
        extent=limits,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.contour(
        xtest, ytest, mu.reshape(*xtest.shape), levels=[0.5], colors="k", linewidths=2.0
    )
    ax.axis("equal")
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
    # ax.axis('off')
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", "box")


plot(sfr)

plt.savefig(
    os.path.join(save_dir, "sfr-banana.pdf"),
    transparent=True,
)
