import random

import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


seed = 42


class banana_dataset(Dataset):
    def __init__(self, x, y, labels):
        super(banana_dataset).__init__()
        self.x = x
        self.y = y
        self.labels = labels

    def __getitem__(self, index):
        point = torch.FloatTensor((self.x[index], self.y[index]))
        label = int(self.labels[index])
        return point, label

    def __len__(self):
        return len(self.labels)


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
hamiltorch.set_random_seed(seed)

# Training data
# data = read_csv(datapath+"banana_datapoints_subset.csv", header = None, prefix = 'col')
# classes = read_csv(datapath+"banana_classes_subset.csv", header = None, prefix = 'col')

# Banana data
X = np.loadtxt(
    "https://raw.githubusercontent.com/AaltoML/BayesNewton/main/data/banana_X_train",
    delimiter=",",
)
Y = np.loadtxt(
    "https://raw.githubusercontent.com/AaltoML/BayesNewton/main/data/banana_Y_train"
)[:, None]

Y = Y.astype("float")

n = X.shape[0]
train_set = banana_dataset(
    [X[i, 0] for i in range(n)], [X[i, 1] for i in range(n)], [Y[i] for i in range(n)]
)

print("data set")

# Test data
gridwidth = 100
gridlength = 2.8
x_vals = np.linspace(-gridlength, gridlength, gridwidth)
y_vals = np.linspace(-gridlength, gridlength, gridwidth)
grid_samples = np.zeros((gridwidth * gridwidth, 2))
for i in range(gridwidth):
    for j in range(gridwidth):
        grid_samples[i * gridwidth + j, 0] = x_vals[i]
        grid_samples[i * gridwidth + j, 1] = y_vals[j]

grid_set = torch.from_numpy(grid_samples).float()
big_n = grid_set.shape[0]

test_set = banana_dataset(
    [grid_set[i, 0] for i in range(big_n)],
    [grid_set[i, 1] for i in range(big_n)],
    [0 for i in range(big_n)],
)

N_train = len(train_set)
N_test = len(test_set)

# data_array = np.zeros((N_train,2))
# class_array = np.zeros(N_train)

# data_array[:,0] = np.asarray(data.col0[:])
# data_array[:,1] = np.asarray(data.col1[:])
# class_array[:] = np.asarray(classes.col0)


batch_size = N_train
trainloader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=1,
    pin_memory=True,
)


def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = torch.cat(data, dim=0).reshape(-1, 1)
    return (
        torch.as_tensor(data).reshape((-1, 2)),
        torch.as_tensor(labels).reshape((-1, 1)).float(),
    )


testloader = DataLoader(
    test_set,
    batch_size=N_test,
    shuffle=False,
    # num_workers=1,
    pin_memory=True,
    collate_fn=custom_collate,
)

print("This is the model")

width = 64
model = torch.nn.Sequential(
    torch.nn.Linear(2, width),
    torch.nn.Sigmoid(),
    torch.nn.Linear(width, width),
    torch.nn.Sigmoid(),
    torch.nn.Linear(width, 1),
)


# Init weights from Gaussian
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


model.apply(weights_init_normal)


print("Set up HMC")

for i, (input, target) in enumerate(trainloader):
    x_train = input
    y_train = target.reshape((-1, 1)).float()

for i, (input, target) in enumerate(testloader):
    x_val = input
    y_val = target  # .reshape((-1,1)).float()

# print(y_val.shape)

prior_precision = 0.0002 * N_train
tau = prior_precision
tau_list = [
    tau * torch.ones(2 * width),
    tau * torch.ones(width),
    tau * torch.ones(width * width),
    tau * torch.ones(width),
    tau * torch.ones(width),
    tau * torch.ones(1),
]

num_samples = 10000
burn = 2000
step_size = 0.01  # 0.01 orig

# params_init = hamiltorch.util.flatten(model).clone()
#
# print('Run HMC')
#
# params_hmc = hamiltorch.sample_model(model, x_train, y_train, params_init=params_init,
#                                     num_samples=num_samples, step_size=step_size, num_steps_per_sample=10,
#                                     burn=burn, model_loss = 'binary_class_linear_output',
#                                     tau_list=tau_list)
#
# print('Predict')
#
# pred_list,log_prob_list = hamiltorch.predict_model(model, params_hmc, test_loader=testloader,
#                                                   model_loss = 'binary_class_linear_output',
#                                                   tau_list=tau_list)


def run_hmc():
    params_init = torch.randn(4417)
    params_hmc = hamiltorch.sample_model(
        model,
        x_train,
        y_train,
        params_init=params_init,
        num_samples=num_samples,
        step_size=step_size,
        num_steps_per_sample=10,
        burn=burn,
        model_loss="binary_class_linear_output",
        tau_list=tau_list,
    )
    pred_list, log_prob_list = hamiltorch.predict_model(
        model,
        params_hmc,
        test_loader=testloader,
        model_loss="binary_class_linear_output",
        tau_list=tau_list,
    )
    return torch.mean(pred_list, axis=0), torch.var(pred_list, axis=0)


# Run first chain
Fmu, Fvar = run_hmc()

# The rest of the chains
for i in range(9):
    F0, V0 = run_hmc()
    Fmu = 0.5 * (Fmu + F0)
    Fvar = 0.5 * (Fvar + V0)


## Plotting

import os

import matplotlib
import src


# Custom colormap
cmap0 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "C1"])
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C0", "white"])
C1 = f"#69a9ce"
C0 = f"#df6679"
cmap0 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", C1])
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", [C0, "white"])

colors0 = cmap0(np.linspace(0, 1.0, 128))
colors1 = cmap1(np.linspace(0, 1.0, 128))
colors = np.append(colors0, colors1, axis=0)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)

# For the neural net
cmap_nn = matplotlib.colors.LinearSegmentedColormap.from_list("", [C1, "white", C0])


# Set up plotting
def plot(
    Fmu, Fvar, ax=None, vmax=None, network=None, plotSVGP=True, save=None, data=None
):
    limits = [-2.8, 2.8, -2.8, 2.8]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    xtest, ytest = np.mgrid[limits[0] : limits[1] : 100j, limits[2] : limits[3] : 100j]
    Xtest = np.vstack((xtest.flatten(), ytest.flatten())).T

    # Show observations
    if data is not None:
        X, Y = data
        for i, mark, c in [[1, "o", C0], [0, "s", C1]]:
            ind = Y[:, 0] == i
            ax.scatter(
                X[ind, 0],
                X[ind, 1],
                s=50,
                alpha=0.5,
                edgecolor="k",
                marker=mark,
                color=c,
            )

    # Predict f
    # Fmu = torch.mean(hmc_pred,axis=0)#.detach() #.numpy()
    # Fvar = torch.var(hmc_pred,axis=0)#.detach() #.numpy()

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

    # ax.imshow(foo.reshape(*xtest.shape).transpose(), extent=limits, origin = 'lower')
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

    if save is not None:
        path = os.path.join("./", save)
        plt.savefig(path, dpi=200, bbox_inches="tight")


plot(Fmu, Fvar, data=(X, Y), save="./figs/banana-hmc.png")
