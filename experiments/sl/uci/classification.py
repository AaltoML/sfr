import pickle
import os
import numpy as np
import torch
import tqdm
from torch.nn.utils import parameters_to_vector

from experiments.sl.bnn_predictive.preds.optimizers import LaplaceGGN, get_diagonal_ggn
from experiments.sl.bnn_predictive.preds.models import SiMLP
from src import SFR, NN2GPSubset
import src

from experiments.sl.bnn_predictive.preds.predictives import (
    nn_sampling_predictive,
    linear_sampling_predictive,
    svgp_sampling_predictive,
)
from experiments.sl.bnn_predictive.preds.utils import acc, nll_cls, ece
from experiments.sl.bnn_predictive.preds.datasets import UCIClassificationDatasets

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, likelihood, X_train, y_train, optimizer, n_epochs):
    """Train model with given optimizer and run postprocessing"""
    losses = list()
    for i in range(n_epochs):

        def closure():
            model.zero_grad()
            f = model(X_train)
            return (
                    likelihood.nn_loss(f=f.squeeze(), y=y_train.squeeze()),
                    X_train.shape[0],
                )

        loss = optimizer.step(closure)
        losses.append(loss)
    optimizer.post_process(model, likelihood, [(X_train, y_train)])
    return losses


def preds_glm(X, model, likelihood, mu, Sigma_chol, samples):
    gs = linear_sampling_predictive(
        X, model, likelihood, mu, Sigma_chol, mc_samples=samples
    )
    return gs.mean(dim=0)


def preds_svgp(X, svgp, likelihood, samples=1000, batch_size=100, nn_mean=False,device='cpu'):
    gs = svgp_sampling_predictive(X, svgp, likelihood, mc_samples=samples, batch_size=batch_size, nn_mean=nn_mean, device=device)
    return gs.mean(dim=0)


def preds_nn(X, model, likelihood, mu, Sigma_chol, samples):
    gs = nn_sampling_predictive(
        X, model, likelihood, mu, Sigma_chol, mc_samples=samples
    )
    return gs.mean(dim=0)


def evaluate(p, y, likelihood, name, data):
    # returns are result dictionary with nll, acc, ece named
    res = dict()
    res[f"{data}_nll_{name}"] = nll_cls(p.squeeze(), y.squeeze(), likelihood)
    res[f"{data}_acc_{name}"] = acc(p.squeeze(), y.squeeze(), likelihood)
    res[f"{data}_ece_{name}"] = ece(p.squeeze(), y.squeeze(), likelihood, bins=10)
    if name in ["svgp_ntk", "map", 'glm'] and data == "valid":
        print(f"Val result for: {name}")
        print(nll_cls(p.squeeze(), y.squeeze(), likelihood))
    return res


def create_ntksvgp(
    X_train, y_train, model, likelihood, prior_prec, n_inducing=64, batch_size=1000, device="cpu", subset=False
):
    data = (X_train, y_train)
    n_classes = model(X_train).shape[-1]
    print(f"N classes: {n_classes}")
    print(f"Prior prec: {prior_prec}")
    prior = ntksvgp.priors.Gaussian(params=model.parameters, delta=prior_prec)
    if not subset:
        svgp = SFR(
            network=model,
            prior=prior,
            output_dim=n_classes,
            likelihood=likelihood,
            num_inducing=n_inducing,
            dual_batch_size=batch_size,
            jitter=10**(-6),
            device=device,
        )
    else:
       svgp = NN2GPSubset(network=model,
                          prior=prior,
                          likelihood=likelihood,
                          output_dim=n_classes,
                          subset_size=n_inducing,
                          jitter=10**(-6),
                          device=device)
    svgp.set_data(data)
    return svgp


def inference(
    ds_train,
    ds_test,
    ds_valid,
    prior_prec,
    lr,
    n_epochs,
    device,
    seed,
    n_layers=2,
    n_units=50,
    batch_size=1000,
    activation="tanh",
    n_inducing=64,
    n_samples=1000
):
    """Full inference (training and prediction)
    storing all relevant quantities and returning a state dictionary.
    if sigma_noise is None, we have classification.
    """
    """Training"""
    X_train, y_train = ds_train.data.to(device), ds_train.targets.to(device)
    X_test, y_test = ds_test.data.to(device), ds_test.targets.to(device)
    X_valid, y_valid = ds_valid.data.to(device), ds_valid.targets.to(device)
    D = X_train.shape[1]
    res = dict()
    torch.manual_seed(seed)
    if ds_train.C == 2:
        eps = 0.000000001
        eps = 0
        likelihood = src.likelihoods.BernoulliLh(EPS=eps)
        K = 1
    else:
        eps = 0.0000000001
        eps = 0
        likelihood = src.likelihoodsCategoricalLh(EPS=eps)
        K = ds_train.C

    print(f'X_train shape: {X_train.shape[0]}')
    if X_train.shape[0] <= n_inducing:
        print('WARNING: Using all training data as inducing points')
        n_inducing = X_train.shape[0]

    if n_inducing == 0:
        print('Using all data as inducing points')
        n_inducing = X_train.shape[0]

    prior_prec_n = prior_prec / y_train.shape[0]
    print(f"prior precision: {prior_prec_n}")

    model = SiMLP(D, K, n_layers, n_units, activation=activation).to(device)
    optimizer = LaplaceGGN(model, lr=lr, prior_prec=prior_prec_n)
    print("Training NN...")
    res["losses"] = train(model, likelihood, X_train, y_train, optimizer, n_epochs)

    """Prediction"""
    lh = likelihood
    # MAP
    fs_train = likelihood.inv_link(model(X_train).detach())
    fs_test = likelihood.inv_link(model(X_test).detach())
    fs_valid = likelihood.inv_link(model(X_valid).detach())
    res.update(evaluate(fs_train, y_train, lh, "map", "train"))
    res.update(evaluate(fs_test, y_test, likelihood, "map", "test"))
    res.update(evaluate(fs_valid, y_valid, likelihood, "map", "valid"))

    # SVGP predictive
    if isinstance(likelihood, CategoricalLh):
        eps_2 = eps
        likelihood_svgp = CategoricalLh(EPS=eps_2)
        y_input = y_train.squeeze()
    else:
        y_input = y_train.unsqueeze(-1)
        likelihood_svgp = likelihood
    svgp = create_ntksvgp(
        X_train,
        y_input,
        model,
        likelihood_svgp,
        prior_prec_n,
        n_inducing=n_inducing,
        batch_size=batch_size,
        device=device,
    )
    fs_train = preds_svgp(X_train, svgp, likelihood_svgp, samples=n_samples, batch_size=batch_size, device=device)
    fs_test = preds_svgp(X_test, svgp, likelihood_svgp, samples=n_samples, batch_size=batch_size, device=device)
    fs_valid = preds_svgp(X_valid, svgp, likelihood_svgp, samples=n_samples, batch_size=batch_size, device=device)
    res.update(evaluate(fs_train, y_train, lh, "svgp_ntk", "train"))
    res.update(evaluate(fs_test, y_test, lh, "svgp_ntk", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "svgp_ntk", "valid"))


    fs_train = preds_svgp(X_train, svgp, likelihood_svgp, samples=n_samples, batch_size=batch_size, nn_mean=True,device=device)
    fs_test = preds_svgp(X_test, svgp, likelihood_svgp, samples=n_samples, batch_size=batch_size, nn_mean=True,device=device)
    fs_valid = preds_svgp(X_valid, svgp, likelihood_svgp, samples=n_samples, batch_size=batch_size, nn_mean=True, device=device)
    res.update(evaluate(fs_train, y_train, lh, "svgp_ntk_nn", "train"))
    res.update(evaluate(fs_test, y_test, lh, "svgp_ntk_nn", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "svgp_ntk_nn", "valid"))

    # GP subset predictive

    
    svgp_subset = create_ntksvgp(
        X_train,
        y_input,
        model,
        likelihood_svgp,
        prior_prec_n,
        batch_size=batch_size,
        n_inducing=n_inducing,
        device=device,
        subset=True
    )
    fs_train = preds_svgp(X_train, svgp_subset, likelihood_svgp, samples=n_samples, batch_size=batch_size, device=device)
    fs_test = preds_svgp(X_test, svgp_subset, likelihood_svgp, samples=n_samples, batch_size=batch_size, device=device)
    fs_valid = preds_svgp(X_valid, svgp_subset, likelihood_svgp, samples=n_samples, batch_size=batch_size, device=device)
    res.update(evaluate(fs_train, y_train, lh, "gp_subset", "train"))
    res.update(evaluate(fs_test, y_test, lh, "gp_subset", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "gp_subset", "valid"))

    # GP Subset with NN mean
    fs_train = preds_svgp(X_train, svgp_subset, likelihood_svgp, samples=n_samples, batch_size=batch_size, nn_mean=True,device=device)
    fs_test = preds_svgp(X_test, svgp_subset, likelihood_svgp, samples=n_samples, batch_size=batch_size, nn_mean=True, device=device)
    fs_valid = preds_svgp(X_valid, svgp_subset, likelihood_svgp, samples=n_samples, batch_size=batch_size,  nn_mean=True, device=device)
    res.update(evaluate(fs_train, y_train, lh, "gp_subset_nn", "train"))
    res.update(evaluate(fs_test, y_test, lh, "gp_subset_nn", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "gp_subset_nn", "valid"))

    return res

def main(
    ds_train, ds_test, ds_valid, deltas, device, dataset, name, seed, res_dir, **kwargs
):
    results = list()
    for i, delta in tqdm.tqdm(list(enumerate(deltas))):
        res = inference(
            ds_train,
            ds_test,
            ds_valid,
            prior_prec=delta,
            device=device,
            seed=seed,
            **kwargs,
        )
        results.append(res)

    resdict = dict()
    resdict["results"] = results
    resdict["deltas"] = deltas
    resdict["N_train"] = len(ds_train)
    resdict["N_test"] = len(ds_test)
    resdict["K"] = ds_train.C

    res_file = f"classification_{dataset}_{name}_{seed}.pkl"
    with open(os.path.join(res_dir, res_file), "wb") as f:
        pickle.dump(resdict, f)
    print(f"Wrote results to {res_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    multi_datasets = ["glass", "vehicle", "waveform", "satellite", "digits"]
    binary_datasets = [
        "TwoMoons",
        "australian",
        "breast_cancer",
        "ionosphere",
        "banana",
    ]
    datasets = multi_datasets + binary_datasets
    parser.add_argument(
        "-d", "--dataset", help="dataset", choices=datasets, required=True
    )
    parser.add_argument("--double", help="double precision", action="store_true")
    parser.add_argument("-s", "--seed", help="randomness seed", default=7011, type=int)
    parser.add_argument(
        "--n_epochs", help="epochs training neural network", default=10000, type=int
    )
    parser.add_argument(
        "-b", "--batch_size", help="Jac/Kernel batch size", type=int, default=1000
    )
    parser.add_argument(
        "--lr", help="neural network learning rate", default=1e-3, type=float
    )
    parser.add_argument(
        "--n_deltas", help="number of deltas to try", default=10, type=int
    )
    parser.add_argument("--logd_min", help="min log delta", default=-2.0, type=float)
    parser.add_argument("--logd_max", help="max log delta", default=2.0, type=float)
    parser.add_argument("--n_layers", help="number of layers", default=2, type=int)
    parser.add_argument(
        "--n_units", help="number of hidden units per layer", default=50, type=int
    )
    parser.add_argument(
        "--activation",
        help="activation function",
        default="tanh",
        choices=["tanh", "relu"],
    )
    parser.add_argument("--root_dir", help="Root directory", default="../")
    parser.add_argument("--res_folder", help="Result folder", default="test")
    parser.add_argument("--name", help="name result file", default="", type=str)
    parser.add_argument(
        "--n_samples", help="number predictive samples", type=int, default=1000
    )
    parser.add_argument(
        "--n_inducing",
        help="number of sparse data points to use for the svgp",
        type=int,
        default=64,
    )
    args = parser.parse_args()
    dataset = args.dataset
    double = args.double
    seed = args.seed
    n_epochs = args.n_epochs
    lr = args.lr
    n_deltas = args.n_deltas
    logd_min, logd_max = args.logd_min, args.logd_max
    n_layers, n_units = args.n_layers, args.n_units
    activation = args.activation
    n_samples = args.n_samples
    n_inducing = args.n_inducing
    name = args.name
    root_dir = args.root_dir
    res_folder = args.res_folder
    batch_size = args.batch_size

    data_dir = os.path.join(root_dir, "data")
    res_dir = os.path.join(root_dir,  "results", res_folder)
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    print(f"Writing results to {res_dir}")
    print(f"Reading data from {data_dir}")
    print(f"Dataset: {dataset}")
    print(f'Number of inducing points: {n_inducing}')
    print(f'Batch size: {batch_size}')
    print(f"Seed: {seed}")
    print(double)

    if double:
        torch.set_default_dtype(torch.double)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    if device != "cuda":
        print("Running on CPU")
    else:
        print(f"Device name: {torch.cuda.get_device_name()}")
    ds_train = UCIClassificationDatasets(
        dataset,
        random_seed=seed,
        root=data_dir,
        stratify=True,
        train=True,
        double=double,
    )
    ds_test = UCIClassificationDatasets(
        dataset,
        random_seed=seed,
        root=data_dir,
        stratify=True,
        train=False,
        valid=False,
        double=double,
    )
    ds_valid = UCIClassificationDatasets(
        dataset,
        random_seed=seed,
        root=data_dir,
        stratify=True,
        train=False,
        valid=True,
        double=double,
    )

    deltas = np.logspace(logd_min, logd_max, n_deltas)
    main(
        ds_train,
        ds_test,
        ds_valid,
        deltas,
        device,
        dataset,
        name,
        seed,
        res_dir,
        n_epochs=n_epochs,
        lr=lr,
        n_layers=n_layers,
        n_units=n_units,
        batch_size=batch_size, 
        activation=activation,
        n_inducing=n_inducing,
        n_samples=n_samples
    )
