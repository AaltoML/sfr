import pickle
import os
import numpy as np
import torch
import tqdm
from torch.nn.utils import parameters_to_vector

from preds.optimizers import LaplaceGGN, get_diagonal_ggn
from preds.models import SiMLP
from src import BernoulliLh, CategoricalLh, NTKSVGP
import src as ntksvgp

# from preds.likelihoods import BernoulliLh, CategoricalLh
from preds.predictives import (
    nn_sampling_predictive,
    linear_sampling_predictive,
    svgp_sampling_predictive,
)
from preds.utils import acc, nll_cls, ece
from preds.mfvi import run_bbb
from preds.refine import laplace_refine, vi_refine, vi_diag_refine
from preds.datasets import UCIClassificationDatasets

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, likelihood, X_train, y_train, optimizer, n_epochs):
    """Train model with given optimizer and run postprocessing"""
    losses = list()
    is_bernoulli = isinstance(likelihood, BernoulliLh)
    for i in range(n_epochs):

        def closure():
            model.zero_grad()
            f = model(X_train)
            if not is_bernoulli:
                return (
                    likelihood.nn_loss(f=f.squeeze(), y=y_train.squeeze()),
                    X_train.shape[0],
                )
            else:
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


def preds_svgp(X, svgp, likelihood, samples=1000):
    gs = svgp_sampling_predictive(X, svgp, likelihood, mc_samples=samples)
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
    X_train, y_train, model, likelihood, prior_prec, n_inducing=64, device="cpu"
):
    data = (X_train, y_train)
    n_classes = model(X_train).shape[-1]
    print(f"N classes: {n_classes}")
    print(f"Prior prec: {prior_prec}")
    prior = ntksvgp.priors.Gaussian(params=model.parameters, delta=prior_prec)
    svgp = NTKSVGP(
        network=model,
        prior=prior,
        output_dim=n_classes,
        likelihood=likelihood,
        num_inducing=n_inducing,
        device=device,
    )
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
    activation="tanh",
    n_inducing=64,
    n_samples=1000,
    refine=True,
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
        likelihood = BernoulliLh(EPS=0.000000001)
        K = 1
    else:
        eps = 0.0000000001
        likelihood = CategoricalLh(EPS=eps)
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
        device=device,
    )
    fs_train = preds_svgp(X_train, svgp, likelihood_svgp, samples=n_samples)
    fs_test = preds_svgp(X_test, svgp, likelihood_svgp, samples=n_samples)
    fs_valid = preds_svgp(X_valid, svgp, likelihood_svgp, samples=n_samples)
    res.update(evaluate(fs_train, y_train, lh, "svgp_ntk", "train"))
    res.update(evaluate(fs_test, y_test, lh, "svgp_ntk", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "svgp_ntk", "valid"))

    # GP subset predictive

    sparse_idx = torch.randperm(X_train.shape[0])[:n_inducing]
    sparse_x = X_train[sparse_idx]
    sparse_y = y_train[sparse_idx]
    if isinstance(likelihood_svgp, CategoricalLh):
        sparse_y = sparse_y.squeeze()
    else:
        sparse_y = sparse_y.unsqueeze(-1)
    svgp_subset = create_ntksvgp(
        sparse_x,
        sparse_y,
        model,
        likelihood_svgp,
        prior_prec_n,
        n_inducing=n_inducing,
        device=device,
    )
    fs_train = preds_svgp(X_train, svgp_subset, likelihood_svgp, samples=n_samples)
    fs_test = preds_svgp(X_test, svgp_subset, likelihood_svgp, samples=n_samples)
    fs_valid = preds_svgp(X_valid, svgp_subset, likelihood_svgp, samples=n_samples)
    res.update(evaluate(fs_train, y_train, lh, "gp_subset", "train"))
    res.update(evaluate(fs_test, y_test, lh, "gp_subset", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "gp_subset", "valid"))

    # BBB
    # baseline  (needs higher lr)
    lrv, epochsv = lr * 10, int(n_epochs / 2)
    res_bbb = run_bbb(
        ds_train,
        ds_test,
        ds_valid,
        prior_prec,
        device,
        likelihood,
        epochsv,
        lr=lrv,
        n_samples_train=1,
        n_samples_pred=100, # was n_samples
        n_layers=n_layers,  
        n_units=n_units,
        activation=activation,
    )
    res["elbos_bbb"] = res_bbb["elbos"]
    res.update(evaluate(res_bbb["preds_train"], y_train, lh, "bbb", "train"))
    res.update(evaluate(res_bbb["preds_test"], y_test, lh, "bbb", "test"))
    res.update(evaluate(res_bbb["preds_valid"], y_valid, lh, "bbb", "valid"))

    # return res

    # Extract relevant variables
    theta_star = parameters_to_vector(model.parameters()).detach()
    Sigmad, Sigma_chold = get_diagonal_ggn(optimizer, num_data=X_train.shape[0])
    Sigma_chol = optimizer.state["Sigma_chol"]

    # SVGP predictive (with GP subset, not true sparse GP)
    #   fs_train, data_sparse = preds_svgp(X_train, X_train, y_train, model, likelihood,prior_prec,  n_sparse=n_sparse, samples=n_samples, sparse_points=data_sparse, subset=True)
    #   fs_test, _ = preds_svgp(X_test, X_train, y_train, model, likelihood, prior_prec, n_sparse=n_sparse,  samples=n_samples, sparse_points=data_sparse, subset=True)
    #   fs_valid, _ = preds_svgp(X_valid, X_train, y_train, model, likelihood, prior_prec,  n_sparse=n_sparse, samples=n_samples, sparse_points=data_sparse, subset=True)
    #   res.update(evaluate(fs_train, y_train, lh, 'gp_subset', 'train'))
    #   res.update(evaluate(fs_test, y_test, lh, 'gp_subset', 'test'))
    #   res.update(evaluate(fs_valid, y_valid, lh, 'gp_subset', 'valid'))

    # LinLaplace full Cov assuming convergence
    fs_train = preds_glm(
        X_train, model, likelihood, theta_star, Sigma_chol, samples=n_samples
    )
    fs_test = preds_glm(
        X_test, model, likelihood, theta_star, Sigma_chol, samples=n_samples
    )
    fs_valid = preds_glm(
        X_valid, model, likelihood, theta_star, Sigma_chol, samples=n_samples
    )
    res.update(evaluate(fs_train, y_train, lh, "glm", "train"))
    res.update(evaluate(fs_test, y_test, lh, "glm", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "glm", "valid"))

    # LinLapalce diagonal cov
    fs_train = preds_glm(
        X_train, model, likelihood, theta_star, Sigma_chold, samples=n_samples
    )
    fs_test = preds_glm(
        X_test, model, likelihood, theta_star, Sigma_chold, samples=n_samples
    )
    fs_valid = preds_glm(
        X_valid, model, likelihood, theta_star, Sigma_chold, samples=n_samples
    )
    res.update(evaluate(fs_train, y_train, lh, "glmd", "train"))
    res.update(evaluate(fs_test, y_test, lh, "glmd", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "glmd", "valid"))

    # Laplace-GGN NN sampling (cf Ritter et al.)
    fs_train = preds_nn(
        X_train, model, likelihood, theta_star, Sigma_chol, samples=n_samples
    )
    fs_test = preds_nn(
        X_test, model, likelihood, theta_star, Sigma_chol, samples=n_samples
    )
    fs_valid = preds_nn(
        X_valid, model, likelihood, theta_star, Sigma_chol, samples=n_samples
    )
    res.update(evaluate(fs_train, y_train, lh, "nn", "train"))
    res.update(evaluate(fs_test, y_test, lh, "nn", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "nn", "valid"))

    # Laplace-GGN diagonal cov
    fs_train = preds_nn(
        X_train, model, likelihood, theta_star, Sigma_chold, samples=n_samples
    )
    fs_test = preds_nn(
        X_test, model, likelihood, theta_star, Sigma_chold, samples=n_samples
    )
    fs_valid = preds_nn(
        X_valid, model, likelihood, theta_star, Sigma_chold, samples=n_samples
    )
    res.update(evaluate(fs_train, y_train, lh, "nnd", "train"))
    res.update(evaluate(fs_test, y_test, lh, "nnd", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "nnd", "valid"))

    # REFINEMENT
    if not refine:
        print("Skipping refinement")
        return res

    # Full Laplace
    m, S_chol, S_chold, losses = laplace_refine(
        model, X_train, y_train, likelihood, prior_prec
    )

    res["losses_lap"] = losses
    fs_train = preds_glm(X_train, model, likelihood, m, S_chol, samples=n_samples)
    fs_test = preds_glm(X_test, model, likelihood, m, S_chol, samples=n_samples)
    fs_valid = preds_glm(X_valid, model, likelihood, m, S_chol, samples=n_samples)
    res.update(evaluate(fs_train, y_train, lh, "glmLap", "train"))
    res.update(evaluate(fs_test, y_test, lh, "glmLap", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "glmLap", "valid"))

    # Diag Laplace
    fs_train = preds_glm(X_train, model, likelihood, m, S_chold, samples=n_samples)
    fs_test = preds_glm(X_test, model, likelihood, m, S_chold, samples=n_samples)
    fs_valid = preds_glm(X_valid, model, likelihood, m, S_chold, samples=n_samples)
    res.update(evaluate(fs_train, y_train, lh, "glmLapd", "train"))
    res.update(evaluate(fs_test, y_test, lh, "glmLapd", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "glmLapd", "valid"))

    # Full VI
    m, S_chol, losses = vi_refine(model, optimizer, X_train, y_train, likelihood)
    res["elbos_vi"] = losses
    res["elbo_glm"] = losses[-1]
    fs_train = preds_glm(X_train, model, likelihood, m, S_chol, samples=n_samples)
    fs_test = preds_glm(X_test, model, likelihood, m, S_chol, samples=n_samples)
    fs_valid = preds_glm(X_valid, model, likelihood, m, S_chol, samples=n_samples)
    res.update(evaluate(fs_train, y_train, lh, "glmVI", "train"))
    res.update(evaluate(fs_test, y_test, lh, "glmVI", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "glmVI", "valid"))

    # Diag VI
    m, S_chold, losses = vi_diag_refine(model, optimizer, X_train, y_train, likelihood)
    res["elbos_vid"] = losses
    res["elbo_glmd"] = losses[-1]
    fs_train = preds_glm(X_train, model, likelihood, m, S_chold, samples=n_samples)
    fs_test = preds_glm(X_test, model, likelihood, m, S_chold, samples=n_samples)
    fs_valid = preds_glm(X_valid, model, likelihood, m, S_chold, samples=n_samples)
    res.update(evaluate(fs_train, y_train, lh, "glmVId", "train"))
    res.update(evaluate(fs_test, y_test, lh, "glmVId", "test"))
    res.update(evaluate(fs_valid, y_valid, lh, "glmVId", "valid"))

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
    parser.add_argument(
        "--refine", help="on/off switch for posterior refinement", type=int
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
    refine = args.refine
    refine = bool(refine)
    print(f"Refine: {refine}")

    data_dir = os.path.join(root_dir, "data")
    res_dir = os.path.join(root_dir, "experiments", "results", "uci", res_folder)
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    print(f"Writing results to {res_dir}")
    print(f"Reading data from {data_dir}")
    print(f"Dataset: {dataset}")
    print(f'Number of inducing points: {n_inducing}')
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
        activation=activation,
        n_inducing=n_inducing,
        n_samples=n_samples,
        refine=refine,
    )
