#!/usr/bin/env python3
import random
from typing import Optional

import numpy as np
import src
import torch
import torch.distributions as dists
from experiments.sl.bnn_predictive.experiments.scripts.imgclassification import (
    get_dataset,
    get_model,
    QuickDS,
)
from experiments.sl.bnn_predictive.experiments.scripts.imginference import (
    get_quick_loader,
)
from experiments.sl.bnn_predictive.preds.datasets import UCIClassificationDatasets
from experiments.sl.bnn_predictive.preds.models import SiMLP
from netcal.metrics import ECE
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset


def set_seed_everywhere(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_nll = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_val_nll:
            self.min_val_nll = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_val_nll + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def init_SFR_with_gaussian_prior(
    model: torch.nn.Module,
    delta: float,
    likelihood: src.likelihoods.Likelihood,
    output_dim: int,
    num_inducing: int = 30,
    dual_batch_size: Optional[int] = None,
    jitter: float = 1e-6,
    device: str = "cpu",
) -> src.SFR:
    prior = src.priors.Gaussian(params=model.parameters, delta=delta)
    return src.SFR(
        network=model,
        prior=prior,
        likelihood=likelihood,
        output_dim=output_dim,
        num_inducing=num_inducing,
        dual_batch_size=dual_batch_size,
        jitter=jitter,
        device=device,
    )


def init_NN2GPSubset_with_gaussian_prior(
    model: torch.nn.Module,
    delta: float,
    likelihood: src.likelihoods.Likelihood,
    output_dim: int,
    subset_size: int = 30,
    dual_batch_size: Optional[int] = None,
    jitter: float = 1e-6,
    device: str = "cpu",
) -> src.NN2GPSubset:
    prior = src.priors.Gaussian(params=model.parameters, delta=delta)
    return src.NN2GPSubset(
        network=model,
        prior=prior,
        likelihood=likelihood,
        output_dim=output_dim,
        subset_size=subset_size,
        dual_batch_size=dual_batch_size,
        jitter=jitter,
        device=device,
    )


def train_val_split(
    ds_train: Dataset,
    ds_test: Dataset,
    val_from_test: bool = True,
    val_split: float = 1 / 2,
):
    dataset = ds_test if val_from_test else ds_train
    len_ds = len(dataset)
    perm_ixs = torch.randperm(len_ds)
    val_ixs, ds_ixs = (
        perm_ixs[: int(len_ds * val_split)],
        perm_ixs[int(len_ds * val_split) :],
    )
    ds_val = Subset(dataset, val_ixs)
    if val_from_test:
        ds_test = Subset(dataset, ds_ixs)
    else:
        ds_train = Subset(dataset, ds_ixs)
    return ds_train, ds_val, ds_test


@torch.no_grad()
def compute_metrics(
    pred_fn, data_loader, device: str = "cpu", inference_strategy: str = "sfr"
) -> dict:
    py, targets = [], []
    for idx, (x, y) in enumerate(data_loader):
        idx = None
        if inference_strategy.startswith("sfr"):
            p = pred_fn(x.to(device))
            # p = pred_fn(x.to(device), idx)
        else:
            p = pred_fn(x.to(device))
        py.append(p)
        targets.append(y.to(device))

    targets = torch.cat(targets, dim=0).cpu().numpy()
    probs = torch.cat(py).cpu().numpy()

    if probs.shape[-1] == 1:
        bernoulli = True
    else:
        bernoulli = False

    if bernoulli:
        y_pred = probs >= 0.5
        acc = np.sum((y_pred[:, 0] == targets)) / len(probs)
    else:
        acc = (probs.argmax(-1) == targets).mean()
    ece = ECE(bins=15).measure(probs, targets)  # TODO does this work for bernoulli?

    if bernoulli:
        dist = dists.Bernoulli(torch.Tensor(probs[:, 0]))
    else:
        dist = dists.Categorical(torch.Tensor(probs))
    nll = -dist.log_prob(torch.Tensor(targets)).mean().numpy()
    metrics = {"acc": acc, "nll": nll, "ece": ece}
    return metrics


def get_image_dataset(
    name: str,
    double: bool,
    dir: str,
    device: str,
    debug: bool,
    val_from_test: bool,
    val_split: float,
):
    ds_train, ds_test = get_dataset(
        dataset=name, double=double, dir=dir, device=None, debug=debug
    )
    if debug:
        ds_train.data = ds_train.data[:500]
        ds_train.targets = ds_train.targets[:500]
        ds_test.data = ds_test.data[:500]
        ds_test.targets = ds_test.targets[:500]
    if double:
        print("MAKING DATASET DOUBLE")
        ds_train.data = ds_train.data.to(torch.double)
        ds_test.data = ds_test.data.to(torch.double)
        ds_train.targets = ds_train.targets.long()
        ds_test.targets = ds_test.targets.long()
    # if device is not None:
    #     ds_train.data = ds_train.data.to(device)
    #     ds_test.data = ds_test.data.to(device)
    #     ds_train.targets = ds_train.targets.to(device)
    #     ds_test.targets = ds_test.targets.to(device)
    output_dim = ds_train.K  # set network output dim
    pixels = ds_train.pixels
    channels = ds_train.channels
    ds_train = QuickDS(ds_train, device)
    # ds_val = QuickDS(ds_val, device)
    ds_test = QuickDS(ds_test, device)
    # Split train data set into train and validation
    print("Original num train {}".format(len(ds_train)))
    print("Original num test {}".format(len(ds_test)))
    ds_train, ds_val, ds_test = train_val_split(
        ds_train=ds_train,
        ds_test=ds_test,
        val_from_test=val_from_test,
        val_split=val_split,
    )
    ds_train.K = output_dim
    ds_train.output_dim = output_dim
    ds_train.pixels = pixels
    ds_train.channels = channels
    print("Final num train {}".format(len(ds_train)))
    print("Final num val {}".format(len(ds_val)))
    print("Final num test {}".format(len(ds_test)))
    return ds_train, ds_val, ds_test


def get_uci_dataset(name: str, random_seed: int, dir: str, double: bool, **kwargs):
    ds_train = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=True,
        double=double,
    )
    ds_test = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=False,
        valid=False,
        double=double,
    )
    ds_val = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=False,
        valid=True,
        double=double,
    )
    print(f"dataset={name}")
    if ds_train.C > 2:  # set network output dim
        output_dim = ds_train.C
    else:
        output_dim = 1
    if double:
        ds_train.data = ds_train.data.to(torch.double)
        ds_val.data = ds_val.data.to(torch.double)
        ds_test.data = ds_test.data.to(torch.double)
        ds_train.targets = ds_train.targets.long()
        ds_val.targets = ds_val.targets.long()
        ds_test.targets = ds_test.targets.long()

    # always use Softmax instead of Bernoulli
    output_dim = ds_train.C
    if name in ["australian", "breast_cancer", "ionosphere"]:
        # ds_train.targets = ds_train.targets.to(torch.double)
        # ds_val.targets = ds_val.targets.to(torch.double)
        # ds_test.targets = ds_test.targets.to(torch.double)
        ds_train.targets = ds_train.targets.long()
        ds_val.targets = ds_val.targets.long()
        ds_test.targets = ds_test.targets.long()

    print(f"output_dim={output_dim}")
    print(f"ds_train.C={ds_train.C}")
    # ds_train.K = output_dim
    ds_train.output_dim = output_dim
    return ds_train, ds_val, ds_test


def get_image_network(name: str, ds_train, device: str):
    network = get_model(model_name=name, ds_train=ds_train).to(device)
    return network


def get_uci_network(name, output_dim, ds_train, device: str):
    network = SiMLP(
        input_size=ds_train.data.shape[1],
        output_size=output_dim,
        n_layers=2,
        n_units=50,
        activation="tanh",
    ).to(device)
    return network
