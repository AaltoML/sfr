#!/usr/bin/env python3
import random
from typing import Optional

import numpy as np
import src
import torch
import torch.distributions as dists
from experiments.sl.bnn_predictive.experiments.scripts.imginference import (
    get_quick_loader,
)
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


def train_val_split(ds_train: Dataset,
                    ds_test: Dataset,
                    val_from_test: bool = True, 
                    val_split: float = 1/2):
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
def compute_metrics(pred_fn, data_loader, device: str = "cpu") -> dict:
    # def compute_metrics(pred_fn, ds_test, batch_size: int, device: str = "cpu") -> dict:
    # Split the test data set into test and validation sets
    # num_test = len(ds_test)
    # perm_ixs = torch.randperm(num_test)
    # val_ixs, test_ixs = perm_ixs[: int(num_test / 2)], perm_ixs[int(num_test / 2) :]
    # ds_test = Subset(ds_test, test_ixs)
    # test_loader = get_quick_loader(
    #     DataLoader(ds_test, batch_size=batch_size), device=device
    # )

    # targets = torch.cat([y for x, y in data_loader], dim=0).numpy()
    # targets = data_loader.dataset.targets.numpy()

    py, targets = [], []
    for x, y in data_loader:
        py.append(pred_fn(x.to(device)))
        targets.append(y.to(device))

    targets = torch.cat(targets, dim=0).cpu().numpy()
    probs = torch.cat(py).cpu().numpy()

    acc = (probs.argmax(-1) == targets).mean()
    ece = ECE(bins=15).measure(probs, targets)
    nll = (
        -dists.Categorical(torch.Tensor(probs))
        .log_prob(torch.Tensor(targets))
        .mean()
        .numpy()
    )
    metrics = {"acc": acc, "nll": nll, "ece": ece}
    return metrics
