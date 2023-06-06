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
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


def set_seed_everywhere(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def init_SFR_with_gaussian_prior(
    model: torch.nn.Module,
    delta: float,
    likelihood: src.likelihoods.Likelihood,
    output_dim: int,
    num_inducing: int = 30,
    dual_batch_size: Optional[int] = None,
    jitter: float = 1e-6,
    device: str = "cpu",
):
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


def train_val_split(ds_train, split: float = 1 / 6):
    num_train = len(ds_train)
    perm_ixs = torch.randperm(num_train)
    val_ixs, train_ixs = (
        perm_ixs[: int(num_train * split)],
        perm_ixs[int(num_train * split) :],
    )
    print("val {}".format(len(val_ixs)))
    print("train {}".format(len(train_ixs)))
    ds_val = Subset(ds_train, val_ixs)
    ds_train = Subset(ds_train, train_ixs)
    return ds_train, ds_val


def compute_metrics(pred_fn, ds_test, batch_size: int, device: str = "cpu") -> dict:
    # Split the test data set into test and validation sets
    # num_test = len(ds_test)
    # perm_ixs = torch.randperm(num_test)
    # val_ixs, test_ixs = perm_ixs[: int(num_test / 2)], perm_ixs[int(num_test / 2) :]
    # ds_test = Subset(ds_test, test_ixs)
    test_loader = get_quick_loader(
        DataLoader(ds_test, batch_size=batch_size), device=device
    )

    targets = torch.cat([y for x, y in test_loader], dim=0).numpy()

    py = []
    for x, _ in test_loader:
        py.append(pred_fn(x.to(device)))

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
