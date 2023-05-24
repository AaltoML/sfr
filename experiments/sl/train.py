#!/usr/bin/env python3
import logging
import os
import random


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import numpy as np
import omegaconf
import src
import torch
import wandb
from experiments.sl.bnn_predictive.preds.datasets import CIFAR10, FMNIST, MNIST
from experiments.sl.bnn_predictive.preds.models import CIFAR10Net, CIFAR100Net, MLPS
from experiments.sl.inference import compute_metrics
from omegaconf import DictConfig
from src.likelihoods import BernoulliLh, CategoricalLh
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from tqdm import tqdm


PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = "/".join(PACKAGE_DIR.split("/")[:-2])
DATA_DIR = os.path.join(ROOT, "bnn-predictive/data")


def set_seed_everywhere(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


class QuickDS(VisionDataset):
    def __init__(self, ds, device):
        self.D = [
            (ds[i][0].to(device), torch.tensor(ds[i][1]).to(device))
            for i in range(len(ds))
        ]
        self.K = ds.K
        self.channels = ds.channels
        self.pixels = ds.pixels

    def __getitem__(self, index):
        return self.D[index]

    def __len__(self):
        return len(self.D)


def get_dataset(dataset, double, dir, cfg, device=None):
    if dataset == "MNIST":
        # Download training data from open datasets.
        ds_train = MNIST(train=True, double=double, root=dir)
        ds_test = MNIST(train=False, double=double, root=dir)
    elif dataset == "FMNIST":
        ds_train = FMNIST(train=True, double=double, root=dir)
        ds_test = FMNIST(train=False, double=double, root=dir)
    elif dataset == "CIFAR10":
        ds_train = CIFAR10(train=True, double=double, root=dir)
        ds_test = CIFAR10(train=False, double=double, root=dir)
    else:
        raise ValueError("Invalid dataset argument")
    if device is not None:
        if cfg.debug:
            ds_train.data = ds_train.data[:500]
            ds_train.targets = ds_train.targets[:500]
            ds_test.data = ds_test.data[:500]
            ds_test.targets = ds_test.targets[:500]
        return QuickDS(ds_train, device), QuickDS(ds_test, device)
    else:
        return ds_train, ds_test


def get_model(model_name, ds_train):
    if model_name == "MLP":
        input_size = ds_train.pixels**2 * ds_train.channels
        hidden_sizes = [1024, 512, 256, 128]
        output_size = ds_train.K
        return MLPS(input_size, hidden_sizes, output_size, "tanh", flatten=True)
    elif model_name == "SmallMLP":
        input_size = ds_train.pixels**2 * ds_train.channels
        hidden_sizes = [128, 128]
        output_size = ds_train.K
        return MLPS(input_size, hidden_sizes, output_size, "tanh", flatten=True)
    elif model_name == "CNN":
        return CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True)
    elif model_name == "AllCNN":
        return CIFAR100Net(ds_train.channels, ds_train.K)
    else:
        raise ValueError("Invalid model name")


def evaluate(model, data_loader, criterion, device):
    likelihood = src.likelihoods.CategoricalLh()
    model.eval()
    loss, acc, nll = 0, 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            fs = model(X)
            acc += (torch.argmax(fs, dim=-1) == y).sum().cpu().float().item()
            loss += criterion(fs, y).item()
            nll += -likelihood.log_prob(f=fs, y=y).sum().item()
    return (
        loss / len(data_loader.dataset),
        acc / len(data_loader.dataset),
        nll / len(data_loader.dataset),
    )




def nll_cls(p, y, likelihood):
    """Avg. Negative log likelihood for classification"""
    if type(likelihood) is BernoulliLh:
        p_dist = Bernoulli(probs=p)
        return -p_dist.log_prob(y).mean().item()
    elif type(likelihood) is CategoricalLh:
        p_dist = Categorical(probs=p)
        return -p_dist.log_prob(y).mean().item()
    else:
        raise ValueError("Only Bernoulli and Categorical likelihood.")


def ece(probs, labels, likelihood=None, bins=10):
    # source: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    if type(likelihood) is BernoulliLh:
        probs = torch.stack([1 - probs, probs]).t()
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels.long())

    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


@hydra.main(version_base="1.3", config_path="./configs", config_name="main")
def train(cfg: DictConfig):
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.double:
        logger.info("Using float64")
        torch.set_default_dtype(torch.double)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))


    ds_train, ds_test = get_dataset(
        dataset=cfg.dataset, double=cfg.double, dir="./", device=cfg.device, cfg=cfg
    )

    n_classes = 10
    print("n_classes {}".format(n_classes))
    cfg.output_dim = n_classes

    network = get_model(model_name=cfg.model_name, ds_train=ds_train)
    network = network.to(cfg.device)
    prior = hydra.utils.instantiate(cfg.prior, params=network.parameters)
    sfr = hydra.utils.instantiate(cfg.sfr, prior=prior, network=network)

    if cfg.wandb.use_wandb:  # Initialise WandB
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        )

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    for epoch in tqdm(list(range(cfg.n_epochs))):
        for X, y in train_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
        if epoch % cfg.logging_epoch_freq == 0:
            tr_loss, tr_acc, tr_nll = evaluate(
                network, train_loader, criterion, cfg.device
            )
            wandb.log({"training/loss": tr_loss})
            wandb.log({"training/nll": tr_nll})
            wandb.log({"training/acc": tr_acc})
            wandb.log({"epoch": epoch})

    logger.info("Finished training")

    state = {
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "delta": cfg.prior.delta,
    }

    res_dir = os.path.join(run.dir, "./saved_models")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    fname = (
        "./"
        + "_".join([cfg.dataset, cfg.model_name, str(cfg.random_seed)])
        + f"_{cfg.prior.delta:.1e}.pt"
    )
    logger.info("Saving model and optimiser etc...")
    torch.save(state, os.path.join(res_dir, fname))
    logger.info("Finished saving model and optimiser etc")


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
