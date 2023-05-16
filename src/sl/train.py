#!/usr/bin/env python3
import logging
import random
import time
from collections import deque, namedtuple
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import src
import torch
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

# from src.rl.utils import set_seed_everywhere
from src.sl.datasets import CIFAR10, FMNIST, MNIST
from src.sl.networks import CIFAR10Net, CIFAR100Net, MLPS
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = "/".join(PACKAGE_DIR.split("/")[:-2])
DATA_DIR = os.path.join(ROOT, "bnn-predictive/data")


def set_seed_everywhere(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(cfg.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    # pl.seed_everything(random_seed)


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


def get_dataset(dataset, double, dir, device=None):
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
    model.eval()
    loss, acc = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            fs = model(X)
            acc += (torch.argmax(fs, dim=-1) == y).sum().cpu().float().item()
            loss += criterion(fs, y).item()
    return loss / len(data_loader.dataset), acc / len(data_loader.dataset)


# def acc(g, y, likelihood=None):
#     """Binary accuracy"""
#     if type(likelihood) is CategoricalLh:
#         return macc(g, y)
#     y_pred = (g >= 0.5).type(y.dtype)
#     return torch.sum((y_pred == y).float()).item() / len(y_pred)


# def macc(g, y):
#     """Multiclass accuracy"""
#     return torch.sum(torch.argmax(g, axis=-1) == y).item() / len(y)


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
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # cfg.device = "cpu"
    print("Using device: {}".format(cfg.device))

    # print("DATA_DIR {}".format(DATA_DIR))
    # logger.info("cfg {}".format(cfg))
    ds_train, ds_test = src.sl.train.get_dataset(
        dataset=cfg.dataset, double=cfg.double, dir="./", device=cfg.device
    )
    # print("ds_train {}".format(ds_train.D[0].shape))
    # print("ds_train {}".format(ds_train.D[1].shape))
    # print("ds_test {}".format(ds_test.D[0].shape))
    # print("ds_test {}".format(ds_test.D[1].shape))

    # n_classes = next(iter(ds_train))[1].shape[-1]
    n_classes = 10
    print("n_classes {}".format(n_classes))
    cfg.output_dim = n_classes

    network = get_model(model_name=cfg.model_name, ds_train=ds_train)
    network = network.to(cfg.device)
    # print("network {}".format(network))
    prior = hydra.utils.instantiate(cfg.prior, params=network.parameters)
    # print("prior {}".format(prior))
    sfr = hydra.utils.instantiate(cfg.sfr, prior=prior, network=network)
    # print("sfr {}".format(sfr))

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
    logger.info("cfg {}".format(cfg))

    # dataset = hydra.utils.instantiate(cfg.dataset)
    # print("dataset {}".format(dataset))
    # ds_train, ds_test = dataset
    # ntksvgp = hydra.utils.instantiate(cfg.ntksvgp)
    # print("ntksvgp {}".format(ntksvgp))

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=cfg.lr)

    for epoch in tqdm(list(range(cfg.n_epochs))):
        for X, y in train_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
        if epoch % cfg.logging_epoch_freq == 0:
            # tr_loss_sum, tr_loss_mean, tr_acc = evaluate(network, train_loader, cfg.device)
            # te_loss_sum, te_loss_mean, te_acc = evaluate(network, test_loader, cfg.device)
            criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            tr_loss, tr_acc = evaluate(network, train_loader, criterion, cfg.device)
            te_loss, te_acc = evaluate(network, test_loader, criterion, cfg.device)
            wandb.log({"training/loss": tr_loss})
            wandb.log({"test/loss": te_loss})
            # wandb.log({"training/loss_mean": tr_loss_mean})
            # wandb.log({"test/loss_mean": te_loss_mean})
            wandb.log({"training/acc": tr_acc})
            wandb.log({"test/acc": te_acc})
            wandb.log({"epoch": epoch})

    logger.info("Finished training")
    # # evaluation
    # model = svgp.network
    # criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    # logging.info(f"train loss:\t{tr_loss}")
    # logging.info(f"train acc.:\t{tr_acc}")
    # logging.info(f"test loss:\t{te_loss}")
    # logging.info(f"test acc.:\t{te_acc}")
    # metrics = {
    #     "test_loss": te_loss,
    #     "test_acc": te_acc,
    #     "train_loss": tr_loss,
    #     "train_acc": tr_acc,
    # }

    state = {
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        # "losses": losses,
        # "metrics": metrics,
        "delta": cfg.prior.delta,
    }
    res_dir = "./saved_models"
    fname = (
        "./"
        + "_".join([cfg.dataset, cfg.model_name, str(cfg.random_seed)])
        + "_{cfg.prior.delta:.1e}.pt"
    )
    logger.info("Saving model and optimiser etc...")
    torch.save(state, os.path.join(res_dir, fname.format(delta=cfg.prior.delta)))
    logger.info("Finished saving model and optimiser etc")


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
