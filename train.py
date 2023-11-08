#!/usr/bin/env python3
import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import hydra
import numpy as np
import omegaconf
import torch
import torchvision
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import src

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    dataset: str = "FMNIST"

    # Training config
    batch_size: int = 512
    lr: float = 1e-4
    n_epochs: int = 10000

    # SFR config
    prior_precision: float = 0.008
    num_inducing: int = 1000
    dual_batch_size: int = 5000
    jitter: float = 1e-6
    likelihood_eps: float = 0.0  # for numerical stability

    # Early stopping on validation loss
    early_stop_patience: int = 1000
    early_stop_min_delta: float = 0.0

    # Experiment config
    logging_epoch_freq: int = 100
    seed: int = 42
    device: str = "cuda"  # "cpu" or "cuda" etc
    debug: bool = False

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = ""
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val = np.inf

    def __call__(self, val):
        if val < self.min_val:
            self.min_val = val
            self.counter = 0
        elif val > (self.min_val + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train")
def train(cfg: TrainConfig):
    # Make experiment reproducible
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    # Use GPU if requested and available
    if "cuda" in cfg.device:
        if torch.cuda.is_available():
            cfg.device = "cuda"
        else:
            logger.info("cuda requested but not available")
            cfg.device = "cpu"
    logger.info("Using device: {}".format(cfg.device))

    # Initialize W&B
    run_name = f"{cfg.dataset}_{time.time()}"
    if cfg.use_wandb:
        run = wandb.init(
            project=cfg.wandb_project_name,
            name=run_name,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )

    # Make everything double precision
    torch.set_default_dtype(torch.double)

    # Load the data with train/val/test split
    # build an array = np.asrray( [x for x in range(80000)])
    indices = np.arange(0, 80000)
    np.random.shuffle(indices)  # shuffle the indicies

    # Build the train loader using indices from 0 to 75000
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/mnist",
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        ),
        batch_size=64,
        shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:75000]),
    )
    input_dim = train_loader.dataset
    breakpoint()
    output_dim = train_loader.dataset

    # Build the validation loader using indices from 75000 to 80000
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/mnist",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        ),
        batch_size=64,
        shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(indices[-5000:]),
    )

    # Create the neural network
    hidden_size = 64
    network = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, output_dim),
    )

    # Instantiate SFR
    prior = src.priors.Gaussian(
        params=network.parameters, prior_precision=cfg.prior_precision
    )
    likelihood = src.likelihoods.CategoricalLh(EPS=cfg.likelihood_eps)
    sfr = src.SFR(
        network=network,
        prior=prior,
        likelihood=likelihood,
        output_dim=output_dim,
        num_inducing=cfg.num_inducing,
        dual_batch_size=cfg.dual_batch_size,
        jitter=cfg.jitter,
        device=cfg.device,
    )

    optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=cfg.lr)

    @torch.no_grad()
    def loss_fn(data_loader: DataLoader):
        cum_loss = 0
        for X, y in data_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            cum_loss += loss
        return cum_loss

    early_stopper = EarlyStopper(
        patience=int(cfg.early_stop_patience / cfg.logging_epoch_freq),
        min_delta=cfg.early_stop_min_delta,
    )

    # Train NN weights uthe se empirical regularized risk
    best_loss = float("inf")
    for epoch in tqdm(list(range(cfg.n_epochs))):
        for X, y in train_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})

        if epoch % cfg.logging_epoch_freq == 0:
            val_loss = loss_fn(val_loader)
            # wandb.log({"val_loss": val_loss})
            # wandb.log({"train/": train_metrics})
            # wandb.log({"val/": val_metrics})
            wandb.log({"epoch": epoch})

            if val_loss < best_loss:
                # checkpoint(sfr=sfr, optimizer=optimizer, save_dir=run.dir)
                best_loss = val_loss
            if early_stopper(val_loss):  # (val_loss):
                logger.info("Early stopping criteria met, stopping training...")
                break

    logger.info("Finished training")

    sfr.eval()

    logger.info("Fitting SFR...")
    sfr.fit(train_loader=train_loader)
    logger.info("Finished fitting SFR")


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
