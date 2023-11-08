#!/usr/bin/env python3
import logging
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Union

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torchvision
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from netcal.metrics import ECE
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import likelihoods
import priors
import sfr

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


class CIFAR10Net(nn.Module):
    def __init__(self, in_channels: int = 3, n_out: int = 10, use_tanh: bool = False):
        super().__init__()
        self.output_size = n_out
        activ = nn.Tanh if use_tanh else nn.ReLU

        self.cnn_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=64,
                            kernel_size=(5, 5),
                            stride=(1, 1),
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "maxpool1",
                        nn.Sequential(
                            nn.ZeroPad2d((0, 1, 0, 1)),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        ),
                    ),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=96,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    (
                        "maxpool2",
                        nn.Sequential(
                            nn.ZeroPad2d((0, 1, 0, 1)),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        ),
                    ),
                    (
                        "conv3",
                        nn.Conv2d(
                            in_channels=96,
                            out_channels=128,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                    ),
                    ("relu3", nn.ReLU()),
                    (
                        "maxpool3",
                        nn.Sequential(
                            nn.ZeroPad2d((1, 1, 1, 1)),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        ),
                    ),
                ]
            )
        )
        self.lin_block = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("dense1", nn.Linear(in_features=3 * 3 * 128, out_features=512)),
                    ("activ1", activ()),
                    ("dense2", nn.Linear(in_features=512, out_features=256)),
                    ("activ2", activ()),
                    (
                        "dense3",
                        nn.Linear(in_features=256, out_features=self.output_size),
                    ),
                ]
            )
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.cnn_block(x)
        out = self.lin_block(x)
        return out


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train_config")
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
        wandb.init(
            project=cfg.wandb_project_name,
            name=run_name,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )

    # Load the data with train/val/test split
    # build an array = np.asrray( [x for x in range(80000)])
    indices = np.arange(0, 800)
    # indices = np.arange(0, 80000)
    np.random.shuffle(indices)  # shuffle the indicies
    output_dim = 10

    # Build the train loader using indices from 0 to 75000
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/mnist", download=True, train=True, transform=transform
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:100]),
        # sampler=torch.utils.data.SubsetRandomSampler(indices[:75000]),
    )
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/mnist", download=True, train=True, transform=transform
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(indices[100:200]),
        # sampler=torch.utils.data.SubsetRandomSampler(indices[-5000:]),
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/mnist", download=True, train=False, transform=transform
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # Instantiate SFR
    # network = CNN()
    network = CIFAR10Net(in_channels=1, n_out=10, use_tanh=True)
    prior = priors.Gaussian(
        params=network.parameters, prior_precision=cfg.prior_precision
    )
    likelihood = likelihoods.CategoricalLh(EPS=cfg.likelihood_eps)
    model = sfr.SFR(
        network=network,
        prior=prior,
        likelihood=likelihood,
        output_dim=output_dim,
        num_inducing=cfg.num_inducing,
        dual_batch_size=cfg.dual_batch_size,
        jitter=cfg.jitter,
        device=cfg.device,
    )
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=cfg.lr)

    early_stopper = EarlyStopper(
        patience=int(cfg.early_stop_patience / cfg.logging_epoch_freq),
        min_delta=cfg.early_stop_min_delta,
    )

    def evaluate(model: Union[sfr.SFR, nn.Module], test_loader: DataLoader):
        model.eval()
        with torch.no_grad():
            py, targets, val_losses = [], [], []
            for x, y in test_loader:
                # if isinstance
                py.append(model.network(x=x.to(cfg.device)))
                targets.append(y.to(cfg.device))
                val_losses.append(model.loss(x.to(cfg.device), y.to(cfg.device)))

            val_loss = torch.mean(torch.stack(val_losses, 0))
            targets = torch.cat(targets, dim=0).cpu().numpy()
            probs = torch.cat(py).cpu().numpy()
            acc = (probs.argmax(-1) == targets).mean()
            ece = ECE(bins=15).measure(probs, targets)
            dist = torch.distributions.Categorical(torch.Tensor(probs))
            nll = -dist.log_prob(torch.Tensor(targets)).mean().numpy()
            metrics = {"loss": val_loss, "acc": acc, "nll": nll, "ece": ece}
        return metrics

    # Train NN weights uthe se empirical regularized risk
    best_loss = float("inf")
    for epoch in tqdm(list(range(cfg.n_epochs))):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                loss = model.loss(data.to(cfg.device), target.to(cfg.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

                if wandb.run is not None:
                    wandb.log({"loss": loss})

        if epoch % cfg.logging_epoch_freq == 0:
            metrics = evaluate(model, test_loader=val_loader)
            # metrics = evaluate(model, test_loader=test_loader)
            val_loss = metrics["loss"]
            # val_losses = []
            # for x, y in val_loader:
            #     val_losses.append(model.loss(x.to(cfg.device), y.to(cfg.device)))
            # val_loss = torch.mean(torch.stack(val_losses, 0))
            if wandb.run is not None:
                wandb.log({"val_loss": val_loss, "epoch": epoch})

            if val_loss < best_loss:
                # checkpoint(model=model, optimizer=optimizer, save_dir=run.dir)
                best_loss = val_loss
            if early_stopper(val_loss):  # (val_loss):
                logger.info("Early stopping criteria met, stopping training...")
                break

    logger.info("Finished training")

    # Make everything double precision
    torch.set_default_dtype(torch.double)
    model.double()
    model.eval()

    def to_double(sample):
        breakpoint()
        X, y = sample
        return X, y

    logger.info("Fitting SFR...")
    model.fit(train_loader=train_loader)
    logger.info("Finished fitting SFR")

    @torch.no_grad()
    def compute_metrics(
        data_loader: DataLoader, pred_type: str = "nn", device: str = "cpu"
    ) -> dict:
        py, targets = [], []
        for x, y in data_loader:
            py.append(model(x=x.to(device), pred_type=pred_type))
            targets.append(y.to(device))

        targets = torch.cat(targets, dim=0).cpu().numpy()
        probs = torch.cat(py).cpu().numpy()

        acc = (probs.argmax(-1) == targets).mean()
        ece = ECE(bins=15).measure(probs, targets)
        dist = torch.distributions.Categorical(torch.Tensor(probs))
        nll = -dist.log_prob(torch.Tensor(targets)).mean().numpy()
        metrics = {"acc": acc, "nll": nll, "ece": ece}
        return metrics

    sfr_metrics = compute_metrics(test_loader, pred_type="gp", device=cfg.device)
    logger.info(
        f"SFR NLPD {sfr_metrics['nlpd']} | ACC: {sfr_metrics['acc']} | ECE: {sfr_metrics['ece']}"
    )
    if wandb.run is not None:
        wandb.log(sfr_metrics)


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
