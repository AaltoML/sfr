#!/usr/bin/env python3
import logging
import random
import time
from dataclasses import dataclass

import pandas as pd
import hydra
import numpy as np
import omegaconf
import torch
import torchvision
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from netcal.metrics import ECE
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

import likelihoods
import priors
import sfr
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Dataset
    dataset: str = "FMNIST"  # "FMNIST"/"CIFAR10"/"MNIST"
    debug: bool = False  # If true only use 500 data points

    # SFR config
    prior_precision: float = 0.008
    num_inducing: int = 1000
    dual_batch_size: int = 5000
    jitter: float = 1e-6
    likelihood_eps: float = 0.0  # for numerical stability

    # Training config
    batch_size: int = 512
    lr: float = 1e-4
    n_epochs: int = 10000
    # Early stopping on validation loss
    early_stop_patience: int = 1000
    early_stop_min_delta: float = 0.0

    # Experiment config
    logging_epoch_freq: int = 100
    seed: int = 42
    device: str = "cuda"  # "cpu" or "cuda" etc

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = "sfr"


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


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
            logger.info("CUDA requested but not available")
            cfg.device = "cpu"
    logger.info("Using device: {}".format(cfg.device))

    # Initialize W&B
    run_name = f"{cfg.dataset}_{time.time()}"
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project_name,
            name=run_name,
            group=cfg.dataset,
            tags=[cfg.dataset, f"M={cfg.num_inducing}"],
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )

    # Load the data with train/val/test split
    if "FMNIST" in cfg.dataset:
        dataset_fn = torchvision.datasets.FashionMNIST
    elif "MNIST" in cfg.dataset:
        dataset_fn = torchvision.datasets.MNIST
    elif "CIFAR10" in cfg.dataset:
        dataset_fn = torchvision.datasets.CIFAR10
    else:
        raise NotImplementedError("Only MNIST/FMNIST/CIFAR10 supported for cfg.dataset")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    ds_train = dataset_fn(
        f"data/{cfg.dataset}", download=True, train=True, transform=transform
    )
    output_dim = len(ds_train.classes)
    num_data = 500 if cfg.debug else len(ds_train)
    idxs = np.random.permutation(num_data)
    split_idx = int(0.7 * num_data)

    ds_test = dataset_fn(
        f"data/{cfg.dataset}", download=True, train=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        Subset(ds_train, idxs[:split_idx]), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        Subset(ds_train, idxs[split_idx + 1 :]),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=cfg.batch_size, shuffle=True, pin_memory=True
    )

    # Instantiate SFR
    # network = CNN()
    network = utils.CIFAR10Net(in_channels=1, n_out=10, use_tanh=True)
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

    early_stopper = utils.EarlyStopper(
        patience=int(cfg.early_stop_patience / cfg.logging_epoch_freq),
        min_delta=cfg.early_stop_min_delta,
    )

    @torch.no_grad()
    def evaluate(model: sfr.SFR, data_loader: DataLoader, sfr_pred: bool = False):
        model.eval()
        probs, targets, val_losses = [], [], []
        for data, target in data_loader:
            if sfr_pred:  # predict with SFR
                probs.append(model(data.to(cfg.device))[0])
            else:  # predict with NN
                probs.append(torch.softmax(model.network(data.to(cfg.device)), dim=-1))
            targets.append(target.to(cfg.device))
            val_losses.append(model.loss(data.to(cfg.device), target.to(cfg.device)))

        val_loss = torch.mean(torch.stack(val_losses, 0)).numpy().item()
        targets = torch.cat(targets, dim=0).cpu().numpy()
        probs = torch.cat(probs).cpu().numpy()
        acc = (probs.argmax(-1) == targets).mean()
        ece = ECE(bins=15).measure(probs, targets)
        dist = torch.distributions.Categorical(torch.Tensor(probs))
        nll = -dist.log_prob(torch.Tensor(targets)).mean().numpy()
        metrics = {"loss": val_loss, "acc": acc, "nll": nll, "ece": ece}
        model.train()
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
            val_metrics = evaluate(model, data_loader=val_loader)
            val_loss = val_metrics["loss"]
            if wandb.run is not None:
                wandb.log({"val_loss": val_loss, "epoch": epoch})

            if val_loss < best_loss:
                # checkpoint(model=model, optimizer=optimizer, save_dir=run.dir)
                best_loss = val_loss
            if early_stopper(val_loss):  # (val_loss):
                logger.info("Early stopping criteria met, stopping training...")
                break

    logger.info("Finished training")

    class MetricLogger:
        def __init__(self):
            self.df = pd.DataFrame(columns=["Model", "loss", "acc", "nll", "ece"])

        def log(self, metrics: dict, name: str):
            logger.info(
                f"{name} NLPD {metrics['nll']} | ACC: {metrics['acc']} | ECE: {metrics['ece']}"
            )
            metrics.update({"Model": name})
            if wandb.run is not None:
                self.df.loc[len(self.df.index)] = metrics
                wandb.log({"Metrics": wandb.Table(data=self.df)})

    # Calculate NN's metrics and log
    nn_metrics = evaluate(model, data_loader=test_loader, sfr_pred=False)
    metric_logger = MetricLogger()
    metric_logger.log(nn_metrics, name="NN")

    # Make everything double precision
    torch.set_default_dtype(torch.double)
    model.double()
    model.eval()

    # Calculate posterior (dual parameters etc)
    logger.info("Fitting SFR...")
    model.fit(train_loader=train_loader)
    logger.info("Finished fitting SFR")

    # Calculate SFR's metrics and log
    sfr_metrics = evaluate(model, data_loader=test_loader, sfr_pred=True)
    metric_logger.log(sfr_metrics, name="SFR")
    nn_metrics = evaluate(model, data_loader=test_loader, sfr_pred=False)
    metric_logger.log(nn_metrics, name="NN double")


if __name__ == "__main__":
    train()  # pyright: ignore
