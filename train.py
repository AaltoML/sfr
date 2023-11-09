#!/usr/bin/env python3
import logging

# import random
import time
from dataclasses import dataclass

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import torchvision
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
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Dataset
    dataset: str = "FMNIST"  # "FMNIST"/"CIFAR10"/"MNIST"
    train_val_split: float = 0.8
    debug: bool = False  # If true only use 500 data points

    # SFR config
    prior_precision: float = 0.0013
    num_inducing: int = 2048
    # dual_batch_size: int = 5000
    dual_batch_size: int = 1024
    jitter: float = 1e-6
    likelihood_eps: float = 0.0  # for numerical stability

    # Training config
    batch_size: int = 64
    lr: float = 1e-4
    n_epochs: int = 10000
    # Early stopping on validation loss
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0

    # Experiment config
    logging_epoch_freq: int = 2
    test_batch_size: int = 2048  # batch size for computing metrics
    seed: int = 42
    device: str = "cuda"  # "cpu" or "cuda" etc

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = "sfr"


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train")
def train(cfg: TrainConfig):
    # Make experiment reproducible
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # random.seed(cfg.seed)
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
    save_dir = f"{get_original_cwd()}/data"
    if "FMNIST" in cfg.dataset:
        dataset_fn = torchvision.datasets.FashionMNIST
        normalize_transform = transforms.Normalize((0.2860,), (0.3530,))
        # Calculated with ds_train.train_data.float().mean()/255
    elif "MNIST" in cfg.dataset:
        dataset_fn = torchvision.datasets.MNIST
        # Calculated with ds_train.train_data.float().mean()/255
        normalize_transform = transforms.Normalize((0.1307,), (0.3081,))
    elif "CIFAR10" in cfg.dataset:
        dataset_fn = torchvision.datasets.CIFAR10
        # Calculated with ds_train.data.mean(axis=(0,1,2))/255
        normalize_transform = transforms.Normalize(
            (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
        )
    else:
        raise NotImplementedError("Only MNIST/FMNIST/CIFAR10 supported for cfg.dataset")

    transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    ds_train = dataset_fn(
        f"{save_dir}/{cfg.dataset}", download=True, train=True, transform=transform
    )
    output_dim = len(ds_train.classes)
    if ds_train.data.ndim == 3:
        in_channels = 1
    elif ds_train.data.ndim == 4:
        in_channels = ds_train.data.shape[-1]
    num_data = len(ds_train) if not cfg.debug else 500
    idxs = np.random.permutation(num_data)
    split_idx = int(cfg.train_val_split * num_data)

    if cfg.debug:
        ds_test = Subset(ds_train, idxs[split_idx + 1 :])
    else:
        ds_test = dataset_fn(
            f"{save_dir}/{cfg.dataset}", download=True, train=False, transform=transform
        )
    train_loader = DataLoader(
        Subset(ds_train, idxs[:split_idx]), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(ds_train, idxs[split_idx + 1 :]),
        batch_size=cfg.test_batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ds_test, batch_size=cfg.test_batch_size, shuffle=True, pin_memory=True
    )

    # Instantiate SFR
    # TODO This doesn't use tanh...
    network = utils.CIFAR10Net(in_channels=in_channels, n_out=output_dim, use_tanh=True)
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
        dtype = next(model.parameters()).dtype  # because change NN from float to double
        for data, target in data_loader:
            data = data.to(dtype).to(cfg.device)
            target = target.to(cfg.device)
            if sfr_pred:  # predict with SFR
                probs.append(model(data.to(cfg.device))[0])
            else:  # predict with NN
                probs.append(torch.softmax(model.network(data), dim=-1))
            targets.append(target)
            val_losses.append(model.loss(data, target))

        val_loss = torch.mean(torch.stack(val_losses, 0)).cpu().numpy().item()
        targets = torch.cat(targets, dim=0).cpu().numpy()
        probs = torch.cat(probs).cpu().numpy()
        acc = (probs.argmax(-1) == targets).mean()
        ece = ECE(bins=15).measure(probs, targets)
        dist = torch.distributions.Categorical(torch.Tensor(probs))
        nlpd = -dist.log_prob(torch.Tensor(targets)).mean().numpy()
        metrics = {"loss": val_loss, "acc": acc, "nlpd": nlpd, "ece": ece}
        model.train()
        return metrics

    # Train NN weights with empirical regularized risk
    # best_loss = float("inf")
    for epoch_idx in tqdm(list(range(cfg.n_epochs)), total=cfg.n_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch_idx}/{cfg.n_epochs}")
                loss = model.loss(data.to(cfg.device), target.to(cfg.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

                if wandb.run is not None:
                    wandb.log({"train_loss": loss})

            if epoch_idx % cfg.logging_epoch_freq == 0:
                val_metrics = evaluate(model, data_loader=val_loader)
                val_loss = val_metrics["loss"]
                if wandb.run is not None:
                    wandb.log({"val_loss": val_loss, "epoch": epoch_idx})

                # if val_loss < best_loss:
                # checkpoint(model=model, optimizer=optimizer, save_dir=run.dir)
                # best_loss = val_loss
                if early_stopper(val_loss):  # (val_loss):
                    logger.info("Early stopping criteria met, stopping training...")
                    break

    logger.info("Finished training")

    class MetricLogger:
        def __init__(self):
            self.df = pd.DataFrame(columns=["Model", "loss", "acc", "nlpd", "ece"])

        def log(self, metrics: dict, name: str):
            logger.info(
                f"{name} NLPD {metrics['nlpd']} | ACC: {metrics['acc']} | ECE: {metrics['ece']}"
            )
            metrics.update({"Model": name})
            if wandb.run is not None:
                self.df.loc[len(self.df.index)] = metrics
                wandb.log({"Metrics": wandb.Table(data=self.df)})

    # Calculate NN's metrics and log
    nn_metrics = evaluate(model, data_loader=test_loader, sfr_pred=False)
    metric_logger = MetricLogger()
    metric_logger.log(nn_metrics, name="NN")

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
