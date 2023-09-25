#!/usr/bin/env python3
import copy
import logging
import os
import time
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import hydra
import numpy as np
import pandas as pd
import src
import torch
import wandb
from experiments.sl.train import checkpoint
from experiments.sl.inference import sfr_pred
from experiments.sl.utils import (
    compute_metrics,
    EarlyStopper,
    set_seed_everywhere,
    compute_metrics_regression,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from tqdm import trange




class TableLogger:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # Data dictionary used to make pd.DataFrame
        self.data = {
            "dataset": [],
            "model": [],
            "seed": [],
            "num_inducing": [],
            "acc": [],
            "nlpd": [],
            "ece": [],
            "mse": [],
            "prior_prec": [],
            "time": [],
            "method": [],
        }

    def add_data(
        self,
        model_name: str,
        metrics: dict,
        prior_prec: float,
        num_inducing: Optional[int] = None,
        time: float = None,
        method: str = None,
    ):
        "Add NLL to data dict and wandb table"
        if isinstance(prior_prec, torch.Tensor):
            prior_prec = prior_prec.item()
        self.data["dataset"].append(self.cfg.dataset.name)
        self.data["model"].append(model_name)
        self.data["seed"].append(self.cfg.random_seed)
        self.data["num_inducing"].append(num_inducing)

        print(f"meterics {metrics}")
        self.data["nlpd"].append(metrics["nll"])
        try:
            self.data["acc"].append(metrics["acc"])
        except:
            self.data["acc"].append("")
        try:
            self.data["mse"].append(metrics["mse"])
        except:
            self.data["mse"].append("")
        try:
            self.data["ece"].append(metrics["acc"])
        except:
            self.data["ece"].append("")

        self.data["prior_prec"].append(prior_prec)
        self.data["time"].append(time)
        self.data["method"].append(method)
        if self.cfg.wandb.use_wandb:
            wandb.log({"Metrics": wandb.Table(data=pd.DataFrame(self.data))})


def make_ds_double(ds: Dataset, likelihood: str = "classification") -> Dataset:
    if isinstance(ds, TensorDataset):
        tensors = []
        for tensor in ds.tensors:
            tensors.append(tensor.to(torch.double))
        ds.tensors = tensors
        return ds
    try:
        ds.data = ds.data.to(torch.double)
        if likelihood == "classification":
            ds.targets = ds.targets.long()
        else:
            ds.targets = ds.targets.double()
    except:
        ds.dataset.data = ds.dataset.data.to(torch.double)
        ds.dataset.targets = ds.dataset.targets.long()
        if likelihood == "classification":
            ds.dataset.targets = ds.dataset.targets.long()
        else:
            ds.dataset.targets = ds.dataset.targets.double()
    return ds


def make_data_loader_double(
    data_loader: DataLoader, likelihood: str = "classification"
) -> DataLoader:
    double_data_loader = copy.deepcopy(data_loader)
    ds = double_data_loader.dataset
    make_ds_double(ds, likelihood=likelihood)
    return double_data_loader


@hydra.main(version_base="1.3", config_path="./configs", config_name="regression")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd

    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    print("Using device: {}".format(cfg.device))

    table_logger = TableLogger(cfg)

    # torch.set_default_dtype(torch.float)
    if cfg.double_train:
        torch.set_default_dtype(torch.double)

    # Load train/val/test/update data sets
    ds_train, ds_val, ds_test, ds_update = hydra.utils.instantiate(
        cfg.dataset,
        dir=os.path.join(get_original_cwd(), "data"),
        double=cfg.double_train,
    )

    # Why do I have to see this abominations?
    output_dim = ds_train.output_dim
    cfg.output_dim = ds_train.output_dim

    logger.info(f"D: {len(ds_train)}")
    logger.info(f"F: {ds_train.output_dim}")
    try:
        logger.info(f"D: {ds_train.data.shape[-1]}")
    except:
        pass
    ds_train = ConcatDataset([ds_train, ds_val])
    ds_train.output_dim = output_dim
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

    # Init Weight and Biases
    print(f"cfg.output_dim {cfg.output_dim}")
    print(OmegaConf.to_yaml(cfg))
    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        run_dir = run.dir
    else:
        run_dir = "./"

    # Instantiate the neural network
    network = hydra.utils.instantiate(cfg.network, ds_train=ds_train)

    # Instantiate SFR
    sfr = hydra.utils.instantiate(cfg.sfr, model=network)
    sfr.double()
    print(f"made sfr {sfr}")
    if isinstance(sfr.likelihood, src.likelihoods.Gaussian):
        likelihood = "regresssion"
    else:
        likelihood = "classification"

    # Sample Z from train and update
    ds_train_and_update = ConcatDataset([ds_train, ds_update])
    indices = torch.randperm(len(ds_train_and_update))[: sfr.num_inducing]
    print(f"indices {indices.shape}")
    print(f"indices {indices}")
    # breakpoint()
    Z_ds = torch.utils.data.Subset(ds_train_and_update, indices)
    Z_ds = DataLoader(Z_ds, batch_size=len(Z_ds))
    Z = next(iter(Z_ds))[0].to(sfr.device)
    print(f"Z {Z.shape}")
    sfr.Z = Z.to(sfr.device)

    @torch.no_grad()
    def loss_fn(data_loader: DataLoader, model: src.SFR = None):
        if model is None:
            model = sfr
        losses = []
        for X, y in data_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = model.loss(X, y)
            losses.append(loss)
        losses = torch.stack(losses, 0)
        cum_loss = torch.mean(losses, 0)
        return cum_loss

    def train_loop(sfr, data_loader: DataLoader, val_loader: DataLoader):
        optimizer = torch.optim.Adam(
            [
                {"params": sfr.parameters()},
                # {"params": sfr.likelihood.sigma_noise},
                {"params": sfr.likelihood.log_sigma_noise},
            ],
            lr=cfg.lr,
        )

        early_stopper = EarlyStopper(
            patience=int(cfg.early_stop.patience / cfg.logging_epoch_freq),
            min_prior_precision=cfg.early_stop.min_prior_precision,
        )

        @torch.no_grad()
        def map_pred_fn(x, idx=None):
            f = sfr.network(x.to(cfg.device))
            return sfr.likelihood.inv_link(f)

        best_loss = float("inf")
        for epoch in trange(cfg.n_epochs, ncols=100):
            for X, y in data_loader:
                X = X.to(cfg.device)
                y = y.to(cfg.device)
                loss = sfr.loss(X, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cfg.wandb.use_wandb:
                    wandb.log({"loss": loss})
                    wandb.log({"log_sigma_noise": sfr.likelihood.sigma_noise})

            val_loss = loss_fn(val_loader, model=sfr)
            if epoch % cfg.logging_epoch_freq == 0 and cfg.wandb.use_wandb:
                wandb.log({"val_loss": val_loss})
                if likelihood == "classification":
                    train_metrics = compute_metrics(
                        pred_fn=map_pred_fn, data_loader=train_loader, device=cfg.device
                    )
                    val_metrics = compute_metrics(
                        pred_fn=map_pred_fn, data_loader=val_loader, device=cfg.device
                    )
                    test_metrics = compute_metrics(
                        pred_fn=map_pred_fn, data_loader=test_loader, device=cfg.device
                    )
                else:
                    train_metrics = compute_metrics_regression(
                        model=sfr, data_loader=train_loader, device=cfg.device, map=True
                    )
                    val_metrics = compute_metrics_regression(
                        model=sfr, data_loader=val_loader, device=cfg.device, map=True
                    )
                    test_metrics = compute_metrics_regression(
                        model=sfr, data_loader=test_loader, device=cfg.device, map=True
                    )
                wandb.log({"train/": train_metrics})
                wandb.log({"val/": val_metrics})
                wandb.log({"test/": test_metrics})
                wandb.log({"epoch": epoch})

            if val_loss < best_loss:
                best_ckpt_fname = checkpoint(
                    sfr=sfr, optimizer=optimizer, save_dir=run_dir
                )
                best_loss = val_loss
            if early_stopper(val_loss):  # (val_loss):
                logger.info("Early stopping criteria met, stopping training...")
                break

        # Load checkpoint
        ckpt = torch.load(best_ckpt_fname)
        sfr.load_state_dict(ckpt["model"])
        return sfr

    def train_and_log(
        sfr: src.SFR,
        train_loader: DataLoader,
        name: str,
        inference_loader: Optional[DataLoader] = None,
        train_val_split: float = 0.7,
    ):
        ds_train, ds_val = torch.utils.data.random_split(
            train_loader.dataset, [train_val_split, 1 - train_val_split]
        )
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=True)
        sfr.train()
        start_time = time.time()
        sfr = train_loop(sfr, data_loader=train_loader, val_loader=val_loader)
        train_time = time.time() - start_time
        
        if cfg.double_inference:
            torch.set_default_dtype(torch.double)
            sfr.double()

        sfr.eval()
        log_map_metrics(
            sfr,
            test_loader,
            name=name,
            table_logger=table_logger,
            device=cfg.device,
            time=train_time,
        )

        # Fit SFR
        logger.info("Fitting SFR...")
        if inference_loader is None:
            inference_loader = train_loader
        start_time = time.time()

        sfr.Z = Z
        all_train = DataLoader(
            inference_loader.dataset, batch_size=len(inference_loader.dataset)
        )
        sfr.train_data = next(iter(all_train))
        sfr._build_sfr()

        inference_time = time.time() - start_time
        logger.info("Finished fitting SFR")
        log_sfr_metrics(
            sfr,
            name=name,
            test_loader=test_loader,
            table_logger=table_logger,
            device=cfg.device,
            time=inference_time + train_time,
        )
        return sfr

    sfr = train_and_log(
        sfr,
        train_loader=train_loader,
        name="train",
    )

    return table_logger


def log_map_metrics(
    sfr, test_loader, name: str, table_logger, device, time: float = None
):
    from experiments.sl.utils import compute_metrics

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x.to(device))
        return sfr.likelihood.inv_link(f)

    if isinstance(sfr.likelihood, src.likelihoods.Gaussian):
        map_metrics = compute_metrics_regression(
            model=sfr, data_loader=test_loader, device=device, map=True
        )
    else:
        map_metrics = compute_metrics(
            pred_fn=map_pred_fn, data_loader=test_loader, device=device
        )
    table_logger.add_data(
        "NN MAP",
        metrics=map_metrics,
        num_inducing=None,
        prior_prec=sfr.prior.prior_precision,
        time=time,
        method=name,
    )
    logger.info(f"map_metrics: {map_metrics}")


def log_sfr_metrics(
    sfr,
    test_loader,
    name: str,
    table_logger: TableLogger,
    device="cuda",
    num_samples=100,
    time: float = None,
):
    if isinstance(sfr.likelihood, src.likelihoods.Gaussian):
        gp_metrics = compute_metrics_regression(
            model=sfr,
            pred_type="gp",
            # pred_type="nn",
            data_loader=test_loader,
            device=device,
        )
    else:
        gp_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr,
                pred_type="gp",
                # pred_type="nn",
                num_samples=num_samples,
                device=device,
            ),
            data_loader=test_loader,
            device=device,
        )
    table_logger.add_data(
        "SFR (GP)",
        metrics=gp_metrics,
        num_inducing=sfr.num_inducing,
        prior_prec=sfr.prior.prior_precision,
        time=time,
        method=name,
    )
    logger.info(f"SFR metrics: {gp_metrics}")


if __name__ == "__main__":
    main()  # pyright: ignore
