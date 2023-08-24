#!/usr/bin/env python3
import copy
import logging
import os
import time
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from experiments.sl.train import checkpoint
from experiments.sl.utils import compute_metrics, EarlyStopper, set_seed_everywhere
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


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
            "prior_prec": [],
            "time": [],
            "method": [],
        }
        self.tbl = wandb.Table(
            columns=[
                "dataset",
                "model",
                "seed",
                "num_inducing",
                "acc",
                "nlpd",
                "ece",
                "prior_prec",
                "time",
                "method",
            ]
        )

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
        self.data["acc"].append(metrics["acc"])
        self.data["nlpd"].append(metrics["nll"])
        self.data["ece"].append(metrics["ece"])
        self.data["prior_prec"].append(prior_prec)
        self.data["time"].append(time)
        self.data["method"].append(method)
        self.tbl.add_data(
            self.cfg.dataset.name,
            model_name,
            self.cfg.random_seed,
            num_inducing,
            metrics["acc"],
            metrics["nll"],
            metrics["ece"],
            prior_prec,
            time,
            method,
        )
        wandb.log({"Metrics": wandb.Table(data=pd.DataFrame(self.data))})


def make_ds_double(ds: Dataset) -> Dataset:
    try:
        ds.data = ds.data.to(torch.double)
        ds.targets = ds.targets.long()
    except:
        ds.dataset.data = ds.dataset.data.to(torch.double)
        ds.dataset.targets = ds.dataset.targets.long()
    return ds


def make_data_loader_double(data_loader: DataLoader) -> DataLoader:
    double_data_loader = copy.deepcopy(data_loader)
    ds = double_data_loader.dataset
    make_ds_double(ds)
    return double_data_loader


@hydra.main(version_base="1.3", config_path="./configs", config_name="fast_updates")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd

    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))

    table_logger = TableLogger(cfg)

    torch.set_default_dtype(torch.float)

    # Load train/val/test/update data sets
    ds_train, ds_val, ds_test, ds_update = hydra.utils.instantiate(
        cfg.dataset,
        dir=os.path.join(get_original_cwd(), "data"),
        # double=cfg.double_inference,
        double=False,
        train_update_split=cfg.train_update_split,
    )
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)
    update_loader = DataLoader(ds_update, batch_size=cfg.batch_size, shuffle=True)
    train_and_update_loader = DataLoader(
        ConcatDataset([ds_train, ds_update]), batch_size=cfg.batch_size, shuffle=True
    )
    test_loader_double = make_data_loader_double(data_loader=test_loader)
    update_loader_double = make_data_loader_double(data_loader=update_loader)

    # Init Weight and Biases
    cfg.output_dim = ds_train.output_dim
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        dir=get_original_cwd(),  # don't nest wandb inside hydra dir
    )

    # Instantiate the neural network
    network = hydra.utils.instantiate(cfg.network, ds_train=ds_train)

    # Instantiate SFR
    sfr = hydra.utils.instantiate(cfg.sfr, model=network)
    print(f"made sfr {sfr}")

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x.to(cfg.device))
        return sfr.likelihood.inv_link(f)
        # return torch.softmax(sfr.network(x.to(cfg.device)), dim=-1)

    @torch.no_grad()
    def loss_fn(data_loader: DataLoader):
        cum_loss = 0
        for X, y in data_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            cum_loss += loss
        return cum_loss

    def train_loop(sfr, data_loader: DataLoader):
        optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=cfg.lr)

        early_stopper = EarlyStopper(
            patience=int(cfg.early_stop.patience / cfg.logging_epoch_freq),
            min_delta=cfg.early_stop.min_delta,
        )

        best_nll = float("inf")
        for epoch in tqdm(list(range(cfg.n_epochs))):
            for X, y in data_loader:
                X = X.to(torch.float).to(cfg.device)
                y = y.to(cfg.device)
                loss = sfr.loss(X, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({"loss": loss})

            if epoch % cfg.logging_epoch_freq == 0:
                val_loss = loss_fn(val_loader)
                wandb.log({"val_loss": val_loss})
                train_metrics = compute_metrics(
                    pred_fn=map_pred_fn,
                    data_loader=train_loader,
                    # ds_test=ds_train,
                    # batch_size=cfg.batch_size,
                    device=cfg.device,
                )
                val_metrics = compute_metrics(
                    pred_fn=map_pred_fn,
                    data_loader=val_loader,
                    # ds_test=ds_val,
                    # batch_size=cfg.batch_size,
                    device=cfg.device,
                )
                test_metrics = compute_metrics(
                    pred_fn=map_pred_fn,
                    data_loader=test_loader,
                    # ds_test=ds_test,
                    # batch_size=cfg.batch_size,
                    device=cfg.device,
                )
                wandb.log({"train/": train_metrics})
                wandb.log({"val/": val_metrics})
                wandb.log({"test/": test_metrics})
                wandb.log({"epoch": epoch})

                if val_metrics["nll"] < best_nll:
                    checkpoint(sfr=sfr, optimizer=optimizer, save_dir=run.dir)
                    best_nll = val_metrics["nll"]
                    wandb.log({"best_test/": test_metrics})
                    wandb.log({"best_val/": val_metrics})
                if early_stopper(val_metrics["nll"]):  # (val_loss):
                    logger.info("Early stopping criteria met, stopping training...")
                    break
        return sfr

    def train_and_log(
        sfr: src.SFR,
        train_loader: DataLoader,
        name: str,
        inference_loader: Optional[DataLoader] = None,
    ):
        torch.set_default_dtype(torch.float)

        # Train NN
        sfr.float()
        sfr.train()
        start_time = time.time()
        sfr = train_loop(sfr, data_loader=train_loader)
        train_time = time.time() - start_time

        # Make everything double for inference
        torch.set_default_dtype(torch.double)

        # Log MAP before updates
        sfr.double()
        sfr.eval()
        log_map_metrics(
            sfr,
            test_loader_double,
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
        sfr.fit(train_loader=inference_loader)
        inference_time = time.time() - start_time
        logger.info("Finished fitting SFR")

        # Log SFR
        log_sfr_metrics(
            sfr,
            name=name,
            test_loader=test_loader_double,
            table_logger=table_logger,
            device=cfg.device,
            time=inference_time,
        )
        return sfr

    # Train on D1 and log
    sfr = train_and_log(sfr, train_loader=train_loader, name="Train D1")

    # Dual updates on D2 and log
    start_time = time.time()
    sfr.update_from_dataloader(data_loader=update_loader_double)
    update_inference_time = time.time() - start_time
    log_sfr_metrics(
        sfr,
        name="Train D1 -> Update D2",
        test_loader=test_loader_double,
        table_logger=table_logger,
        device=cfg.device,
        time=update_inference_time,
    )

    # Continue training on D1+D2 and log
    sfr = train_and_log(
        sfr, train_loader=train_and_update_loader, name="Train D1 -> Train D1+D2"
    )

    # Train on D1+D2 (from scratch) and log
    network = hydra.utils.instantiate(cfg.network, ds_train=ds_train)
    sfr = hydra.utils.instantiate(cfg.sfr, model=network)
    sfr = train_and_log(sfr, train_loader=train_and_update_loader, name="Train D1+D2")

    # Continue training on just D2 and log
    print("Continue training on just D2 and log")
    torch.set_default_dtype(torch.float)
    network = hydra.utils.instantiate(cfg.network, ds_train=ds_train)
    sfr.float()
    sfr.network.float()
    network.load_state_dict(sfr.network.state_dict())  # copy weights and stuff
    sfr_copy = hydra.utils.instantiate(cfg.sfr, model=network)
    _ = train_and_log(
        sfr_copy,
        train_loader=update_loader,
        name="Train D1 -> Train D2",
        inference_loader=train_and_update_loader,
    )

    # # Log table on W&B and save latex table as .tex
    # df = pd.DataFrame(table_logger.data)
    # print(df)
    # wandb.log({"Metrics": wandb.Table(data=df)})

    # df_latex = df.to_latex(escape=False)
    # print(df_latex)

    # with open("uci_table.tex", "w") as file:
    #     file.write(df_latex)
    #     wandb.save("uci_table.tex")


def log_map_metrics(
    sfr, test_loader, name: str, table_logger, device, time: float = None
):
    from experiments.sl.utils import compute_metrics

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x.to(device))
        return sfr.likelihood.inv_link(f)

    map_metrics = compute_metrics(
        pred_fn=map_pred_fn, data_loader=test_loader, device=device
    )
    table_logger.add_data(
        "NN MAP",
        metrics=map_metrics,
        num_inducing=None,
        prior_prec=sfr.prior.delta,
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
    from experiments.sl.inference import sfr_pred

    # nn_metrics = compute_metrics(
    #     pred_fn=sfr_pred(
    #         model=sfr, pred_type="nn", num_samples=num_samples, device=device
    #     ),
    #     data_loader=test_loader,
    #     device=device,
    # )
    # table_logger.add_data(
    #     "SFR (NN) " + name,
    #     metrics=nn_metrics,
    #     num_inducing=sfr.num_inducing,
    #     prior_prec=sfr.prior.delta,
    #     time=time,
    # )

    gp_metrics = compute_metrics(
        pred_fn=sfr_pred(
            model=sfr, pred_type="gp", num_samples=num_samples, device=device
        ),
        data_loader=test_loader,
        device=device,
    )
    table_logger.add_data(
        "SFR (GP)",
        metrics=gp_metrics,
        num_inducing=sfr.num_inducing,
        prior_prec=sfr.prior.delta,
        time=time,
        method=name,
    )


if __name__ == "__main__":
    main()  # pyright: ignore
