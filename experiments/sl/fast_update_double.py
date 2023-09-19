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
from experiments.sl.utils import (
    compute_metrics,
    EarlyStopper,
    set_seed_everywhere,
    compute_metrics_regression,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
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
            "mse": [],
            "prior_prec": [],
            "time": [],
            "method": [],
        }
        # self.tbl = wandb.Table(
        #     columns=[
        #         "dataset",
        #         "model",
        #         "seed",
        #         "num_inducing",
        #         "acc",
        #         "nlpd",
        #         "ece",
        #         "mse",
        #         "prior_prec",
        #         "time",
        #         "method",
        #     ]
        # )

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
        # self.tbl.add_data(
        #     self.cfg.dataset.name,
        #     model_name,
        #     self.cfg.random_seed,
        #     num_inducing,
        #     self.data["acc"],
        #     self.data["nlpd"],
        #     self.data["ece"],
        #     self.data["mse"],
        #     prior_prec,
        #     time,
        #     method,
        # )
        # wandb.log({"Metrics": self.tbl})
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

    # torch.set_default_dtype(torch.float)
    torch.set_default_dtype(torch.double)

    # Load train/val/test/update data sets
    ds_train, ds_val, ds_test, ds_update = hydra.utils.instantiate(
        cfg.dataset,
        dir=os.path.join(get_original_cwd(), "data"),
        # double=cfg.double_inference,
        # double=False,
        double=True,
        # train_update_split=cfg.train_update_split,
    )
    print(f"num_data: {ds_train.data.shape[0]}")
    print(f"D: {ds_train.data.shape[1]}")
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)
    update_loader = DataLoader(ds_update, batch_size=cfg.batch_size, shuffle=True)
    train_and_update_loader = DataLoader(
        ConcatDataset([ds_train, ds_update]), batch_size=cfg.batch_size, shuffle=True
    )

    # Init Weight and Biases
    cfg.output_dim = ds_train.output_dim
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
    train_loader = make_data_loader_double(
        data_loader=train_loader, likelihood=likelihood
    )
    val_loader = make_data_loader_double(data_loader=val_loader, likelihood=likelihood)
    test_loader = make_data_loader_double(
        data_loader=test_loader, likelihood=likelihood
    )
    update_loader = make_data_loader_double(
        data_loader=update_loader, likelihood=likelihood
    )
    # train_and_update_loader = make_data_loader_double(
    #     data_loader=train_and_update_loader, likelihood=likelihood
    # )
    # train_loader_double = make_data_loader_double(
    #     data_loader=train_loader, likelihood=likelihood
    # )
    # test_loader_double = make_data_loader_double(
    #     data_loader=test_loader, likelihood=likelihood
    # )
    # val_loader_double = make_data_loader_double(
    #     data_loader=val_loader, likelihood=likelihood
    # )
    # print(next(iter(test_loader_double))[0].dtype)
    # print(next(iter(test_loader_double))[1].dtype)
    # update_loader_double = make_data_loader_double(
    #     data_loader=update_loader, likelihood=likelihood
    # )

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x.to(cfg.device))
        return sfr.likelihood.inv_link(f)
        # return torch.softmax(sfr.network(x.to(cfg.device)), dim=-1)

    @torch.no_grad()
    def loss_fn(data_loader: DataLoader):
        losses = []
        for X, y in data_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            losses.append(loss)
        losses = torch.stack(losses, 0)
        cum_loss = torch.mean(losses, 0)
        return cum_loss

    def train_loop(sfr, data_loader: DataLoader):
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

        best_nll = float("inf")
        best_loss = float("inf")
        for epoch in tqdm(list(range(cfg.n_epochs))):
            for X, y in data_loader:
                X = X.to(cfg.device)
                # X = X.to(torch.float).to(cfg.device)
                y = y.to(cfg.device)
                # print(f"X {X.dtype}")
                # print(f"y {y.dtype}")
                # print(f"X {X.dtype}")
                # print(f"y {y.dtype}")
                loss = sfr.loss(X, y)
                # print(f"loss {loss.dtype}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cfg.wandb.use_wandb:
                    wandb.log({"loss": loss})
                    wandb.log({"log_sigma_noise": sfr.likelihood.sigma_noise})

            val_loss = loss_fn(val_loader)
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

                # if val_metrics["nll"] < best_nll:
                #     checkpoint(sfr=sfr, optimizer=optimizer, save_dir=run.dir)
                #     best_nll = val_metrics["nll"]
                #     wandb.log({"best_test/": test_metrics})
                #     wandb.log({"best_val/": val_metrics})
                # if early_stopper(val_metrics["nll"]):  # (val_loss):
                #     logger.info("Early stopping criteria met, stopping training...")
                #     break
            if val_loss < best_loss:
                best_ckpt_fname = checkpoint(
                    sfr=sfr, optimizer=optimizer, save_dir=run_dir
                )
                best_loss = val_loss
                # wandb.log({"best_test/": test_metrics})
                # wandb.log({"best_val/": val_metrics})
            if early_stopper(val_loss):  # (val_loss):
                logger.info("Early stopping criteria met, stopping training...")
                break

        # Load checkpoint
        ckpt = torch.load(best_ckpt_fname)
        # print(f"ckpt {ckpt}")
        # print(f"sfr {[p for p in sfr.parameters()]}")
        sfr.load_state_dict(ckpt["model"])
        # print(f"sfr loaded {[p for p in sfr.parameters()]}")
        return sfr

    def train_and_log(
        sfr: src.SFR,
        train_loader: DataLoader,
        val_loader: DataLoader,
        name: str,
        inference_loader: Optional[DataLoader] = None,
    ):
        # torch.set_default_dtype(torch.float)
        print(f"yo {next(iter(train_loader))[0].dtype}")
        print(f"yo {next(iter(train_loader))[1].dtype}")
        print(f"yo {next(iter(val_loader))[0].dtype}")
        print(f"yo {next(iter(val_loader))[1].dtype}")
        # breakpoint()

        # Train NN
        # sfr.float()
        sfr.train()
        start_time = time.time()
        sfr = train_loop(sfr, data_loader=train_loader)
        train_time = time.time() - start_time

        log_map_metrics(
            sfr,
            test_loader,
            name=name,
            table_logger=table_logger,
            device=cfg.device,
            time=train_time,
        )

        # Make everything double for inference
        # torch.set_default_dtype(torch.double)

        # Log MAP before updates
        # sfr.double()
        # sfr.network.double()
        sfr.eval()
        log_map_metrics(
            sfr,
            test_loader,
            # test_loader_double,
            name=name,
            table_logger=table_logger,
            device=cfg.device,
            time=train_time,
        )

        # TODO sample Z from train+update
        #

        # Fit SFR
        logger.info("Fitting SFR...")
        if inference_loader is None:
            inference_loader = train_loader
        start_time = time.time()

        Z = torch.concat([ds_train.data, ds_update.data], 0)
        print(f"Z {Z.shape}")
        indices = torch.randperm(Z.shape[0])[: sfr.num_inducing]
        sfr.Z = Z[indices.to(sfr.device)].to(sfr.device)
        all_train = DataLoader(
            train_loader.dataset, batch_size=len(train_loader.dataset)
        )
        sfr.train_data = next(iter(all_train))

        # Z = torch.concat([ds_train.data, ds_update.data], 0)
        # print(f"Z {Z.shape}")
        # # breakpoint()
        # indices = torch.randperm(Z.shape[0])[: sfr.num_inducing]
        # sfr.Z = Z[indices.to(sfr.device)].to(sfr.device)
        # # print(f"train_loader.dataset[0] {train_loader.dataset[0].shape}")
        # #
        # sfr.train_data = (
        #     train_loader.dataset.data.double(),
        #     train_loader.dataset.targets.double(),
        # )
        sfr._build_sfr()
        # breakpoint()
        # sfr.fit(train_loader=train_loader)
        # sfr.fit(train_loader=inference_loader)
        inference_time = time.time() - start_time
        logger.info("Finished fitting SFR")
        log_sfr_metrics(
            sfr,
            name=name,
            test_loader=test_loader,
            # test_loader=test_loader_double,
            table_logger=table_logger,
            device=cfg.device,
            time=inference_time,
        )

        # # sfr.prior.prior_precision = prior_prec
        # num_bo_trials = 10
        # num_bo_trials = 50
        # # num_bo_trials = 30
        # sfr.optimize_prior_precision(
        #     pred_type="gp",
        #     val_loader=val_loader,
        #     method="grid",
        #     # method="bo",
        #     prior_prec_min=1e-4,
        #     prior_prec_max=1.0,
        #     # prior_prec_min=1e-6,
        #     # prior_prec_max=1e-3,
        #     num_trials=num_bo_trials,
        # )

        # # Log SFR
        # print(f"test_loader_double {test_loader_double.dataset}")
        # print(f"test_loader_double {test_loader_double.dataset.data.dtype}")
        # print(f"test_loader_double {test_loader_double.dataset.targets.dtype}")
        # print(f"test_loader_double {next(iter(test_loader_double))[0].dtype}")
        # print(f"test_loader_double {next(iter(test_loader_double))[1].dtype}")
        # log_sfr_metrics(
        #     sfr,
        #     name=name + " tuning",
        #     test_loader=test_loader_double,
        #     table_logger=table_logger,
        #     device=cfg.device,
        #     time=inference_time,
        # )
        return sfr

    # Train on D1 and log
    sfr = train_and_log(
        sfr,
        train_loader=train_loader,
        val_loader=val_loader,
        # val_loader=val_loader_double,
        # inference_loader=train_loader_double,
        name="Train D1",
    )

    # def plot(i):
    #     min = ds_train.data[:, i].min()
    #     max = ds_train.data[:, i].max()
    #     Xtest = torch.ones(300, ds_train.data.shape[-1])
    #     print(f"Xtest {Xtest.shape}")
    #     mean = torch.mean(ds_train.data, 0).reshape(1, -1)
    #     Xtest = Xtest * mean
    #     print(f"mean {mean.shape}")
    #     Xtest[:, i] = torch.linspace(min, max, 300)
    #     f_mean, f_var = sfr(Xtest, pred_type="gp")
    #     f_mean = f_mean.detach().numpy()
    #     f_var = f_var.detach().numpy()

    #     plt.figure()
    #     plt.plot(Xtest[:, i], f_mean, color="cyan")
    #     # plt.fill_between(Xtest[:, i], (f_mean - f_var)[:, 0], (f_mean + f_var)[:, 0], color="cyan")
    #     plt.scatter(ds_train.data[:, i], ds_train.targets[:, i], color="k", marker="x")
    #     plt.savefig(f"reg_dim={i}_.pdf")

    # for i in range(ds_train.data.shape[-1]):
    #     plot(i)

    # breakpoint()
    # exit()

    # Dual updates on D2 and log
    start_time = time.time()
    # sfr.update(data_loader=update_loader_double)
    sfr.update(data_loader=update_loader)
    update_inference_time = time.time() - start_time
    log_sfr_metrics(
        sfr,
        name="Train D1 -> Update D2",
        test_loader=test_loader,
        # test_loader=test_loader_double,
        table_logger=table_logger,
        device=cfg.device,
        time=update_inference_time,
    )
    # breakpoint()
    # exit()
    # num_bo_trials = 50
    # # num_bo_trials = 30
    # sfr.optimize_prior_precision(
    #     pred_type="gp",
    #     # pred_type="nn",
    #     # val_loader=val_loader_double,
    #     val_loader=val_loader,
    #     method="grid",
    #     # method="bo",
    #     # prior_prec_min=1e-4,
    #     prior_prec_min=1e-9,
    #     prior_prec_max=1.0,
    #     # prior_prec_min=1e-6,
    #     # prior_prec_max=1e-3,
    #     num_trials=num_bo_trials,
    # )
    # update_inference_time = time.time() - start_time
    # log_sfr_metrics(
    #     sfr,
    #     name="Train D1 -> Update D2 + tuning",
    #     test_loader=test_loader,
    #     # test_loader=test_loader_double,
    #     table_logger=table_logger,
    #     device=cfg.device,
    #     time=update_inference_time,
    # )
    # exit()

    # Continue training on D1+D2 and log
    sfr = train_and_log(
        sfr,
        train_loader=train_and_update_loader,
        val_loader=val_loader,
        # val_loader=val_loader_double,
        name="Train D1 -> Train D1+D2",
    )

    # Train on D1+D2 (from scratch) and log
    network = hydra.utils.instantiate(cfg.network, ds_train=ds_train)
    sfr = hydra.utils.instantiate(cfg.sfr, model=network)
    sfr = train_and_log(
        sfr,
        train_loader=train_and_update_loader,
        val_loader=val_loader,
        # val_loader=val_loader_double,
        name="Train D1+D2",
    )

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
        val_loader=val_loader,
        # val_loader=val_loader_double,
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
    #     prior_prec=sfr.prior.prior_precision,
    #     time=time,
    # )

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
