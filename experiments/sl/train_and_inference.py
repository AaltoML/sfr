#!/usr/bin/env python3
import logging
import os
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


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
            ]
        )

    def add_data(
        self,
        model_name: str,
        metrics: dict,
        prior_prec: float,
        num_inducing: Optional[int] = None,
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
        self.tbl.add_data(
            self.cfg.dataset.name,
            model_name,
            self.cfg.random_seed,
            num_inducing,
            metrics["acc"],
            metrics["nll"],
            metrics["ece"],
            prior_prec,
        )
        wandb.log({"Metrics": wandb.Table(data=pd.DataFrame(self.data))})


@hydra.main(
    version_base="1.3", config_path="./configs", config_name="train_and_inference"
)
def train_and_inference(cfg: DictConfig):
    from experiments.sl.cluster_train import train
    from hydra.utils import get_original_cwd

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))

    table_logger = TableLogger(cfg)

    torch.set_default_dtype(torch.float)

    # Load train/val/test data sets
    ds_train, ds_val, ds_test = hydra.utils.instantiate(
        cfg.dataset, dir=os.path.join(get_original_cwd(), "data")
    )

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

    # Train
    sfr = train(cfg)  # Train the NN

    # Make everything double for inference
    torch.set_default_dtype(torch.double)
    sfr.double()
    sfr.eval()

    ds_train, ds_val, ds_test = hydra.utils.instantiate(
        cfg.dataset,
        dir=os.path.join(get_original_cwd(), "data"),
        double=cfg.double_inference,
    )
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    # Log MAP NLPD
    # torch.cuda.empty_cache()
    log_map_metrics(sfr, test_loader, table_logger=table_logger, device=cfg.device)

    # Log Laplace BNN/GLM NLPD/ACC/ECE
    if cfg.run_laplace_flag:
        print("starting laplace")
        torch.cuda.empty_cache()
        for hessian_structure in cfg.hessian_structures:
            log_la_metrics(
                network=sfr.network,
                delta=sfr.prior.delta,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                table_logger=table_logger,
                device=cfg.device,
                posthoc_prior_opt=cfg.posthoc_prior_opt_laplace,
                hessian_structure=hessian_structure,
            )

    num_data = len(ds_train)
    logger.info(f"num_data: {num_data}")
    for num_inducing in cfg.num_inducings:
        torch.cuda.empty_cache()
        # Log SFR GP/NN NLPD/ACC/ECE
        log_sfr_metrics(
            network=sfr.network,
            output_dim=ds_train.output_dim,
            delta=sfr.prior.delta,  # TODO how to set this?
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            table_logger=table_logger,
            num_inducing=num_inducing,
            dual_batch_size=cfg.dual_batch_size,
            device=cfg.device,
            posthoc_prior_opt_grid=cfg.posthoc_prior_opt_grid,
            posthoc_prior_opt_bo=cfg.posthoc_prior_opt_bo,
            EPS=cfg.EPS,
            jitter=cfg.jitter,
        )

        # Log GP GP/NN NLPD/ACC/ECE
        log_gp_metrics(
            network=sfr.network,
            output_dim=ds_train.output_dim,
            delta=sfr.prior.delta,  # TODO how to set this?
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            table_logger=table_logger,
            num_inducing=num_inducing,
            dual_batch_size=cfg.dual_batch_size,
            device=cfg.device,
            posthoc_prior_opt_grid=cfg.posthoc_prior_opt_grid,
            posthoc_prior_opt_bo=cfg.posthoc_prior_opt_bo,
            EPS=cfg.EPS,
            jitter=cfg.jitter,
        )

    # Log table on W&B and save latex table as .tex
    df = pd.DataFrame(table_logger.data)
    print(df)
    wandb.log({"Metrics": wandb.Table(data=df)})

    df_latex = df.to_latex(escape=False)
    print(df_latex)

    with open("uci_table.tex", "w") as file:
        file.write(df_latex)
        wandb.save("uci_table.tex")


def log_map_metrics(sfr, test_loader, table_logger, device):
    from experiments.sl.utils import compute_metrics

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x.to(device))
        return sfr.likelihood.inv_link(f)

    map_metrics = compute_metrics(
        pred_fn=map_pred_fn, data_loader=test_loader, device=device
    )
    table_logger.add_data(
        "NN MAP", metrics=map_metrics, num_inducing=None, prior_prec=sfr.prior.delta
    )
    logger.info(f"map_metrics: {map_metrics}")


def log_sfr_metrics(
    network,
    output_dim,
    delta,
    train_loader,
    val_loader,
    test_loader,
    table_logger: TableLogger,
    num_inducing: int = 128,
    dual_batch_size: int = 1000,
    # device="cpu",
    device="cuda",
    posthoc_prior_opt_grid: bool = True,
    posthoc_prior_opt_bo: bool = True,
    num_samples=100,
    EPS=0.01,
    # EPS=0.0,
    jitter: float = 1e-6,
):
    import src
    from experiments.sl.inference import sfr_pred
    from experiments.sl.utils import compute_metrics, init_SFR_with_gaussian_prior

    likelihood = src.likelihoods.CategoricalLh(EPS=EPS)
    sfr = init_SFR_with_gaussian_prior(
        model=network,
        delta=delta,  # TODO what should this be
        likelihood=likelihood,
        output_dim=output_dim,
        num_inducing=num_inducing,
        dual_batch_size=dual_batch_size,
        jitter=jitter,
        device=device,
    )
    sfr.double()
    sfr.eval()
    logger.info("Fitting SFR...")
    sfr.fit(train_loader=train_loader)
    logger.info("Finished fitting SFR")

    nn_metrics = compute_metrics(
        pred_fn=sfr_pred(
            model=sfr, pred_type="nn", num_samples=num_samples, device=device
        ),
        data_loader=test_loader,
        device=device,
    )
    table_logger.add_data(
        "SFR (NN)",
        metrics=nn_metrics,
        num_inducing=num_inducing,
        prior_prec=sfr.prior.delta,
    )

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
        num_inducing=num_inducing,
        prior_prec=sfr.prior.delta,
    )

    # Get NLL for NN predict
    if posthoc_prior_opt_bo:
        sfr.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="bo",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=20,
        )
        nn_metrics_bo = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="nn", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "SFR (NN) BO",
            metrics=nn_metrics_bo,
            num_inducing=num_inducing,
            prior_prec=sfr.prior.delta,
        )

    if posthoc_prior_opt_grid:
        sfr.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="grid",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=50,
        )
        nn_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="nn", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "SFR (NN) GRID",
            metrics=nn_metrics,
            num_inducing=num_inducing,
            prior_prec=sfr.prior.delta,
        )
        # logger.info(f"map_metrics: {map_metrics}")

    # Get NLL for GP predict
    if posthoc_prior_opt_bo:
        sfr.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="bo",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=20,
        )
        gp_metrics_bo = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "SFR (GP) BO",
            metrics=gp_metrics_bo,
            num_inducing=num_inducing,
            prior_prec=sfr.prior.delta,
        )

    if posthoc_prior_opt_grid:
        sfr.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="grid",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=50,
        )
        gp_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "SFR (GP) GRID",
            metrics=gp_metrics,
            num_inducing=num_inducing,
            prior_prec=sfr.prior.delta,
        )


def log_gp_metrics(
    network,
    output_dim,
    delta,
    train_loader,
    val_loader,
    test_loader,
    table_logger: TableLogger,
    num_inducing: int = 128,
    dual_batch_size: int = 1000,
    # device="cpu",
    device="cuda",
    posthoc_prior_opt_grid: bool = True,
    posthoc_prior_opt_bo: bool = True,
    num_samples=100,
    EPS=0.01,
    jitter: float = 1e-6,
):
    import src
    from experiments.sl.inference import sfr_pred
    from experiments.sl.utils import (
        compute_metrics,
        init_NN2GPSubset_with_gaussian_prior,
    )

    # if output_dim <= 2:
    #     likelihood = src.likelihoods.BernoulliLh(EPS=0.0)
    # else:
    #     likelihood = src.likelihoods.CategoricalLh(EPS=0.0)
    # likelihood = src.likelihoods.CategoricalLh(EPS=0.0)
    likelihood = src.likelihoods.CategoricalLh(EPS=EPS)
    gp = init_NN2GPSubset_with_gaussian_prior(
        model=network,
        delta=delta,  # TODO what should this be
        likelihood=likelihood,
        output_dim=output_dim,
        subset_size=num_inducing,
        dual_batch_size=dual_batch_size,
        jitter=jitter,
        # jitter=1e-4,
        device=device,
    )
    gp = gp.double()
    gp.eval()
    logger.info("Fitting GP...")
    gp.fit(train_loader=train_loader)
    logger.info("Finished fitting GP")

    nn_metrics_bo = compute_metrics(
        pred_fn=sfr_pred(
            model=gp, pred_type="nn", num_samples=num_samples, device=device
        ),
        data_loader=test_loader,
        device=device,
    )
    table_logger.add_data(
        "GP Subest (NN)",
        metrics=nn_metrics_bo,
        num_inducing=num_inducing,
        prior_prec=gp.prior.delta,
    )

    nn_metrics_bo = compute_metrics(
        pred_fn=sfr_pred(
            model=gp, pred_type="gp", num_samples=num_samples, device=device
        ),
        data_loader=test_loader,
        device=device,
    )
    table_logger.add_data(
        "GP Subest (GP)",
        metrics=nn_metrics_bo,
        num_inducing=num_inducing,
        prior_prec=gp.prior.delta,
    )

    if posthoc_prior_opt_bo:
        gp.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="bo",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=20,
        )
        nn_metrics_bo = compute_metrics(
            pred_fn=sfr_pred(
                model=gp, pred_type="nn", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "GP Subest (NN) BO",
            metrics=nn_metrics_bo,
            num_inducing=num_inducing,
            prior_prec=gp.prior.delta,
        )

    if posthoc_prior_opt_grid:
        gp.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="grid",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=50,
        )
        nn_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=gp, pred_type="nn", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "GP Subest (NN) GRID",
            metrics=nn_metrics,
            num_inducing=num_inducing,
            prior_prec=gp.prior.delta,
        )

    if posthoc_prior_opt_bo:
        gp.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="bo",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=20,
        )
        gp_metrics_bo = compute_metrics(
            pred_fn=sfr_pred(
                model=gp, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "GP Subest (GP) BO",
            metrics=gp_metrics_bo,
            num_inducing=num_inducing,
            prior_prec=gp.prior.delta,
        )

    if posthoc_prior_opt_grid:
        gp.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="grid",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=40,
        )
        gp_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=gp, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "GP Subest (GP) GRID",
            metrics=gp_metrics,
            num_inducing=num_inducing,
            prior_prec=gp.prior.delta,
        )


def log_la_metrics(
    network,
    delta: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    table_logger: TableLogger,
    device: str,
    posthoc_prior_opt: bool = True,
    num_samples: int = 100,
    hessian_structure: str = "kron",
):
    import laplace
    from experiments.sl.inference import la_pred
    from experiments.sl.utils import compute_metrics

    la = laplace.Laplace(
        likelihood="classification",
        subset_of_weights="all",
        hessian_structure=hessian_structure,
        sigma_noise=1,
        # prior_precision: ???
        backend=laplace.curvature.asdl.AsdlGGN,
        model=network,
    )
    print(f"la {la}")
    la.prior_precision = delta
    logger.info("Fitting Laplace...")
    la.fit(train_loader)
    logger.info("Finished fitting Laplace")

    bnn_pred_fn = la_pred(
        model=la,
        pred_type="nn",
        link_approx="mc",
        num_samples=num_samples,
        device=device,
    )
    bnn_metrics = compute_metrics(
        pred_fn=bnn_pred_fn, data_loader=test_loader, device=device
    )
    table_logger.add_data(
        f"BNN {hessian_structure}", metrics=bnn_metrics, prior_prec=la.prior_precision
    )
    glm_pred_fn = la_pred(
        model=la,
        pred_type="glm",
        link_approx="mc",
        num_samples=num_samples,
        device=device,
    )
    glm_metrics = compute_metrics(
        pred_fn=glm_pred_fn, data_loader=test_loader, device=device
    )
    table_logger.add_data(
        f"GLM {hessian_structure}", metrics=glm_metrics, prior_prec=la.prior_precision
    )

    # Get NLL for BNN predict
    if posthoc_prior_opt:
        la.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="CV",  # "marglik"
            log_prior_prec_min=1,
            log_prior_prec_max=10,
            # log_prior_prec_max=5,
            grid_size=40,
        )
        bnn_pred_fn = la_pred(
            model=la,
            pred_type="nn",
            link_approx="mc",
            num_samples=num_samples,
            device=device,
        )
        bnn_metrics = compute_metrics(
            pred_fn=bnn_pred_fn, data_loader=test_loader, device=device
        )
        table_logger.add_data(
            f"BNN {hessian_structure} GRID",
            metrics=bnn_metrics,
            prior_prec=la.prior_precision,
        )

    # Get NLL for GLM predict
    if posthoc_prior_opt:
        la.optimize_prior_precision(
            pred_type="glm",
            val_loader=val_loader,
            method="CV",  # "marglik"
            log_prior_prec_min=1,
            log_prior_prec_max=10,
            grid_size=40,
        )
        glm_pred_fn = la_pred(
            model=la,
            pred_type="glm",
            link_approx="mc",
            num_samples=num_samples,
            device=device,
        )
        glm_metrics = compute_metrics(
            pred_fn=glm_pred_fn, data_loader=test_loader, device=device
        )
        table_logger.add_data(
            f"GLM {hessian_structure} GRID",
            metrics=glm_metrics,
            prior_prec=la.prior_precision,
        )


if __name__ == "__main__":
    train_and_inference()  # pyright: ignore
