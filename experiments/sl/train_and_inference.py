#!/usr/bin/env python3
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base="1.3", config_path="./configs", config_name="train_and_inference"
)
def train_and_inference(cfg: DictConfig):
    import numpy as np
    import pandas as pd
    import wandb
    from experiments.sl.cluster_train import train
    from hydra.utils import get_original_cwd
    from torch.utils.data import DataLoader

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))

    # Data dictionary used to make pd.DataFrame
    data = {
        "dataset": [],
        "model": [],
        "seed": [],
        "num_inducing": [],
        "acc": [],
        "nlpd": [],
        "ece": [],
    }
    tbl = wandb.Table(
        columns=["dataset", "model", "seed", "num_inducing", "acc", "nlpd", "ece"]
    )

    def add_data(model_name, acc, nll, ece, num_inducing):
        "Add NLL to data dict and wandb table"
        # dataset = dataset_name.replace("_uci", "").title()
        data["dataset"].append(cfg.dataset.name)
        data["model"].append(model_name)
        data["seed"].append(cfg.random_seed)
        data["num_inducing"].append(num_inducing)
        data["acc"].append(acc)
        data["nlpd"].append(nll)
        data["ece"].append(ece)
        tbl.add_data(
            cfg.dataset.name, model_name, cfg.random_seed, num_inducing, acc, nll, ece
        )
        wandb.log({"NLPD": wandb.Table(data=pd.DataFrame(data))})
        return data

    torch.set_default_dtype(torch.float)

    # Load train/val/test data sets
    ds_train, ds_val, ds_test = hydra.utils.instantiate(
        # cfg.dataset, dir=os.path.join(get_original_cwd(), "data"), double=True
        cfg.dataset,
        dir=os.path.join(get_original_cwd(), "data"),
    )
    # ds_train.data = ds_train.data.to(torch.double)
    # ds_val.data = ds_val.data.to(torch.double)
    # ds_test.data = ds_test.data.to(torch.double)
    # ds_train.targets = ds_train.targets.long()
    # ds_val.targets = ds_val.targets.long()
    # ds_test.targets = ds_test.targets.long()

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
    cfg.n_epochs = 1
    sfr = train(cfg)  # Train the NN

    # Log MAP NLPD
    # torch.cuda.empty_cache()
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)
    map_metrics = calc_map_metrics(sfr, test_loader, device=cfg.device)
    data = add_data(
        model_name="NN MAP",
        acc=map_metrics["acc"],
        nll=map_metrics["nll"],
        ece=map_metrics["ece"],
        num_inducing=None,
    )
    logger.info(f"map_metrics: {map_metrics}")

    torch.set_default_dtype(torch.double)

    cfg.device = "cpu"
    print("Using device: {}".format(cfg.device))
    # sfr.to(cfg.device)
    # sfr.network.to(cfg.device)
    # sfr.network = sfr.network.double()
    sfr.double()
    sfr.eval()
    # sfr.cpu()
    # sfr.network.cpu()

    # ds_train, ds_val, ds_test = hydra.utils.instantiate(
    #     cfg.dataset, dir=os.path.join(get_original_cwd(), "data"), double=True
    # )
    # # ds_train.data = ds_train.data.to(torch.double)
    # # ds_val.data = ds_val.data.to(torch.double)
    # # ds_test.data = ds_test.data.to(torch.double)
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    # test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    # # Log MAP NLPD
    # # torch.cuda.empty_cache()
    # map_metrics = calc_map_metrics(sfr, test_loader, device=cfg.device)
    # data = add_data(
    #     model_name="NN MAP",
    #     acc=map_metrics["acc"],
    #     nll=map_metrics["nll"],
    #     ece=map_metrics["ece"],
    #     num_inducing=None,
    # )
    # print(f"map_metrics: {map_metrics}")

    # # Log Laplace BNN/GLM NLPD
    # # print("starting laplace")
    # torch.cuda.empty_cache()
    # la_metrics = calc_la_metrics(
    #     network=sfr.network,
    #     delta=sfr.prior.delta,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     device=cfg.device,
    #     posthoc_prior_opt=cfg.posthoc_prior_opt,
    # )
    # data = add_data(
    #     model_name="BNN",
    #     acc=la_metrics["bnn"]["acc"],
    #     nll=la_metrics["bnn"]["nll"],
    #     ece=la_metrics["bnn"]["ece"],
    #     num_inducing=None,
    # )
    # data = add_data(
    #     model_name="GLM",
    #     acc=la_metrics["glm"]["acc"],
    #     nll=la_metrics["glm"]["nll"],
    #     ece=la_metrics["glm"]["ece"],
    #     num_inducing=None,
    # )
    # print(f"la_metrics {la_metrics}")

    num_data = len(ds_train)
    print(f"num_data: {num_data}")
    for num_inducing in cfg.num_inducings:
        if num_inducing >= num_data:
            break
        torch.cuda.empty_cache()
        # Log SFR GP/NN NLPD
        sfr_metrics = calc_sfr_metrics(
            network=sfr.network,
            output_dim=ds_train.output_dim,
            delta=sfr.prior.delta,  # TODO how to set this?
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_inducing=num_inducing,
            device=cfg.device,
            posthoc_prior_opt=cfg.posthoc_prior_opt,
        )
        data = add_data(
            model_name="SFR (NN)",
            acc=sfr_metrics["nn"]["acc"],
            nll=sfr_metrics["nn"]["nll"],
            ece=sfr_metrics["nn"]["ece"],
            num_inducing=num_inducing,
        )
        data = add_data(
            model_name="SFR (GP)",
            acc=sfr_metrics["gp"]["acc"],
            nll=sfr_metrics["gp"]["nll"],
            ece=sfr_metrics["gp"]["ece"],
            num_inducing=num_inducing,
        )
        logger.info(f"sfr_metrics: {sfr_metrics}")

        # Log GP GP/NN NLPD
        gp_metrics = calc_gp_metrics(
            network=sfr.network,
            output_dim=ds_train.output_dim,
            delta=sfr.prior.delta,  # TODO how to set this?
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_inducing=num_inducing,
            device=cfg.device,
            posthoc_prior_opt=cfg.posthoc_prior_opt,
        )
        data = add_data(
            model_name="GP Subset (NN)",
            acc=gp_metrics["nn"]["acc"],
            nll=gp_metrics["nn"]["nll"],
            ece=gp_metrics["nn"]["ece"],
            num_inducing=num_inducing,
        )
        data = add_data(
            model_name="GP Subset (GP)",
            acc=gp_metrics["gp"]["acc"],
            nll=gp_metrics["gp"]["nll"],
            ece=gp_metrics["gp"]["ece"],
            num_inducing=num_inducing,
        )
        print(f"gp_metrics: {gp_metrics}")

    df = pd.DataFrame(data)
    wandb.log({"NLPD raw": wandb.Table(data=df)})
    print(df)

    with open("uci_table.tex", "w") as file:
        file.write(df.to_latex(escape=False))
        wandb.save("uci_table.tex")

    # Print the LaTeX table
    print(df.to_latex(escape=False))


def calc_map_metrics(sfr, test_loader, device):
    from experiments.sl.utils import compute_metrics

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        print("x.dtype")
        print(x.dtype)
        f = sfr.network(x.to(device))
        return sfr.likelihood.inv_link(f)

    map_metrics = compute_metrics(
        pred_fn=map_pred_fn, data_loader=test_loader, device=device
    )
    return map_metrics


def calc_sfr_metrics(
    network,
    output_dim,
    delta,
    train_loader,
    val_loader,
    test_loader,
    num_inducing=128,
    device="cpu",
    posthoc_prior_opt: bool = True,
    num_samples=100,
    EPS=0.01,
    # EPS=0.0,
):
    import src
    from experiments.sl.inference import sfr_pred
    from experiments.sl.utils import compute_metrics, init_SFR_with_gaussian_prior

    # if output_dim <= 2:
    #     # likelihood = src.likelihoods.BernoulliLh()
    #     likelihood = src.likelihoods.BernoulliLh(EPS=0.0)
    #     # likelihood = src.likelihoods.BernoulliLh(EPS=0.0004)
    # else:
    #     likelihood = src.likelihoods.CategoricalLh(EPS=0.0)
    likelihood = src.likelihoods.CategoricalLh(EPS=EPS)
    sfr = init_SFR_with_gaussian_prior(
        model=network,
        delta=delta,  # TODO what should this be
        likelihood=likelihood,
        output_dim=output_dim,
        num_inducing=num_inducing,
        dual_batch_size=2000,
        jitter=1e-8,
        device=device,
    )
    sfr.double()
    sfr.eval()
    logger.info("Fitting SFR...")
    sfr.fit(train_loader=train_loader)
    logger.info("Finished fitting SFR")

    # Get NLL for NN predict
    if posthoc_prior_opt:
        sfr.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="grid",
            log_prior_prec_min=-8,
            log_prior_prec_max=-1,
            grid_size=100,
        )
    nn_metrics = compute_metrics(
        pred_fn=sfr_pred(model=sfr, pred_type="nn", num_samples=num_samples),
        data_loader=test_loader,
        device=device,
    )

    # Get NLL for GP predict
    if posthoc_prior_opt:
        sfr.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="grid",
            log_prior_prec_min=-8,
            log_prior_prec_max=-1,
            grid_size=100,
        )
    gp_metrics = compute_metrics(
        pred_fn=sfr_pred(model=sfr, pred_type="gp", num_samples=num_samples),
        data_loader=test_loader,
        device=device,
    )
    return {"nn": nn_metrics, "gp": gp_metrics}


def calc_gp_metrics(
    network,
    output_dim,
    delta,
    train_loader,
    val_loader,
    test_loader,
    num_inducing=128,
    device="cpu",
    posthoc_prior_opt: bool = True,
    num_samples=100,
    EPS=0.01,
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
        dual_batch_size=2000,
        jitter=1e-8,
        # jitter=1e-4,
        device=device,
    )
    gp = gp.double()
    gp.eval()
    logger.info("Fitting GP...")
    gp.fit(train_loader=train_loader)
    logger.info("Finished fitting GP")

    if posthoc_prior_opt:
        gp.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="grid",
            # log_prior_prec_min=-10,
            # log_prior_prec_max=5,
            # grid_size=50,
            log_prior_prec_min=-8,
            log_prior_prec_max=-1,
            grid_size=100,
        )
    nn_metrics = compute_metrics(
        pred_fn=sfr_pred(model=gp, pred_type="nn", num_samples=num_samples),
        data_loader=test_loader,
        device=device,
    )
    if posthoc_prior_opt:
        gp.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="grid",
            # log_prior_prec_min=-10,
            # log_prior_prec_max=5,
            # grid_size=50,
            log_prior_prec_min=-8,
            log_prior_prec_max=-1,
            grid_size=100,
        )
    gp_metrics = compute_metrics(
        pred_fn=sfr_pred(model=gp, pred_type="gp", num_samples=num_samples),
        data_loader=test_loader,
        device=device,
    )
    return {"nn": nn_metrics, "gp": gp_metrics}


def calc_la_metrics(
    network,
    delta,
    train_loader,
    val_loader,
    test_loader,
    device,
    posthoc_prior_opt: bool = True,
    num_samples=100,
):
    import laplace
    from experiments.sl.inference import la_pred
    from experiments.sl.utils import compute_metrics

    la = laplace.Laplace(
        likelihood="classification",
        subset_of_weights="all",
        hessian_structure="full",
        # hessian_structure="diag",
        sigma_noise=1,
        # prior_precision: ???
        # backend=laplace.curvature.backpack.BackPackGGN,
        backend=laplace.curvature.asdl.AsdlGGN,
        model=network,
    )
    la.prior_precision = delta
    logger.info("Fitting Laplace...")
    la.fit(train_loader)
    logger.info("Finished fitting Laplace")

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
        model=la, pred_type="nn", link_approx="mc", num_samples=num_samples
    )
    bnn_metrics = compute_metrics(
        pred_fn=bnn_pred_fn, data_loader=test_loader, device=device
    )
    print(f"bnn_metrics: {bnn_metrics}")

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
        model=la, pred_type="glm", link_approx="mc", num_samples=num_samples
    )
    glm_metrics = compute_metrics(
        pred_fn=glm_pred_fn, data_loader=test_loader, device=device
    )
    print(f"glm_metrics: {glm_metrics}")
    return {"glm": glm_metrics, "bnn": bnn_metrics}


if __name__ == "__main__":
    train_and_inference()  # pyright: ignore
