#!/usr/bin/env python3
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base="1.3", config_path="./configs", config_name="train_and_inference"
)
def make_uci_table(cfg: DictConfig):
    import numpy as np
    import pandas as pd
    import wandb
    from experiments.sl.cluster_train import train
    from experiments.sl.utils import get_uci_dataset
    from hydra.utils import get_original_cwd
    from torch.utils.data import DataLoader

    # Data dictionary used to make pd.DataFrame
    data = {"dataset": [], "model": [], "seed": [], "num_inducing": [], "result": []}
    tbl = wandb.Table(columns=["dataset", "model", "seed", "num_inducing", "result"])

    def add_data(model_name, nll, num_inducing):
        "Add NLL to data dict and wandb table"
        # dataset = dataset_name.replace("_uci", "").title()
        data["dataset"].append(cfg.dataset.name)
        data["model"].append(model_name)
        data["seed"].append(cfg.random_seed)
        data["num_inducing"].append(num_inducing)
        data["result"].append(nll)
        tbl.add_data(cfg.dataset.name, model_name, cfg.random_seed, num_inducing, nll)
        wandb.log({"NLPD": wandb.Table(data=pd.DataFrame(data))})
        return data

    torch.set_default_dtype(torch.float)

    # Load train/val/test data sets
    ds_train, ds_val, ds_test = hydra.utils.instantiate(
        cfg.dataset, dir=os.path.join(get_original_cwd(), "data")
    )
    ds_train.data = ds_train.data.to(torch.double)
    ds_val.data = ds_val.data.to(torch.double)
    ds_test.data = ds_test.data.to(torch.double)
    ds_train.targets = ds_train.targets.long()
    ds_val.targets = ds_val.targets.long()
    ds_test.targets = ds_test.targets.long()

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

    torch.set_default_dtype(torch.double)

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    # sfr.network = sfr.network.double()
    sfr = sfr.double()
    sfr.eval()

    # Log MAP NLPD
    map_nll = calc_map_nll(sfr, test_loader, device=cfg.device)
    data = add_data(model_name="NN MAP", nll=map_nll, num_inducing=None)
    print(f"map_nll: {map_nll}")

    # Log Laplace BNN/GLM NLPD
    # print("starting laplace")
    la_nlls = calc_la_metrics(
        network=sfr.network,
        delta=sfr.prior.delta,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=cfg.device,
        posthoc_prior_opt=cfg.posthoc_prior_opt,
    )
    data = add_data(model_name="BNN", nll=la_nlls["bnn"], num_inducing=None)
    data = add_data(model_name="GLM", nll=la_nlls["glm"], num_inducing=None)
    print(f"la_nlls {la_nlls}")

    for num_inducing in cfg.num_inducings:
        # Log SFR GP/NN NLPD
        sfr_nlls = calc_sfr_nll(
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
            model_name="SFR (NN)", nll=sfr_nlls["nn"], num_inducing=num_inducing
        )
        data = add_data(
            model_name="SFR (GP)", nll=sfr_nlls["gp"], num_inducing=num_inducing
        )
        print(f"sfr_nlls: {sfr_nlls}")

        # Log GP GP/NN NLPD
        gp_nlls = calc_gp_nll(
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
            model_name="GP Subset (NN)", nll=gp_nlls["nn"], num_inducing=num_inducing
        )
        data = add_data(
            model_name="GP Subset (GP)", nll=gp_nlls["gp"], num_inducing=num_inducing
        )
        print(f"gp_nlls: {gp_nlls}")

    df = pd.DataFrame(data)
    wandb.log({"NLPD raw": wandb.Table(data=df)})
    print(df)

    with open("uci_table.tex", "w") as file:
        file.write(df.to_latex(escape=False))
        wandb.save("uci_table.tex")

    # Print the LaTeX table
    print(df.to_latex(escape=False))


def calc_map_nll(sfr, test_loader, device):
    from experiments.sl.utils import compute_metrics

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x.to(device))
        return sfr.likelihood.inv_link(f)

    map_nll = compute_metrics(
        pred_fn=map_pred_fn, data_loader=test_loader, device=device
    )["nll"]
    return map_nll


def calc_sfr_nll(
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
    likelihood = src.likelihoods.CategoricalLh(EPS=0.0)
    sfr = init_SFR_with_gaussian_prior(
        model=network,
        delta=delta,  # TODO what should this be
        likelihood=likelihood,
        output_dim=output_dim,
        num_inducing=num_inducing,
        dual_batch_size=2000,
        jitter=1e-4,
        device=device,
    )
    sfr = sfr.double()
    sfr.eval()
    sfr.fit(train_loader=train_loader)

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
    nn_nll = compute_metrics(
        pred_fn=sfr_pred(model=sfr, pred_type="nn", num_samples=num_samples),
        data_loader=test_loader,
        device=device,
    )["nll"]

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
    gp_nll = compute_metrics(
        pred_fn=sfr_pred(model=sfr, pred_type="gp", num_samples=num_samples),
        data_loader=test_loader,
        device=device,
    )["nll"]
    return {"nn": nn_nll, "gp": gp_nll}


def calc_gp_nll(
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
    likelihood = src.likelihoods.CategoricalLh(EPS=0.0)
    gp = init_NN2GPSubset_with_gaussian_prior(
        model=network,
        delta=delta,  # TODO what should this be
        likelihood=likelihood,
        output_dim=output_dim,
        subset_size=num_inducing,
        dual_batch_size=2000,
        jitter=1e-4,
        device=device,
    )
    gp = gp.double()
    gp.eval()
    gp.fit(train_loader=train_loader)

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
    nn_nll = compute_metrics(
        pred_fn=sfr_pred(model=gp, pred_type="nn", num_samples=num_samples),
        data_loader=test_loader,
        device=device,
    )["nll"]
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
    gp_nll = compute_metrics(
        pred_fn=sfr_pred(model=gp, pred_type="gp", num_samples=num_samples),
        data_loader=test_loader,
        device=device,
    )["nll"]
    return {"nn": nn_nll, "gp": gp_nll}


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
    la.fit(train_loader)

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
    return {"glm": glm_metrics["nll"], "bnn": bnn_metrics["nll"]}


if __name__ == "__main__":
    make_uci_table()  # pyright: ignore
