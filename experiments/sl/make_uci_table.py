#!/usr/bin/env python3
import os

import hydra
import laplace
import numpy as np
import pandas as pd
import src
import torch
import wandb
from experiments.sl.cluster_train import train
from experiments.sl.inference import sfr_pred
from experiments.sl.utils import (
    compute_metrics,
    get_uci_dataset,
    init_NN2GPSubset_with_gaussian_prior,
    init_SFR_with_gaussian_prior,
)
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def make_uci_table(cfg: DictConfig):
    COLUMNS_TITLES = [
        "NN MAP",
        "BNN",
        "GLM",
        "GP Subset (GP)",
        "GP Subset (NN)",
        "SFR (GP)",
        "SFR (NN)",
    ]
    NUM_SAMPLES = 100
    num_inducing = 64
    num_inducing = 256
    posthoc_prior_opt = False
    posthoc_prior_opt = True

    # Data dictionary used to make pd.DataFrame
    data = {"dataset": [], "model": [], "experiment": [], "result": []}
    # tbl = wandb.Table(data=df)
    tbl = wandb.Table(columns=["dataset", "model", "experiment", "result"])

    def add_data(model_name, nll):
        "Add NLL to data dict and wandb table"
        dataset = dataset_name.replace("_uci", "").title()
        data["dataset"].append(dataset)
        data["model"].append(model_name)
        data["experiment"].append(experiment)
        data["result"].append(nll)
        tbl.add_data(dataset, model_name, experiment, nll)
        return data

    # Init Weight and Biases
    run = wandb.init(
        project="uci-table",
        name=cfg.wandb.run_name,
        # group=cfg.wandb.group,
        # tags=cfg.wandb.tags,
        # config=omegaconf.OmegaConf.to_container(
        #     cfg, resolve=True, throw_on_missing=True
        # ),
        dir=get_original_cwd(),  # don't nest wandb inside hydra dir
    )

    for dataset_name in [
        # BINARY
        "australian_uci",
        "breast_cancer_uci",
        "ionosphere_uci",
        # MULTI CLASS
        "glass_uci",
        "vehicle_uci",
        "waveform_uci",
        "digits_uci",
        "satellite_uci",
    ]:
        cfg = hydra.compose(
            config_name="train", overrides=[f"+experiment={dataset_name}"]
        )
        # for experiment, random_seed in enumerate([42, 100]):
        for experiment, random_seed in enumerate(
            [42, 100, 50, 1024, 55, 38, 93, 1002, 32, 521]
        ):
            torch.set_default_dtype(torch.float)
            cfg.random_seed = random_seed
            print(OmegaConf.to_yaml(cfg))
            ds_train, ds_val, ds_test = get_uci_dataset(
                dataset_name.replace("_uci", ""),
                random_seed=cfg.random_seed,
                dir=os.path.join(get_original_cwd(), "data"),
                double=cfg.double,
            )
            ds_train.data = ds_train.data.to(torch.double)
            ds_val.data = ds_val.data.to(torch.double)
            ds_test.data = ds_test.data.to(torch.double)
            ds_train.targets = ds_train.targets.long()
            ds_val.targets = ds_val.targets.long()
            ds_test.targets = ds_test.targets.long()
            # if dataset_name in [
            #     "australian_uci",
            #     "breast_cancer_uci",
            #     "ionosphere_uci",
            # ]:
            #     ds_train.targets = ds_train.targets.to(torch.double)
            #     ds_val.targets = ds_val.targets.to(torch.double)
            #     ds_test.targets = ds_test.targets.to(torch.double)

            cfg.output_dim = ds_train.output_dim
            sfr = train(cfg)  # Train the NN

            torch.set_default_dtype(torch.double)

            train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
            test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

            # sfr.network = sfr.network.double()
            sfr = sfr.double()
            sfr.eval()

            map_nll = calc_map_nll(sfr, test_loader, device=cfg.device)
            data = add_data(model_name="NN MAP", nll=map_nll)
            print(f"map_nll: {map_nll}")

            sfr_nlls = calc_sfr_nll(
                network=sfr.network,
                output_dim=ds_train.output_dim,
                delta=sfr.prior.delta,  # TODO how to set this?
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_inducing=num_inducing,
                device=cfg.device,
                posthoc_prior_opt=posthoc_prior_opt,
            )
            data = add_data(model_name="SFR (NN)", nll=sfr_nlls["nn"])
            data = add_data(model_name="SFR (GP)", nll=sfr_nlls["gp"])
            print(f"sfr_nlls: {sfr_nlls}")

            gp_nlls = calc_gp_nll(
                network=sfr.network,
                output_dim=ds_train.output_dim,
                delta=sfr.prior.delta,  # TODO how to set this?
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_inducing=num_inducing,
                device=cfg.device,
                posthoc_prior_opt=posthoc_prior_opt,
            )
            data = add_data(model_name="GP Subset (NN)", nll=gp_nlls["nn"])
            data = add_data(model_name="GP Subset (GP)", nll=gp_nlls["gp"])
            print(f"gp_nlls: {gp_nlls}")

            # print("starting laplace")
            la_nlls = calc_la_metrics(
                network=sfr.network,
                delta=sfr.prior.delta,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=cfg.device,
                posthoc_prior_opt=posthoc_prior_opt,
            )
            data = add_data(model_name="BNN", nll=la_nlls["bnn"])
            data = add_data(model_name="GLM", nll=la_nlls["glm"])
            print(f"la_nlls {la_nlls}")

            # wandb.log({"NLL": tbl})
            df = pd.DataFrame(data)
            wandb.log({"NLPD raw": wandb.Table(data=df)})
            print(df)

            # Calculate mean and 95% confidence interval for each combination of dataset and model
            table_df = (
                df.groupby(["dataset", "model"])
                .agg(
                    mean=("result", "mean"),
                    std=("result", "std"),
                    count=("result", "count"),
                )
                .reset_index()
            )

            # Calculate the 95% confidence interval for each combination
            table_df["lower_bound"] = table_df["mean"] - 1.96 * (
                table_df["std"] / np.sqrt(table_df["count"])
            )
            table_df["upper_bound"] = table_df["mean"] + 1.96 * (
                table_df["std"] / np.sqrt(table_df["count"])
            )
            print(table_df)

            # Function to determine if an element should be bolded based on the paired t-test
            def bold_if_significant(row):
                return f"${row['mean']:.4f} \\pm {(row['std']):.4f}$"
                # return f"{row['mean']:.4f} $\\pm$ {((row['upper_bound'] - row['lower_bound']) / 2):.4f}"

            # Apply the function to the DataFrame to create the final LaTeX table
            table_df["mean_conf_interval"] = table_df.apply(bold_if_significant, axis=1)
            wandb.log({"NLPD with confidence intervals": wandb.Table(data=table_df)})

            # Pivot the DataFrame to obtain the desired table format
            latex_table = table_df.pivot(
                index="dataset", columns="model", values="mean_conf_interval"
            )
            latex_table = latex_table.reindex(columns=COLUMNS_TITLES)
            wandb.log({"NLPD paper": wandb.Table(data=latex_table)})

            with open("uci_table.tex", "w") as file:
                file.write(latex_table.to_latex(escape=False))
                wandb.save("uci_table.tex")

            # Print the LaTeX table
            print(latex_table.to_latex(escape=False))


def calc_map_nll(sfr, test_loader, device):
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
):
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
            log_prior_prec_min=-10,
            log_prior_prec_max=5,
            grid_size=50,
        )
    nn_nll = compute_metrics(
        pred_fn=sfr_pred(model=sfr, pred_type="nn", num_samples=NUM_SAMPLES),
        data_loader=test_loader,
        device=device,
    )["nll"]

    # Get NLL for GP predict
    if posthoc_prior_opt:
        sfr.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="grid",
            log_prior_prec_min=-10,
            log_prior_prec_max=5,
            grid_size=50,
        )
    gp_nll = compute_metrics(
        pred_fn=sfr_pred(model=sfr, pred_type="gp", num_samples=NUM_SAMPLES),
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
):
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
            log_prior_prec_min=-10,
            log_prior_prec_max=5,
            grid_size=50,
        )
    nn_nll = compute_metrics(
        pred_fn=sfr_pred(model=gp, pred_type="nn", num_samples=NUM_SAMPLES),
        data_loader=test_loader,
        device=device,
    )["nll"]
    if posthoc_prior_opt:
        gp.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="grid",
            log_prior_prec_min=-10,
            log_prior_prec_max=5,
            grid_size=50,
        )
    gp_nll = compute_metrics(
        pred_fn=sfr_pred(model=gp, pred_type="gp", num_samples=NUM_SAMPLES),
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
):
    from experiments.sl.inference import la_pred

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
            log_prior_prec_max=5,
            grid_size=50,
        )
    bnn_pred_fn = la_pred(model=la, pred_type="nn", link_approx="mc", num_samples=100)
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
            log_prior_prec_max=5,
            grid_size=50,
        )
    glm_pred_fn = la_pred(model=la, pred_type="glm", link_approx="mc", num_samples=100)
    glm_metrics = compute_metrics(
        pred_fn=glm_pred_fn, data_loader=test_loader, device=device
    )
    return {"glm": glm_metrics["nll"], "bnn": bnn_metrics["nll"]}


if __name__ == "__main__":
    make_uci_table()  # pyright: ignore
