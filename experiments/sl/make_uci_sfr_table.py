#!/usr/bin/env python3
import os
import seaborn as sns

import experiments
import hydra
import laplace
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import scipy as sp
import src
import torch
import wandb
from experiments.sl.cluster_train import train
from experiments.sl.inference import main as inference
from experiments.sl.inference import sfr_pred
from experiments.sl.make_uci_table import calc_gp_nll, calc_sfr_nll
from experiments.sl.utils import (
    compute_metrics,
    get_uci_dataset,
    init_NN2GPSubset_with_gaussian_prior,
    init_SFR_with_gaussian_prior,
)
from hydra import compose, initialize
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


COLUMNS_TITLES = [
    "num_inducing",
    "GP Subset (GP)",
    "GP Subset (NN)",
    "SFR (GP)",
    "SFR (NN)",
]


NUM_SAMPLES = 100
NUM_INDUCING = [16, 32, 64, 128, 256, 512, 1024]
# NUM_INDUCING = [16]

posthoc_prior_opt = False
# posthoc_prior_opt = True


# global initialization
# initialize(version_base=None, config_path="./configs", job_name="make_uci_table")
@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def make_uci_table(cfg: DictConfig):
    # Data dictionary used to make pd.DataFrame
    data = {
        "dataset": [],
        "model": [],
        "experiment": [],
        "num_inducing": [],
        "result": [],
    }
    tbl = wandb.Table(
        columns=["dataset", "model", "experiment", "num_inducing", "result"]
    )

    def add_data(model_name, nll, num_inducing):
        "Add NLL to data dict and wandb table"
        dataset = dataset_name.replace("_uci", "").title()
        data["dataset"].append(dataset)
        data["model"].append(model_name)
        data["experiment"].append(experiment)
        data["num_inducing"].append(num_inducing)
        data["result"].append(nll)
        tbl.add_data(dataset, model_name, experiment, num_inducing, nll)
        return data

    # Init Weight and Biases
    run = wandb.init(
        project="uci-table-num-inducing",
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
        cfg = compose(config_name="train", overrides=[f"+experiment={dataset_name}"])
        for experiment, random_seed in enumerate([42, 100]):
            torch.set_default_dtype(torch.float)
            # for random_seed in [42, 100, 50, 1024, 55]:
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

            cfg.output_dim = ds_train.output_dim
            sfr = train(cfg)  # Train the NN

            torch.set_default_dtype(torch.double)

            train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
            test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

            # sfr.network = sfr.network.double()
            sfr = sfr.double()
            sfr.eval()

            for num_inducing in NUM_INDUCING:
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
                data = add_data(
                    model_name="SFR (NN)", nll=sfr_nlls["nn"], num_inducing=num_inducing
                )
                data = add_data(
                    model_name="SFR (GP)", nll=sfr_nlls["gp"], num_inducing=num_inducing
                )
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
                data = add_data(
                    model_name="GP Subset (NN)",
                    nll=gp_nlls["nn"],
                    num_inducing=num_inducing,
                )
                data = add_data(
                    model_name="GP Subset (GP)",
                    nll=gp_nlls["gp"],
                    num_inducing=num_inducing,
                )
                print(f"gp_nlls: {gp_nlls}")

                # wandb.log({"NLL": tbl})
                df = pd.DataFrame(data)
                wandb.log({"NLPD raw": wandb.Table(data=df)})
                print(df)

                # Calculate mean and 95% confidence interval for each combination of dataset and model
                table_df = (
                    df.groupby(["dataset", "model", "num_inducing"])
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
                table_df["mean_conf_interval"] = table_df.apply(
                    bold_if_significant, axis=1
                )
                wandb.log(
                    {"NLPD with confidence intervals": wandb.Table(data=table_df)}
                )

                # Pivot the DataFrame to obtain the desired table format
                latex_table = table_df.pivot(
                    index="dataset",
                    columns=["model", "num_inducing"],
                    values="mean_conf_interval",
                )
                latex_table = latex_table.reindex(columns=COLUMNS_TITLES)
                wandb.log({"NLPD paper": wandb.Table(data=latex_table)})

                with open("uci_table.tex", "w") as file:
                    file.write(latex_table.to_latex(escape=False))
                    wandb.save("uci_table.tex")

                # Print the LaTeX table
                print(latex_table.to_latex(escape=False))
                # breakpoint()

                # fig, ax = plt.subplots(figsize=(11.7, 8.27))
                # fig, ax = plt.subplots()
                # sns.relplot(data=table_df, x="num_inducing", y="mean", style="model")
                # wandb.log({"M vs NLPD": fig})
            # wandb.log(
            #     {
            #         dataset_name: wandb.plot.line(
            #             wandb.Table(data=table_df),
            #             "num_inducing",
            #             "result",
            #             title="title",
            #         )
            #     }
            # )

            # fig, ax = plt.subplots()
            # for num_inducing in NUM_INDUCING:
            #     dataset = dataset_name.replace("_uci", "").title()
            #     table_df["dataset"] == dataset
            #     pass


if __name__ == "__main__":
    make_uci_table()  # pyright: ignore
