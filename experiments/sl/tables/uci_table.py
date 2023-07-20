#!/usr/bin/env python3
import os
import pandas as pd

import wandb
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

WANDB_RUNS = OmegaConf.create(
    {
        "Autstralian": [
            "aalto-ml/sl-train-and-inference/gdqed1ab",
        ],
        "Cancer": [
            "aalto-ml/sl-train-and-inference/7dgsiab8",
        ],
        "Ionosphere": [
            "aalto-ml/sl-train-and-inference/pvqvv0hz",
        ],
    }
)


def make_uci_table():
    df = load_table_as_dataframe()

    # Calculate mean and std (over seeds) for each combination of dataset and model
    df_with_stats = (
        df.groupby(["dataset", "model"])
        .agg(
            mean=("result", "mean"),
            std=("result", "std"),
            count=("result", "count"),
        )
        .reset_index()
    )

    def bold_if_significant(row):
        return f"${row['mean']:.4f} \\pm {(row['std']):.4f}$"

    # Apply the function to the DataFrame to create the final LaTeX table
    df_with_stats["mean_pm_std"] = df_with_stats.apply(bold_if_significant, axis=1)

    latex_table = df_with_stats.pivot(
        index="dataset", columns="model", values="mean_pm_std"
    )
    # latex_table = latex_table.reindex(columns=COLUMNS_TITLES)

    with open("uci_table.tex", "w") as file:
        file.write(latex_table.to_latex(escape=False))

    # Print the LaTeX table
    print(latex_table.to_latex(escape=False))


def load_table_as_dataframe():
    # api = wandb.Api()
    run = wandb.init()

    dfs = []
    for dataset in WANDB_RUNS.keys():
        print("Data set: {}".format(dataset))
        experiment_list = WANDB_RUNS[dataset]
        for run_id in experiment_list:
            print("Getting data for seed with run_id: {}".format(run_id))
            # run = api.run(run_id)
            table_artifact = run.use_artifact(
                run_id.split("/")[0]
                + "/"
                + run_id.split("/")[1]
                + "/run-"
                + run_id.split("/")[2]
                + "-NLPDraw:v0",
                type="run_table",
            )
            print(table_artifact)
            table_artifact.download(
                # os.path.join("saved_runs", run_id.split("/")[-1]), exist_ok=True
            )
            table = table_artifact.get("NLPD raw")
            print(table)
            df = pd.DataFrame(data=table.data, columns=table.columns)
            print(df)
            dfs.append(df)
    df_all = pd.concat(dfs)
    print(30 * "-")
    print(df_all)
    return df_all


if __name__ == "__main__":
    make_uci_table()
