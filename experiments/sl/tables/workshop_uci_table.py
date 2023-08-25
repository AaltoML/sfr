#!/usr/bin/env python3
import math
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import tikzplotlib
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy.stats import ttest_rel, ttest_ind


WANDB_RUNS = OmegaConf.create(
    {
        "autstralian": [
            "aalto-ml/sl-train-and-inference/mjul83ni",  # 101
            "aalto-ml/sl-train-and-inference/qk0p7938",  # 29
            "aalto-ml/sl-train-and-inference/zjslbv6y",  # 93
            "aalto-ml/sl-train-and-inference/zyqpoey4",  # 17
            "aalto-ml/sl-train-and-inference/1kx8m1wy",  # 69
        ],
        "breast_cancer": [
            "aalto-ml/sl-train-and-inference/5s59fx78",  # 101
            "aalto-ml/sl-train-and-inference/2ezyvbdg",  # 29
            "aalto-ml/sl-train-and-inference/sl2g35we",  # 93
            "aalto-ml/sl-train-and-inference/xo9xf9mt",  # 17
            "aalto-ml/sl-train-and-inference/hv6w7qaq",  # 69
        ],
        "Ionosphere": [
            "aalto-ml/sl-train-and-inference/xdbtokyk",  # 101
            "aalto-ml/sl-train-and-inference/kow7xgzo",  # 29
            "aalto-ml/sl-train-and-inference/5s6z2mb5",  # 93
            "aalto-ml/sl-train-and-inference/8ro0hyrj",  # 17
            "aalto-ml/sl-train-and-inference/p4l0h40s",  # 69
            # "aalto-ml/sl-train-and-inference/",  # 117
        ],
        "Glass": [
            "aalto-ml/sl-train-and-inference/vamcf07x",  # 101
            "aalto-ml/sl-train-and-inference/vqs9j71r",  # 29
            "aalto-ml/sl-train-and-inference/7k6yukv7",  # 93
            "aalto-ml/sl-train-and-inference/ff8873aa",  # 17
            "aalto-ml/sl-train-and-inference/uccfhpqu",  # 69
        ],
        "Waveform": [
            "aalto-ml/sl-train-and-inference/ut6nqqbp",  # 101
            "aalto-ml/sl-train-and-inference/yle8p37y",  # 29
            "aalto-ml/sl-train-and-inference/y3jigiwf",  # 93
            "aalto-ml/sl-train-and-inference/e7x31pmq",  # 17
            "aalto-ml/sl-train-and-inference/jmrixful",  # 69
            # "aalto-ml/sl-train-and-inference/7hpxvcxa",  # 117
        ],
        "Vehicle": [
            "aalto-ml/sl-train-and-inference/18bsvted",  # 101
            "aalto-ml/sl-train-and-inference/1q25e9x3",  # 29
            # "aalto-ml/sl-train-and-inference/4kcpej6a",  # 93
            "aalto-ml/sl-train-and-inference/9c45uds7",  # 17
            "aalto-ml/sl-train-and-inference/c182pboq",  # 69
        ],
        "Digits": [
            "aalto-ml/sl-train-and-inference/pq90yqod",  # 101
            "aalto-ml/sl-train-and-inference/qu54ekcu",  # 29
            "aalto-ml/sl-train-and-inference/m4ccu4t5",  # 93
            "aalto-ml/sl-train-and-inference/0rvjacmi",  # 17
            "aalto-ml/sl-train-and-inference/v05fht2h",  # 69
        ],
        "Satellite": [
            "aalto-ml/sl-train-and-inference/apemft1w",  # 101
            "aalto-ml/sl-train-and-inference/1rqe6hcz",  # 29
            "aalto-ml/sl-train-and-inference/1kly90nv",  # 93
            "aalto-ml/sl-train-and-inference/7pnyjv23",  # 17
            "aalto-ml/sl-train-and-inference/yle8p37y",  # 69
            # "aalto-ml/sl-train-and-inference/",  # 117
        ],
    }
)
COLUMNS_TITLES_NUM_INDUCING = [
    "SFR (GP)",
    "SFR (NN)",
    # "GP Subest (GP) BO",
    # "GP Subest (NN) BO",
    "GP Subset (GP) BO",
    "GP Subset (NN) BO",
]
COLUMNS_TITLES = [
    "NN MAP",
    "BNN full GRID",
    "GLM full GRID",
    # "GP Subest (GP) BO",
    "GP Subset (GP) BO",
    # "GP Subest (NN)",
    "SFR (GP) BO",
    # "SFR (NN)",
]
COLUMNS_TITLES_DICT = {
    "NN MAP": "\sc nn map",
    "BNN full GRID": "\sc bnn",
    "GLM full GRID": "\sc glm",
    # "GP Subest (GP)": "{\sc gp} subset (\sc gp)",
    # "GP Subest (NN)": "{\sc gp} subset (\sc nn)",
    "GP Subset (GP)": "{\sc gp} subset (\sc gp)",
    "GP Subset (NN)": "{\sc gp} subset (\sc nn)",
    "SFR (GP)": "\our (\sc gp)",
    "SFR (NN)": "\our (\sc nn)",
    # "GP Subest (GP) BO": "{\sc gp} subset (\sc gp)",
    "GP Subset (GP) BO": "{\sc gp} subset (\sc gp)",
    "SFR (GP) BO": "\our (\sc gp)",
}

# DATASETS = {
#     "australian": "Australian",
#     "breast_cancer": "Cancer",
# }

DATASETS_NAMES = {
    "australian": "Australian",
    "breast_cancer": "Breast cancer",
    "ionosphere": "Ionosphere",
    "glass": "Glass",
    "vehicle": "Vehicle",
    "waveform": "Waveform",
    "digits": "Digits",
    "satellite": "Satellite",
}
DATASETS = {
    "australian": "Australian (N=690, D=14, C=2)",
    "breast_cancer": "Breast cancer (N=683, D=10, C=2)",
    "ionosphere": "Ionosphere (N=351, D=34, C=2)",
    "glass": "Glass (N=214, D=9, C=6)",
    "vehicle": "Vehicle (N=846, D=18, C=4)",
    "waveform": "Waveform (N=1000, D=21, C=3)",
    "digits": "Digits (N=1797, D=64, C=10)",
    "satellite": "Satellite (N=6435, D=35, C=6)",
}

NUM_DATAS = {
    "australian": 690,
    "breast_cancer": 683,
    "ionosphere": 351,
    "glass": 214,
    "vehicle": 846,
    "waveform": 1000,
    "digits": 1797,
    "satellite": 6435,
}


def bold_if_significant(row):
    print(f"ROW ROW: {row}")
    if row["pvalue"] > 0.05:
        mean = "\mathbf{" + f"{row['mean']:.2f}" + "}"
        std = "\mathbf{" + f"{(row['std']):.2f}" + "}"
    else:
        mean = f"{row['mean']:.2f}"
        std = f"{(row['std']):.2f}"
    return "\\val{" + mean + "}{" + std + "}"


def make_uci_table():
    df = load_table_as_dataframe()
    print("df")
    print(df)

    # Only keeps models we want in table
    df = df[
        df["model"].isin(
            [
                "BNN full GRID",
                "NN MAP",
                "GLM full GRID",
                "GP Subset (GP) BO",
                "SFR (GP) BO",
            ]
        )
    ]

    # Drop num_inducings we don't want in table
    num_inducings_to_drop = [1, 2, 5, 10, 15, 40, 60, 80, 100]
    for num_inducing in num_inducings_to_drop:
        df = df[df.num_inducing_percent != num_inducing]
    print("df after drop")
    print(df)

    # Calculate mean and std (over seeds) for each combination of dataset and model
    df_with_stats = (
        df.groupby(["dataset", "model"])
        .agg(
            mean=("nlpd", "mean"),
            std=("nlpd", "std"),
            count=("nlpd", "count"),
        )
        .reset_index()
    )
    print("df_with_stats")
    print(df_with_stats)

    # Calculate pvalues for bolding
    df_with_stats["pvalue"] = np.nan
    groups = []
    best_nlpd = np.inf
    pvalues = []
    for dataset_name in DATASETS.keys():
        df_dataset = df_with_stats[df_with_stats["dataset"] == dataset_name]
        # Find the best NLPD to use as base for paired t test
        for model_name in COLUMNS_TITLES:
            # breakpoint()
            # nlpd = df_dataset[df_dataset["model"] == model_name].iloc[0]["mean"]
            nlpd = df_dataset.loc[df_dataset.model == model_name, "mean"].values[0]
            if nlpd < best_nlpd:
                best_nlpd = nlpd
                best_model_name = model_name
        print("groups")
        print(groups)
        print("best_model_name")
        print(best_model_name)

        group1 = df[df["dataset"] == dataset_name]
        group1 = group1[group1["model"] == best_model_name]
        print("group1")
        print(group1)
        for model_name in COLUMNS_TITLES:
            group2 = df[df["dataset"] == dataset_name]
            group2 = group2[group2["model"] == model_name]
            print("group2")
            print(group2)
            if model_name in best_model_name:
                pvalue = np.inf
            else:
                pvalue = ttest_ind(group1["nlpd"], group2["nlpd"]).pvalue
            # pvalues.append(pvalue)
            # df_dataset[df_dataset["model"] == model_name]["pvalue"] = pvalue
            df_dataset.loc[df_dataset.model == model_name, "pvalue"] = pvalue
            df_with_stats[df_with_stats["dataset"] == dataset_name] = df_dataset
            # breakpoint()
            # df_with_stats[df_with_stats["dataset"] == dataset_name] = df_dataset

    # Apply the function to the DataFrame to create the final LaTeX table
    # df_with_stats["pvalue"] = pvalues
    df_with_stats["mean_pm_std"] = df_with_stats.apply(bold_if_significant, axis=1)

    uci_table = df_with_stats.pivot(
        index=["dataset"], columns="model", values="mean_pm_std"
    )
    print("uci_table")
    print(uci_table)

    uci_table.index.names = ["Dataset"]
    uci_table.rename(index=DATASETS, inplace=True)
    uci_table.fillna("-", inplace=True)
    uci_table = uci_table.reindex(columns=COLUMNS_TITLES).rename_axis(columns=None)
    uci_table.rename(columns=COLUMNS_TITLES_DICT, inplace=True)

    # Print the LaTeX table
    print(uci_table.to_latex(column_format="lcccccccc", escape=False))

    with open("../../../workshop/tables/workshop_uci_table.tex", "w") as file:
        file.write(uci_table.to_latex(column_format="lcccccccc", escape=False))


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
                + "-Metrics:latest",
                # + "-NLPD:latest",
                type="run_table",
            )
            print(table_artifact)
            table_artifact.download(
                # os.path.join("saved_runs", run_id.split("/")[-1]), exist_ok=True
            )
            # table = table_artifact.get("NLPD raw")
            # table = table_artifact.get("NLPD")
            table = table_artifact.get("Metrics")
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
