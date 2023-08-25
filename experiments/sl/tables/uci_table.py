#!/usr/bin/env python3
import math
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import seaborn as sns
import tikzplotlib
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


WANDB_RUNS = OmegaConf.create(
    {
        "autstralian": [
            "aalto-ml/sl-train-and-inference/mjul83ni",  # 101
            "aalto-ml/sl-train-and-inference/qk0p7938",  # 29
            "aalto-ml/sl-train-and-inference/zjslbv6y",  # 93
            "aalto-ml/sl-train-and-inference/zyqpoey4",  # 17
            "aalto-ml/sl-train-and-inference/1kx8m1wy",  # 69
            # "aalto-ml/sl-train-and-inference/mpwwugws",  # 117
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
        # "autstralian": [
        #     "aalto-ml/sl-train-and-inference/q34xn7a5",  # 101
        #     "aalto-ml/sl-train-and-inference/xmmglj38",  # 29
        #     "aalto-ml/sl-train-and-inference/uk6em4ao",  # 93
        #     "aalto-ml/sl-train-and-inference/hzyu8yv1",  # 17
        #     "aalto-ml/sl-train-and-inference/3qs207zt",  # 69
        # ],
        # "breast_cancer": [
        #     "aalto-ml/sl-train-and-inference/nhkps8ay",  # 101
        #     "aalto-ml/sl-train-and-inference/y8hoxxhr",  # 29
        #     "aalto-ml/sl-train-and-inference/cvmfmk14",  # 93
        #     "aalto-ml/sl-train-and-inference/5i2xzznq",  # 17
        #     "aalto-ml/sl-train-and-inference/cbqijws8",  # 69
        # ],
        # "Ionosphere": [
        #     "aalto-ml/sl-train-and-inference/mjyh3t6s",  # 101
        #     "aalto-ml/sl-train-and-inference/kfr8qu0z",  # 29
        #     "aalto-ml/sl-train-and-inference/05wpbooq",  # 93
        #     "aalto-ml/sl-train-and-inference/qmtac2ha",  # 17
        #     "aalto-ml/sl-train-and-inference/uea3zd15",  # 69
        # ],
        # "Glass": [
        #     "aalto-ml/sl-train-and-inference/rfo82vct",  # 101
        #     "aalto-ml/sl-train-and-inference/47jrzp7q",  # 29
        #     "aalto-ml/sl-train-and-inference/47jrzp7q",  # 93
        #     "aalto-ml/sl-train-and-inference/f3r41rfk",  # 17
        #     "aalto-ml/sl-train-and-inference/7497v5ry",  # 69
        # ],
        # "Waveform": [
        #     "aalto-ml/sl-train-and-inference/7hej55v2",  # 101
        #     "aalto-ml/sl-train-and-inference/j1t8gf27",  # 29
        #     "aalto-ml/sl-train-and-inference/qqe4r5uv",  # 93
        #     "aalto-ml/sl-train-and-inference/gssgzt7c",  # 17
        #     "aalto-ml/sl-train-and-inference/hcs5zclk",  # 69
        # ],
        # "Vehicle": [
        #     "aalto-ml/sl-train-and-inference/bng68bq4",  # 101
        #     "aalto-ml/sl-train-and-inference/lnt6gx0a",  # 29
        #     "aalto-ml/sl-train-and-inference/nbf29gag",  # 93
        #     "aalto-ml/sl-train-and-inference/n1cu6o83",  # 17
        #     "aalto-ml/sl-train-and-inference/usxhfy1k",  # 69
        # ],
        # "Digits": [
        #     "aalto-ml/sl-train-and-inference/hviwi5ct",  # 101
        #     "aalto-ml/sl-train-and-inference/tlmht6dn",  # 29
        #     "aalto-ml/sl-train-and-inference/wmt26jxr",  # 93
        #     "aalto-ml/sl-train-and-inference/znt3c6vs",  # 17
        #     "aalto-ml/sl-train-and-inference/lvs7slsb",  # 69
        # ],
        # "Satellite": [
        #     "aalto-ml/sl-train-and-inference/bbj00azg",  # 101
        #     "aalto-ml/sl-train-and-inference/q3x79mt7",  # 29
        #     "aalto-ml/sl-train-and-inference/hn58vvxn",  # 93
        #     "aalto-ml/sl-train-and-inference/rg505ovc",  # 17
        #     "aalto-ml/sl-train-and-inference/2tknuq9v",  # 69
        # ],
        # "autstralian": [
        #     "aalto-ml/sl-train-and-inference/s7mzflkz",  # 101
        #     "aalto-ml/sl-train-and-inference/ypyevmrq",  # 29
        #     "aalto-ml/sl-train-and-inference/s5nmcd6r",  # 93
        #     "aalto-ml/sl-train-and-inference/no9woj5p",  # 17
        #     "aalto-ml/sl-train-and-inference/go5hch2z",  # 69
        # ],
        # "breast_cancer": [
        #     "aalto-ml/sl-train-and-inference/oto1xw5v",  # 101
        #     "aalto-ml/sl-train-and-inference/tsnh8lw3",  # 29
        #     "aalto-ml/sl-train-and-inference/4jku4ncv",  # 93
        #     "aalto-ml/sl-train-and-inference/0ip41bw2",  # 17
        #     "aalto-ml/sl-train-and-inference/7blt5mhl",  # 69
        # ],
        # "Ionosphere": [
        #     "aalto-ml/sl-train-and-inference/nlruy30m",  # 101
        #     "aalto-ml/sl-train-and-inference/p2pvlde9",  # 29
        #     "aalto-ml/sl-train-and-inference/610wzbcm",  # 93
        #     "aalto-ml/sl-train-and-inference/vd4gu8k6",  # 17
        #     "aalto-ml/sl-train-and-inference/iv5bnlrh",  # 69
        # ],
        # "Glass": [
        #     "aalto-ml/sl-train-and-inference/iva3hwkk",  # 101
        #     "aalto-ml/sl-train-and-inference/0qan6yja",  # 29
        #     "aalto-ml/sl-train-and-inference/buszrpek",  # 93
        #     "aalto-ml/sl-train-and-inference/8jo2daaa",  # 17
        #     "aalto-ml/sl-train-and-inference/5aa9a8qm",  # 69
        # ],
        # "Waveform": [
        #     "aalto-ml/sl-train-and-inference/zigah5ji",  # 101
        #     "aalto-ml/sl-train-and-inference/c7qtm1tz",  # 29
        #     "aalto-ml/sl-train-and-inference/2h2y4i0k",  # 93
        #     "aalto-ml/sl-train-and-inference/2uagbgjw",  # 17
        #     "aalto-ml/sl-train-and-inference/rzyg25hm",  # 69
        # ],
        # "Vehicle": [
        #     "aalto-ml/sl-train-and-inference/wlg8p6kw",  # 101
        #     "aalto-ml/sl-train-and-inference/y0ovxmmv",  # 29
        #     "aalto-ml/sl-train-and-inference/9d610ray",  # 93
        #     "aalto-ml/sl-train-and-inference/5xnhnpe8",  # 17
        #     "aalto-ml/sl-train-and-inference/j0krf8xy",  # 69
        # ],
        # "Digits": [
        #     "aalto-ml/sl-train-and-inference/gr8cf8hx",  # 101
        #     "aalto-ml/sl-train-and-inference/8b7y90uo",  # 29
        #     "aalto-ml/sl-train-and-inference/isd8x1ji",  # 93
        #     "aalto-ml/sl-train-and-inference/4wid057e",  # 17
        #     "aalto-ml/sl-train-and-inference/eu3xz50w",  # 69
        # ],
        # "Satellite": [
        #     "aalto-ml/sl-train-and-inference/sok1b7wz",  # 101
        #     "aalto-ml/sl-train-and-inference/dkxrc62v",  # 29
        #     "aalto-ml/sl-train-and-inference/9akmnaiq",  # 93
        #     "aalto-ml/sl-train-and-inference/4ml7ncea",  # 17
        #     "aalto-ml/sl-train-and-inference/1ejyggfc",  # 69
        # ],
    }
)
# WANDB_RUNS = OmegaConf.create(
#     {
#         "autstralian": [
#             "aalto-ml/sl-train-and-inference/sj0rz22o",  # 77
#             "aalto-ml/sl-train-and-inference/xny3x9p1",  # 68
#             "aalto-ml/sl-train-and-inference/zb67rlk0",  # 109
#             "aalto-ml/sl-train-and-inference/p9xteu8u",  # 1023
#             "aalto-ml/sl-train-and-inference/a43e1bef",  # 432
#         ],
#         "breast_cancer": [
#             "aalto-ml/sl-train-and-inference/lhsvqsnh",  # 68
#             "aalto-ml/sl-train-and-inference/a539hb98",  # 77
#             "aalto-ml/sl-train-and-inference/30pkn3uu",  # 109
#             "aalto-ml/sl-train-and-inference/l5ykig5f",  # 1023
#             "aalto-ml/sl-train-and-inference/eyh6nsfw",  # 432
#         ],
#         "Ionosphere": [
#             "aalto-ml/sl-train-and-inference/mrzxnki4",  # 68
#             "aalto-ml/sl-train-and-inference/z0je7sv5",  # 77
#             "aalto-ml/sl-train-and-inference/d6d9bj7v",  # 109
#             "aalto-ml/sl-train-and-inference/qv4o1xz5",  # 1023
#             "aalto-ml/sl-train-and-inference/ydfn399p",  # 432
#         ],
#         "Glass": [
#             "aalto-ml/sl-train-and-inference/qy3y9gm0",  # 68
#             "aalto-ml/sl-train-and-inference/owzm29iq",  # 77
#             "aalto-ml/sl-train-and-inference/9061mbza",  # 109
#             "aalto-ml/sl-train-and-inference/5ov2sh8j",  # 1023
#             "aalto-ml/sl-train-and-inference/s92m21f9",  # 432
#         ],
#         "Vehicle": [
#             "aalto-ml/sl-train-and-inference/7p5ko5b4",  # 68
#             "aalto-ml/sl-train-and-inference/uuqwi4id",  # 77
#             "aalto-ml/sl-train-and-inference/8y7r0d7f",  # 109
#             "aalto-ml/sl-train-and-inference/9zpd5yz4",  # 1023
#             "aalto-ml/sl-train-and-inference/2lcmhxtr",  # 432
#         ],
#         "Waveform": [
#             "aalto-ml/sl-train-and-inference/ekn55k4n",  # 68
#             "aalto-ml/sl-train-and-inference/t07azz74",  # 77
#             "aalto-ml/sl-train-and-inference/jodx04oe",  # 109
#             "aalto-ml/sl-train-and-inference/xmthdcio",  # 1023
#             "aalto-ml/sl-train-and-inference/1h4x7phi",  # 432
#         ],
#         "Digits": [
#             "aalto-ml/sl-train-and-inference/3ybhiwx4",  # 68
#             "aalto-ml/sl-train-and-inference/w9qaxg5z",  # 77
#             "aalto-ml/sl-train-and-inference/g55u46g6",  # 109
#             "aalto-ml/sl-train-and-inference/iwt5famz",  # 1023
#             "aalto-ml/sl-train-and-inference/w1vvn21r",  # 432
#         ],
#         "Satellite": [
#             "aalto-ml/sl-train-and-inference/8revgf5l",  # 68
#             "aalto-ml/sl-train-and-inference/o82znd3j",  # 77
#             "aalto-ml/sl-train-and-inference/6p4d6txw",  # 109
#             "aalto-ml/sl-train-and-inference/s1bpyi9r",  # 1023
#             "aalto-ml/sl-train-and-inference/4947qsiu",  # 432
#         ],
#     }
# )
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


def make_uci_table():
    df = load_table_as_dataframe()
    print("df")
    print(df)

    # Calculate mean and std (over seeds) for each combination of dataset and model
    df_with_stats_no_inducing = (
        df.groupby(["dataset", "model"])
        .agg(
            mean=("nlpd", "mean"),
            std=("nlpd", "std"),
            count=("nlpd", "count"),
        )
        .reset_index()
    )
    df_with_stats_no_inducing = df_with_stats_no_inducing[
        df_with_stats_no_inducing["model"].isin(
            ["BNN full GRID", "NN MAP", "GLM full GRID"]
        )
    ]
    print("df_with_stats_no_inducing")
    print(df_with_stats_no_inducing)

    def dataset_percent_fn(row):
        print(f"row {row}")
        print(f"row['dataset'] {row['dataset']}")
        num_data = NUM_DATAS[row["dataset"]]
        print(f"num_data {num_data}")
        num_inducing = row["num_inducing"]
        print(f"num_inducing {num_inducing}")
        num_data = num_data * 0.7
        if isinstance(num_inducing, int):
            dataset_percent = round(num_inducing / num_data * 100)
            print(f"dataset_percent {dataset_percent}")
            return dataset_percent
        else:
            return "-"

    # df["dataset_percent"] = df.apply(dataset_percent_fn, axis=1)

    df_with_stats = (
        df.groupby(["dataset", "model", "num_inducing"])
        .agg(
            mean=("nlpd", "mean"),
            std=("nlpd", "std"),
            count=("nlpd", "count"),
        )
        .reset_index()
    )
    print("df_with_stats")
    print(df_with_stats)

    uci_table = pd.concat(
        [df_with_stats_no_inducing, df_with_stats],
        join="outer",
    ).reset_index()
    print("uci_table")
    print(uci_table)

    def bold_if_significant(row):
        print(f"row: {row}")
        # return "\val{"+ {str(row['mean']:.2f)+"}{" + str(row['std']):.2f) +""
        # return f"${row['mean']:.2f} \\pm {(row['std']):.2f}$"
        mean = f"{row['mean']:.2f}"
        std = f"{(row['std']):.2f}"
        return "\\val{" + mean + "}{" + std + "}"
        # return mean

    def num_inducing_fn(row):
        if math.isnan(row):
            return "-"
        else:
            return int(row)

    # Apply the function to the DataFrame to create the final LaTeX table
    df_with_stats["mean_pm_std"] = df_with_stats.apply(bold_if_significant, axis=1)
    df_with_stats["num_inducing"] = df_with_stats["num_inducing"].apply(num_inducing_fn)
    df_with_stats["dataset_percent"] = df_with_stats.apply(dataset_percent_fn, axis=1)
    # df["dataset_percent"] = df.apply(dataset_percent_fn, axis=1)
    # print(df_with_stats)
    # exit()
    # df_with_stats["dataset_percent"] = df_with_stats.apply(dataset_percent_fn, axis=1)
    print("df_with_stats")
    print(df_with_stats)

    # df_with_stats = df_with_stats.pivot(
    #     index=["dataset", "num_inducing"], columns="model", values="mean"
    # )

    def num_inducing_to_percent_fn(idx):
        print(f"dataset {idx[0]}")
        dataset = idx[0]
        num_data = NUM_DATAS[dataset]
        print(f"num_data {num_data}")
        num_inducing = idx[1]
        print(f"num_inducing {num_inducing}")
        num_data = num_data * 0.7
        if isinstance(num_inducing, int):
            if dataset in "glass":
                num_inducing += 1
            dataset_percent = round(num_inducing / num_data * 100)
        else:
            dataset_percent = "-"
        print(f"dataset_percent {dataset_percent}")
        return (idx[0], dataset_percent)

    # Make UCI table with NN MAP/BNN/GLM etc
    # uci_table = result.pivot(
    #     index=["dataset", "num_inducing"], columns="model", values="mean"
    # )
    uci_table["mean_pm_std"] = uci_table.apply(bold_if_significant, axis=1)
    uci_table["num_inducing"] = uci_table["num_inducing"].apply(num_inducing_fn)
    uci_table["dataset_percent"] = uci_table.apply(dataset_percent_fn, axis=1)
    print("uci")
    print(uci_table)
    uci_table = uci_table.pivot(
        index=["dataset", "num_inducing"], columns="model", values="mean_pm_std"
    )
    uci_table.index.names = ["Dataset", "$M (\%)$"]
    uci_table = uci_table.reindex(columns=COLUMNS_TITLES).rename_axis(columns=None)
    uci_table.rename(columns=COLUMNS_TITLES_DICT, inplace=True)
    uci_table.index = uci_table.index.map(num_inducing_to_percent_fn)
    for num_inducing in [1, 2, 10, 15, 80, 100]:
        for dataset in DATASETS.keys():
            try:
                uci_table = uci_table.drop((dataset, num_inducing))
            except:
                pass
    uci_table.rename(index=DATASETS, inplace=True)
    uci_table.fillna("-", inplace=True)
    print("uci_table")
    print(uci_table)
    # Print the LaTeX table
    print(uci_table.to_latex(column_format="lcccccccc", escape=False))
    print(f"df_with_stats['dataset'] {df_with_stats['dataset']}")
    print(f"df_with_stats['dataset'] {type(df_with_stats['dataset'])}")

    # Make figures
    for dataset in DATASETS.keys():
        print(f"dataset: {dataset}")
        df_dataset = df_with_stats[df_with_stats["dataset"] == dataset]
        print(f"df_dataset: {df_dataset}")
        for model_name in [
            "SFR (NN)",
            "SFR (NN) BO",
            "SFR (GP)",
            "GP Subset (NN)",
            "GP Subset (GP)",
            "GP Subset (NN) BO",
            "GP Subset (NN)",
            "GP Subset (GP)",
            "GP Subset (NN) BO",
        ]:
            df_dataset = df_dataset.drop(
                df_dataset[df_dataset["model"] == model_name].index
            )

        COLORS = {"SFR (GP) BO": "steelblue", "GP Subset (GP) BO": "darkorange"}
        fig, ax = plt.subplots()
        for model_name in ["SFR (GP) BO", "GP Subset (GP) BO"]:
            df_model = df_dataset[df_dataset["model"] == model_name]
            print(f"df_model['mean'] {df_model['mean'].to_numpy()}")
            print(f"df_model['std'] {df_model['std'].to_numpy()}")
            print(
                f"df_model['dataset_percent'] {df_model['dataset_percent'].to_numpy()}"
            )
            ax.plot(
                df_model["dataset_percent"].to_numpy(),
                df_model["mean"].to_numpy(),
                color=COLORS[model_name],
                label=model_name,
            )
            ax.fill_between(
                df_model["dataset_percent"].to_numpy(),
                df_model["mean"].to_numpy() - df_model["std"].to_numpy(),
                df_model["mean"].to_numpy() + df_model["std"].to_numpy(),
                alpha=0.1,
                color=COLORS[model_name],
            )
        plt.locator_params(axis="y", nbins=4)
        ax.set_title("\sc{" + DATASETS_NAMES[dataset] + "}")
        save_dir = "./"
        # ax.set_xlabel("$M$ as % of N")
        # ax.set_xlabel("$\\frac{M}{N} \\times 100$")
        # ax.set_ylabel("NLPD")
        ax.set_xlim(-5, 105)
        plt.savefig(os.path.join(save_dir, dataset + ".png"))
        tikzplotlib.save(
            os.path.join(save_dir, dataset + ".tex"),
            axis_width="\\figurewidth",
            axis_height="\\figureheight",
        )
        print(f"df_dataset new: {df_dataset}")

    # g = sns.relplot(
    #     data=df_dataset,
    #     x="dataset_percent",
    #     # x="num_inducing",
    #     y="mean",
    #     # col="dataset",
    #     hue="model",
    #     style="model",
    #     legend=False,
    #     # size="size",
    #     #     kind="line",
    # )
    # g.map_dataframe(sns.lineplot, x="num_inducing", y="mean", style="model")
    # g = sns.lineplot(
    #     data=df_dataset,
    #     x="dataset_percent",
    #     y="mean",
    #     #     # hue="smoker",
    #     #     style="model",
    #     #     # size="size",
    #     #     # kind="line",
    # )
    # # g.map_dataframe(sns.scatterplot, x="num_inducing", y="mean")
    # g.set_axis_labels("$\\frac{M}{N} \\times 100$", "NLPD")
    # # g.set_axis_labels("$M$", "NLPD")
    # # g.set(xlim=(0, 256), xticks=[8, 16, 32, 64, 128, 256])
    # g.set(xlim=(-5, 105))
    # g.savefig(dataset + ".png")
    # save_dir = "./"
    # tikzplotlib.save(
    #     os.path.join(save_dir, dataset + ".tex"),
    #     axis_width="\\figurewidth",
    #     axis_height="\\figureheight",
    # )
    # g = sns.relplot(
    #     data=df_with_stats,
    #     x="num_inducing",
    #     y="mean",
    #     col="dataset",
    #     # hue="smoker",
    #     style="model",
    #     # size="size",
    #     # kind="line",
    # )
    # g.map_dataframe(sns.scatterplot, x="num_inducing", y="mean")
    # g.set_axis_labels("$M$", "NLPD")
    # g.set(xlim=(0, 256), xticks=[8, 16, 32, 64, 128, 256])
    # g.savefig("uci_num_inducing.png")
    # g.set_titles(col_template="{col_name}")
    # g.show()

    # print(f"index: {df_with_stats.index}")
    # p_values = {}
    # for idx in df_with_stats.index:
    #     print(f"idx: {idx}")
    #     samples = df_with_stats.loc[idx].values
    #     print(f"samples: {samples}")
    #     print(f"samples.mean(): {samples.mean()}")
    #     # p_value = sp.stats.ttest_rel(samples, samples).pvalue
    #     p_value = sp.stats.ttest_rel(samples, [samples.mean()] * len(samples)).pvalue
    #     print(f"p_value: {p_value}")
    #     p_values[idx] = p_value
    #

    inducing_comparison_table = df_with_stats.pivot(
        index=["dataset", "num_inducing"], columns="model", values="mean_pm_std"
    )

    def num_inducing_to_percent_fn(idx):
        print(f"dataset {idx[0]}")
        dataset = idx[0]
        num_data = NUM_DATAS[dataset]
        print(f"num_data {num_data}")
        num_inducing = idx[1]
        print(f"num_inducing {num_inducing}")
        num_data = num_data * 0.7
        dataset_percent = round(num_inducing / num_data * 100)
        print(f"dataset_percent {dataset_percent}")
        return (idx[0], dataset_percent)

    inducing_comparison_table.index = inducing_comparison_table.index.map(
        num_inducing_to_percent_fn
    )
    print("inducing_comparison_table")
    print(inducing_comparison_table)

    inducing_comparison_table = inducing_comparison_table.reindex(
        columns=COLUMNS_TITLES
        # columns=COLUMNS_TITLES_NUM_INDUCING
    ).rename_axis(columns=None)
    print("latex_table")
    print(inducing_comparison_table)

    # p_values = {}
    # for dataset in WANDB_RUNS.keys():
    #     print(f"dataset: {dataset}")
    #     for model in COLUMNS_TITLES:
    #         print(f"model: {model}")
    #         for num_inducing in [16, 32, 64, 128, 256, 512]:
    #             samples = inducing_comparison_table.loc[(dataset, num_inducing)]
    #             print(f"samples: {samples}")
    #             p_value = sp.stats.ttest_rel(samples, samples.mean()).pvalue
    #             p_values[(dataset, num_inducing)] = p_value
    #             print(f"p_values: {p_values}")

    inducing_comparison_table.index.names = ["Dataset", "$M (\%)$"]
    # inducing_comparison_table.rename(columns=COLUMNS_TITLES_NUM_INDUCING, inplace=True)
    inducing_comparison_table.rename(columns=COLUMNS_TITLES_DICT, inplace=True)
    inducing_comparison_table.rename(index=DATASETS, inplace=True)
    inducing_comparison_table.fillna("-", inplace=True)

    # latex_table = latex_table.rename_axis(index=None, columns=None)

    with open("../../../paper/tables/uci_table.tex", "w") as file:
        file.write(
            inducing_comparison_table.to_latex(column_format="lcccccccc", escape=False)
        )
    with open("../../../paper/tables/uci_num_inducing_table.tex", "w") as file:
        file.write(uci_table.to_latex(column_format="lcccccccc", escape=False))

    # Print the LaTeX table
    print(inducing_comparison_table.to_latex(column_format="lccccc", escape=False))


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
