#!/usr/bin/env python3 import argparse
import math
import os
import argparse
from pprint import pprint

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import wandb
from omegaconf import OmegaConf
from scipy.stats import sem


# wandb_runs = OmegaConf.create({"id": "aalto-ml/nn2svgp/ursttc7k"})
WANDB_RUNS = OmegaConf.create(
    {
        "mnist": [
            # "aalto-ml/nn2svgp-sl/uh8up7xn",  #
            # "aalto-ml/nn2svgp-sl/oi1qbc4r",
            # "aalto-ml/nn2svgp-sl/lavsf2m1",
            # "aalto-ml/nn2svgp-sl/oi1qbc4r",
            "aalto-ml/nn2svgp-sl/aj28i48l",  # kron
            "aalto-ml/nn2svgp-sl/3z1u65h2",  # kron
            "aalto-ml/nn2svgp-sl/f8jmvcge",  # kron
            "aalto-ml/nn2svgp-sl/0n3vve7n",  # kron
            "aalto-ml/nn2svgp-sl/tl1jk3bo",  # kron
            "aalto-ml/nn2svgp-sl/n97vc28b",  # kron
            # "aalto-ml/nn2svgp-sl/dlz56hd2", # diag
            # "aalto-ml/nn2svgp-sl/vq76j3qx", # diag
            # "aalto-ml/nn2svgp-sl/t8cd9rss", # diag
            # "aalto-ml/nn2svgp-sl/ulykl6w3", # diag
            # "aalto-ml/nn2svgp-sl/jwhqf4sa", # diag
            # "aalto-ml/nn2svgp-sl/qk9vnpn1", # diag
        ],
        "fmnist": [
            #     "aalto-ml/nn2svgp-sl/uh8up7xn",  #
            # "aalto-ml/nn2svgp-sl/wmqt5qlz",
            # "aalto-ml/nn2svgp-sl/5dvzjiud"
            # "aalto-ml/nn2svgp-sl/75mx5r65",  # kron
            # "aalto-ml/nn2svgp-sl/9knszip2",  # kron
            # "aalto-ml/nn2svgp-sl/tnv0pwvs",  # kron
            # "aalto-ml/nn2svgp-sl/jxrx7zcv",  # kron
            # "aalto-ml/nn2svgp-sl/o7z19ozg",  # kron
            # "aalto-ml/nn2svgp-sl/rou049qc",  # kron
            # "aalto-ml/nn2svgp-sl/kbraqree",  # diag
            # "aalto-ml/nn2svgp-sl/02au4etu",  # diag
            # "aalto-ml/nn2svgp-sl/8gbrc7dq",  # diag
            # "aalto-ml/nn2svgp-sl/fjz4ffck",  # diag
            # "aalto-ml/nn2svgp-sl/4n7xav0w",  # diag
            # "aalto-ml/nn2svgp-sl/ti3fx2wx",  # diag
            "aalto-ml/nn2svgp-sl/yw94vxss",
            "aalto-ml/nn2svgp-sl/cd1lbynk",
            "aalto-ml/nn2svgp-sl/a2fi80be",
            "aalto-ml/nn2svgp-sl/36isce0q",
            "aalto-ml/nn2svgp-sl/s83iwn6v",
        ],
    }
)
DATASET = {
    "mnist": "MNIST",
    "fmnist": "FMNIST",
    "cifar10": "CIFAR10",
}

LABELS = {
    "bnn": "\sc bnn",
    "glm": "\sc glm",
    "map": "\sc map",
    "sfr_nn": "\sc \our (NN)",
    "sfr": "\sc \our",
    "gp_subset": "\sc GP subset",
    "gp_subset_nn": "\sc GP subset (NN)",
}
TABLE_LABELS = {
    "map": "\sc map ",
    "bnn": "\sc bnn predictive \cite{immer2021improving}",
    "glm": "\sc glm \cite{immer2021improving}",
    "sfr_nn": "\our NN",
    "sfr": "\our",
    "gp_subset_nn": "\sc gp predictive NN \cite{immer2021improving}",
    "gp_subset": "\sc gp predictive ",
}

NUM_INDUCINGS = ["256", "512", "1024", "2048", "3200"]


def plot_figures(
    save_dir: str = "../../paper/fig", filename: str = "", random_seed: int = 42
):
    # Fix random seed for reproducibility
    np.random.seed(random_seed)

    plot_metrics(save_dir=save_dir, filename=filename)


def plot_metrics(
    save_dir: str = "../../paper/fig", filename: str = "rl", window_width: int = 8
):
    api = wandb.Api()

    metric_dict_all = {}
    metric_dict_mean_all = {}
    metric_dict_std_all = {}
    for experiment_key in WANDB_RUNS.keys():
        print("experiment_key {}".format(experiment_key))
        experiment = WANDB_RUNS[experiment_key]
        print("experiment {}".format(experiment))

        metric_dict = {}
        for seed_id in experiment:
            print("Getting return for seed_id {}".format(seed_id))
            run = api.run(seed_id)
            # print("run.summary {}".format(run.summary))
            # metric_vals = []
            for key in ["map_acc_te", "map_nll_te", "map_ece_te"]:
                # print("run.summary {}".format(run.summary[key]))
                try:
                    metric_dict[key] = metric_dict.get(key, list()) + [run.summary[key]]
                except:
                    metric_dict[key] = metric_dict.get(key, list()) + ["-"]

            for key in ["bnn_acc_te", "bnn_nll_te", "bnn_ece_te"]:
                # print("run.summary {}".format(run.summary[key]))
                try:
                    metric_dict[key] = metric_dict.get(key, list()) + [run.summary[key]]
                except:
                    metric_dict[key] = metric_dict.get(key, list()) + ["-"]
            for key in ["glm_acc_te", "glm_nll_te", "glm_ece_te"]:
                # print("run.summary {}".format(run.summary[key]))
                # metric_dict[key] = metric_dict.get(key, list()) + [run.summary[key]]
                try:
                    metric_dict[key] = metric_dict.get(key, list()) + [run.summary[key]]
                except:
                    metric_dict[key] = metric_dict.get(key, list()) + ["-"]

            for metric in ["nll", "acc", "ece"]:
                for num_inducing in NUM_INDUCINGS:
                    for method in ["gp_subset", "sfr", "gp_subset_nn", "sfr_nn"]:
                        key = (
                            method
                            + "_sparse"
                            + str(num_inducing)
                            + "_"
                            + metric
                            + "_te"
                        )
                        # print("run.summary {}".format(run.summary[key]))
                        # metric_dict[key] = metric_dict.get(key, list()) + [
                        #     run.summary[key]
                        # ]
                        try:
                            metric_dict[key] = metric_dict.get(key, list()) + [
                                run.summary[key]
                            ]
                        except:
                            metric_dict[key] = metric_dict.get(key, list()) + ["-"]

                # gp_subset_sparse256_nll_te
                # for metric in ["nll", "acc", "ece"]:
                # metric_val = run.scan_history(
                #     keys=["sfr_sparse" + str(num_inducing) + "_" + metric + "_te"]
                # )
                # print("run.summary[metric_val] {}".format(run.summary))
                # metric_vals.append(run.summary[metric_val])
        print("metric_dict")
        print(metric_dict)
        metric_dict_mean, metric_dict_std = {}, {}
        for key in metric_dict.keys():
            try:
                mean = np.array(metric_dict[key]).mean()
            except:
                mean = "-"
            try:
                std = np.array(metric_dict[key]).std()
            except:
                std = "-"

            metric_dict_mean[key] = metric_dict_mean.get(key, list()) + [mean]
            metric_dict_std[key] = metric_dict_std.get(key, list()) + [std]
        print("MEAN")
        pprint(metric_dict_mean)
        print("STD")
        pprint(metric_dict_std)
        metric_dict_mean_all.update({experiment_key: metric_dict_mean})
        metric_dict_std_all.update({experiment_key: metric_dict_std})

        # fig, ax = plt.subplots()

        # for metric in ["nll", "acc"]:
        #     for method in ["gp_subset", "sfr", "gp_subset_nn", "sfr_nn"]:
        #         vals = []
        #         for num_inducing in NUM_INDUCINGS:
        #             key = method + "_sparse" + str(num_inducing) + "_" + metric + "_te"
        #             vals.append(metric_dict_mean[key])
        #             ax.plot(NUM_INDUCINGS, vals, label=LABELS[method])

        #     for method in ["bnn", "glm", "map"]:
        #         key = method + "_" + metric + "_te"
        #         ax.plot(
        #             NUM_INDUCINGS,
        #             [metric_dict_mean[key]] * len(NUM_INDUCINGS),
        #             label=LABELS[method],
        #             linestyle="--",
        #         )
        #         ax.legend()
        #         plt.savefig("./" + DATASET[experiment_key] + "_" + metric + ".pdf")
        #         filename = DATASET[experiment_key] + "_" + metric
        #         tikzplotlib.save(
        #             os.path.join(save_dir, filename + ".tex"),
        #             axis_width="\\figurewidth",
        #             axis_height="\\figureheight",
        #         )
        metric_dict_all.update({experiment_key: metric_dict})

    METRICS = ["acc", "nll", "ece"]
    lines = []
    lines.append(r"\begin{tabular}{l l C{\tblw} C{\tblw} C{\tblw}}")
    lines.append(r"\toprule")
    lines.append(r"& Method & ACC~$\uparrow$ & NLPD~$\downarrow$ & ECE~$\downarrow$ \\")
    # lines.append(r"\midrule")
    # lines.append(r"\multirow{2}{*}{MNIST}")
    # lines.append(
    #     r"\begin{tabular}{l C{0.6\tblw} C{0.6\tblw} C{0.6\tblw} C{0.6\tblw} C{0.6\tblw}"
    # )
    # lines.append(r"\toprule")
    # lines.append(
    #     r"& Method & ACC~$\uparrow$ & NLPD~$\downarrow$ & ECE~$\downarrow$  \\"
    # )
    # lines.append(r"\midrule")
    # lines.append(r"\multirow{2}{*}{MNIST} ")

    METHODS = ["map", "bnn", "glm", "gp_subset", "gp_subset_nn", "sfr", "sfr_nn"]
    for experiment_key in WANDB_RUNS.keys():
        metric_dict_mean = metric_dict_mean_all[experiment_key]
        metric_dict_std = metric_dict_std_all[experiment_key]
        lines.append(r"\midrule")
        lines.append(r"\multirow{2}{*}{" + DATASET[experiment_key] + "}")
        for method_name in METHODS:
            line_str = f"& \sc {TABLE_LABELS[method_name]} & "
            # line_str = f"\sc {method_name} &"
            for metric in METRICS:
                if method_name in ["bnn", "glm", "map"]:
                    key = method_name + "_" + metric + "_te"
                else:
                    # num_inducing = 512
                    num_inducing = 1024
                    key = (
                        method_name
                        + "_sparse"
                        + str(num_inducing)
                        + "_"
                        + metric
                        + "_te"
                    )

                mean = metric_dict_mean[key]
                if "acc" in metric:
                    print("acc")
                    # if isinstance(mean[0], float):
                    mean = mean[0] * 100
                    # else:
                    #     mean = mean[0]
                    # print(mean)
                else:
                    mean = mean[0]
                std = metric_dict_std[key]
                # (mean, var) = method_map_img[method_name][metric][0]
                # mean_var = f"{mean} {std}"
                line_str += r" \val{"
                # print("mean {}".format(str(mean[0])))
                try:
                    line_str += "{0:.5f}".format(mean)
                except:
                    line_str += mean
                line_str += "}{"
                # line_str += str(std[0])
                try:
                    line_str += "{0:.5f}".format(std[0])
                except:
                    line_str += std[0]
                line_str += "} &"
            line_str = line_str[:-2]
            line_str += r" \\"
            lines.append(line_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table_filename = "img_super.tex"
    save_dir_tab = "../../paper/tables"
    tex_file = os.path.join(save_dir_tab, table_filename)

    if os.path.exists(tex_file):
        os.remove(tex_file)
    with open(tex_file, "w") as file:
        for line in lines:
            file.write(line + "\n")

    print(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        help="directory to save figures",
        default="../../paper/fig",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="fix random seed for reproducibility",
        default=42,
    )
    args = parser.parse_args()

    plot_figures(save_dir=args.save_dir, random_seed=args.random_seed)
