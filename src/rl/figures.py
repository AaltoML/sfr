#!/usr/bin/env python3
import argparse
import math
import os

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
        "nn2svgp-sample": [  # id for multiple seeds
            "aalto-ml/nn2svgp/1981eud7",  # 100
            "aalto-ml/nn2svgp/1y1vf8xk",  # 69
            "aalto-ml/nn2svgp/6cud81zp",  # 50
            "aalto-ml/nn2svgp/mspoc1tp",  # 666
            "aalto-ml/nn2svgp/m20zxspk",  # 54
        ],
        "nn2svgp-sample-with-updates": [  # id for multiple seeds
            # "id": "aalto-ml/nn2svgp/13w22sjr",  # 42
            "aalto-ml/nn2svgp/2nfhcyfw",  # 666
            "aalto-ml/nn2svgp/19g1fzaa",  # 50
            "aalto-ml/nn2svgp/igchyb8j",  # 100
            "aalto-ml/nn2svgp/1ep2s2ne",  # 69
            "aalto-ml/nn2svgp/to5ptnuu",  # 54
        ],
        "mlp": [
            "aalto-ml/nn2svgp/1ags2von",  # 666
            "aalto-ml/nn2svgp/3mnfz5s0",  # 100
            "aalto-ml/nn2svgp/nlcicvp0",  # 69
            # "aalto-ml/nn2svgp/2jo3m0lw",  # 42
            "aalto-ml/nn2svgp/zigq11ow",  # 50
            "aalto-ml/nn2svgp/3cxumxzq",  # 54
        ],
        "ddpg-06": [
            "aalto-ml/nn2svgp/3vrkzcgo",  # 100
            "aalto-ml/nn2svgp/24trnix9",  # 666
            "aalto-ml/nn2svgp/2ej235vk",  # 50
            "aalto-ml/nn2svgp/rbq90bf5",  # 69
            "aalto-ml/nn2svgp/1ytpofrx",  # 54
        ],
    }
)

LABELS = {
    "nn2svgp-sample": "NN2SVGP",
    "nn2svgp-sample-with-updates": "NN2SVGP with updates",
    "mlp": "MLP",
    "ddpg-06": "DDPG",
}
COLORS = {
    "nn2svgp-sample": "c",
    "nn2svgp-sample-with-updates": "b",
    "mlp": "m",
    "ddpg-06": "y",
}
LINESTYLES = {
    "nn2svgp-sample": "-",
    "nn2svgp-sample-with-updates": "-",
    "mlp": "-",
    "ddpg-06": "-",
}
# LINESTYLES = {
#     "nn2svgp-sample": "-",
#     "nn2svgp-sample-with-updates": ":",
#     "mlp": "--",
#     "ddpg-06": "-.",
# }


def plot_figures(
    save_dir: str = "../../paper/fig", filename: str = "rl", random_seed: int = 42
):
    # Fix random seed for reproducibility
    np.random.seed(random_seed)

    plot_training_curves(save_dir=save_dir, filename=filename)

    # run = api.run(wandb_runs.moderl.id)
    # cfg = OmegaConf.create(run.config)
    # env = hydra.utils.instantiate(cfg.env)
    # target_state = hydra.utils.instantiate(cfg.target_state)

    # x = np.linspace(0, 1, 10)
    # y = np.sin(x)

    # plt.plot(x, y)

    # tikzplotlib.save(
    #     os.path.join(save_dir, "example_fig.tex"),
    #     axis_width="\\figurewidth",
    #     axis_height="\\figureheight",
    # )
    # plt.savefig(os.path.join(save_dir, "example_fig.pdf"), transparent=True)


def plot_training_curves(
    save_dir: str = "../../paper/fig", filename: str = "rl", window_width: int = 8
):
    api = wandb.Api()

    fig, ax = plt.subplots()

    for experiment_key in WANDB_RUNS.keys():
        print("experiment_key {}".format(experiment_key))
        experiment = WANDB_RUNS[experiment_key]
        print("experiment {}".format(experiment))

        returns_all = []
        min_length = np.inf
        for seed_id in experiment:
            print("Getting return for seed_id {}".format(seed_id))
            npz_save_name = "./wandb_data/" + seed_id.split("/")[-1] + ".npz"
            try:  # try to load from numpy if already downloaded and saved
                returns = np.load(npz_save_name)["returns"]
                print("loaded run from .npz")
            except:
                print("Couldn't load run from .npz so downloading from wandb")
                run = api.run(seed_id)
                history = run.scan_history(keys=["train/.episode_return"])
                returns = []
                for row in history:
                    if not math.isnan(row["train/.episode_return"]):
                        returns.append(row["train/.episode_return"])
                returns = np.stack(returns, 0)
                print("returns {}".format(returns.shape))
                np.savez(npz_save_name, returns=returns)
            if returns.shape[0] < min_length:
                min_length = returns.shape[0]
            returns_all.append(returns)

        # make all seeds use same number of episodes
        returns_all_copy = []
        for r in returns_all:
            returns_all_copy.append(r[0:min_length])

        values_same_length = []
        for val in returns_all_copy:
            cumsum_vec = np.cumsum(np.insert(val, 0, 0))
            ma_vec = (
                cumsum_vec[window_width:] - cumsum_vec[:-window_width]
            ) / window_width
            values_same_length.append(ma_vec)

        returns_all = np.stack(values_same_length, 0)
        # returns_all = np.stack(returns_all_copy, 0)
        print("returns_all {}".format(returns_all.shape))
        returns_mean = np.mean(returns_all, 0)
        print("returns_mean {}".format(returns_mean.shape))
        # returns_std = np.std(returns_all, 0)
        returns_std = sem(returns_all, 0)
        print("returns_std {}".format(returns_std.shape))
        num_episodes = len(returns_all[0])
        episodes = np.arange(0, num_episodes)
        print("episodes {}".format(episodes.shape))
        ax.plot(
            episodes,
            returns_mean,
            label=LABELS[experiment_key],
            color=COLORS[experiment_key],
            linestyle=LINESTYLES[experiment_key],
        )
        ax.fill_between(
            episodes,
            # returns_mean - 1.96 * returns_std,
            # returns_mean + 1.96 * returns_std,
            returns_mean - returns_std,
            returns_mean + returns_std,
            alpha=0.1,
            color=COLORS[experiment_key],
            # linestyle="",
        )

    # TODO set min episodes automatically
    ax.set_xlim(0, 70)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.xaxis.set_ticks_position("none")
    # ax.yaxis.set_ticks_position("none")
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    plt.legend()

    plt.savefig(os.path.join(save_dir, filename + ".pdf"), transparent=True)
    tikzplotlib.save(
        os.path.join(save_dir, filename + ".tex"),
        axis_width="\\figurewidth",
        axis_height="\\figureheight",
    )
    # plt.savefig(os.path.join(save_dir, "example_fig.pdf"), transparent=True)


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
