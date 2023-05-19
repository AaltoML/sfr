#!/usr/bin/env python3
import numpy as np

metrics = {
    "gp_subset": {
        "ece": np.array([0.05858, 0.05133, 0.04326, 0.04673, 0.05767]),
        "nll": np.array([0.5196, 0.4843, 0.4953, 0.5044, 0.51]),
        "acc": np.array([0.8384, 0.844, 0.8368, 0.8358, 0.844]) * 100,
    },
    "gp_subset_nn": {
        "ece": np.array([0.03331, 0.03279, 0.02923, 0.03736, 0.03191]),
        "nll": np.array([0.3037, 0.2822, 0.2771, 0.2922, 0.2908]),
        "acc": np.array([0.9126, 0.9138, 0.92, 0.9086, 0.9186]) * 100,
    },
    "sfr": {
        "ece": np.array([0.01251, 0.0133, 0.01147, 0.01093, 0.009851]),
        "nll": np.array([0.2582, 0.254, 0.2487, 0.2598, 0.2431]),
        "acc": np.array([0.9146, 0.9126, 0.92, 0.9114, 0.9192]) * 100,
    },
    "sfr_nn": {
        "ece": np.array([0.02822, 0.03117, 0.02461, 0.03307, 0.02518]),
        "nll": np.array([0.273, 0.2667, 0.2556, 0.2841, 0.2539]),
        "acc": np.array([0.9176, 0.917, 0.924, 0.913, 0.925]) * 100,
    },
}


for method in metrics.keys():
    # for metric in ["ece", "nll", "acc"]:
    # mean = metrics[method]["acc"].mean()
    # std = metrics[method][metric].std()
    # # print("{} {} mean={}".format(method, metric, mean))
    print("{}".format(method))
    # r"& \val{"
    acc_mean = str(round(metrics[method]["acc"].mean(), 2))
    acc_std = str(round(metrics[method]["acc"].std(), 2))
    nll_mean = str(round(metrics[method]["nll"].mean(), 3))
    nll_std = str(round(metrics[method]["nll"].std(), 3))
    ece_mean = str(round(metrics[method]["ece"].mean(), 3))
    ece_std = str(round(metrics[method]["ece"].std(), 3))
    string = (
        r"& \val{"
        + acc_mean
        + r"}{"
        + acc_std
        + r"} & \val{"
        + nll_mean
        + r"}{"
        + nll_std
        + r"} & \val{"
        + ece_mean
        + r"}{"
        + ece_std
        + r"} \\"
    )
    print(string)

# sfr_nn_ece = np.array([0.02822, 0.03117, 0.02461, 0.03307, 0.02518])
# sfr_nn_nll = np.array([0.273, 0.2667, 0.2556, 0.2841, 0.2539])
# sfr_nn_acc = np.array([0.9176, 0.917, 0.924, 0.913, 0.925])

# sfr_ece = np.array([0.01251, 0.0133, 0.01147, 0.01093, 0.009851])
# sfr_nll = np.array([0.2582, 0.254, 0.2487, 0.2598, 0.2431])
# sfr_acc = np.array([0.9146, 0.9126, 0.92, 0.9114, 0.9192])

# gp_subset_nn_ece = np.array([0.03331, 0.03279, 0.02923, 0.03736, 0.03191])
# gp_subset_nn_nll = np.array([0.3037, 0.2822, 0.2771, 0.2922, 0.2908])
# gp_subset_nn_acc = np.array([0.9126, 0.9138, 0.92, 0.9086, 0.9186])

# gp_subset_ece = np.array([0.05858, 0.05133, 0.04326, 0.04673, 0.05767])
# gp_subset_nll = np.array([0.5196, 0.4843, 0.4953, 0.5044, 0.51])
# gp_subset_acc = np.array([0.8384, 0.844, 0.8368, 0.8358, 0.844])
