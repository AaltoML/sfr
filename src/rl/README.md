# Reinforcement learning experiments
Instructions for reproducing the RL results in the paper.

# Running experiments
Run all of the RL experiments with (you'll need a cluster for this):
``` sh
python rl/train.py --multirun +experiment=sfr-sample,laplace-sample,ensemble-sample,ddpg
```
Alternatively, run a single experiment (e.g SFR) with:
``` sh
python rl/train.py +experiment=sfr-sample
```
You can display the base config using:
``` shell
python train.py --cfg=job
```
and an experiments config with:
``` shell
python train.py +experiment=sfr-sample --cfg=job
```

# Reproducing figure

To reproduce the RL figure run:
``` shell
python plot_rl_figure.py
```
This downloads the results from W&B and plots the training curves.
