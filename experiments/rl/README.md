# Reinforcement learning experiments
Instructions for reproducing the RL results in the paper.

## Running experiments
Run a single experiment (e.g SFR) for a single random seed with:
``` sh
python rl/train.py +experiment=sfr-sample ++random_seed=100
```
You can display the base config using:
``` shell
python rl/train.py --cfg=job
```
and an experiment's config with:
``` shell
python rl/train.py +experiment=sfr-sample --cfg=job
```
Run all of the RL experiments for 5 random seeds with (you'll need a cluster for this):
``` sh
python rl/train.py --multirun +experiment=sfr-sample,laplace-sample,ensemble-sample,ddpg ++random_seed=100,69,50,666,54
```

## Reproducing figure
To reproduce the RL figure run:
``` shell
python plot_rl_figure.py
```
This downloads the results from W&B and plots the training curves.
