# Reinforcement learning experiments
Instructions for reproducing the RL results in the paper.

## Running experiments
Run all of the RL experiments for 5 random seeds with (you'll need a cluster for this):
``` sh
python train.py --multirun +experiment=sfr-sample,laplace-sample,ensemble-sample,ddpg ++random_seed=100,69,50,666,54
```
Alternatively, run a single experiment (e.g SFR) for a single random seed with:
``` sh
python train.py +experiment=sfr-sample ++random_seed=100
```
You can display the base config using:
``` shell
python train.py --cfg=job
```
and an experiments config with:
``` shell
python train.py +experiment=sfr-sample --cfg=job
```

## Reproducing figure
To reproduce the RL figure run:
``` shell
python plot_rl_figure.py
```
This downloads the results from W&B and plots the training curves.
