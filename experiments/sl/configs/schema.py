#!/usr/bin/env python3
from dataclasses import dataclass
from enum import Enum
from typing import List

import src
from experiments.sl.utils import init_SFR_with_gaussian_prior
from hydra.core.config_store import ConfigStore


@dataclass
class WandbConfig:
    run_name: str
    group: str
    run_name: str
    tags: List[str]
    project: str
    use_wandb: bool


class Device(Enum):  # TODO I guess this could allow more?
    cpu = "cpu"
    gpu = "gpu"


class ModelName(Enum):
    SmallMLP = "SmallMLP"
    MLP = "MLP"
    CNN = "CNN"
    AllCNN = "AllCNN"


class DatasetName(Enum):
    MNIST = "MNIST"
    FMNIST = "FMNIST"
    CIFAR10 = "CIFAR10"


@dataclass
class LikelihoodConfig:
    _target_ = src.likelihoods.CategoricalLh
    EPS: float = 0.0  # TODO default is 0.01


@dataclass
class SFRConfig:
    delta: float
    likelihood: LikelihoodConfig
    num_inducing: int
    dual_batch_size: int
    jitter: float
    device: str
    _target_ = init_SFR_with_gaussian_prior
    _convert_ = all


@dataclass
class TrainConfig:
    dataset: DatasetName
    model_name: ModelName
    batch_size: int
    lr: float
    n_epochs: int
    double: bool
    logging_epoch_freq: int
    random_seed: int
    sfr: SFRConfig
    wandb: WandbConfig
    device: Device
    debug: bool = False


# @dataclass
# class InferenceConfig:
#     dataset: DatasetName
#     model_name: ModelName
#     batch_size: int
#     lr: float
#     n_epochs: int
#     double: bool
#     logging_epoch_freq: int
#     random_seed: int
#     sfr: SFRConfig
#     wandb: WandbConfig
#     device: Device
#     debug: bool = False


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
