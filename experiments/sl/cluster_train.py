#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def train(cfg: DictConfig):
    from experiments.sl.train import train

    return train(cfg)


if __name__ == "__main__":
    train()  # pyright: ignore
