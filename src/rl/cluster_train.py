#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="./configs", config_name="main")
def train(cfg: DictConfig):
    import train

    train.train(cfg)


if __name__ == "__main__":
    train()  # pyright: ignore
