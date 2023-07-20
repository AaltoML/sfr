#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    from experiments.sl.make_uci_table import make_uci_table

    return make_uci_table(cfg)


if __name__ == "__main__":
    main()  # pyright: ignore
