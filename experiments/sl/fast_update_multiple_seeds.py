#!/usr/bin/env python3
import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="./configs", config_name="fast_updates")
def main(cfg: DictConfig):
    from experiments.sl.fast_update_double import main
    from hydra.utils import get_original_cwd

    cfg.output_dim = 12  # hack to work with boston

    use_wandb = False
    if cfg.wandb.use_wandb:
        use_wandb = True
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )

    SEEDS = [42, 100, 48, 412, 46392]
    # SEEDS = [42, 100]
    cfg.wandb.use_wandb = False

    dfs = []
    for seed in SEEDS:
        cfg.random_seed = seed
        table_logger = main(cfg)
        print(f"table_logger {table_logger}")
        df = pd.DataFrame(table_logger.data)
        print(f"df {df}")
        dfs.append(df)

    df_all = pd.concat(dfs)
    print(f"df_all {df_all}")

    # breakpoint()
    df_with_nlpd_stats = (
        df_all.groupby(["method", "model"])
        .agg(mean=("nlpd", "mean"), std=("nlpd", "std"), count=("nlpd", "count"))
        .reset_index()
    )
    print(f"df_with_nlpd_stats {df_with_nlpd_stats}")
    df_with_mse_stats = (
        df_all.groupby(["method", "model"])
        .agg(mean=("mse", "mean"), std=("mse", "std"), count=("mse", "count"))
        .reset_index()
    )
    print(f"df_with_mse_stats {df_with_mse_stats}")

    if use_wandb:
        wandb.log({"NLPD": wandb.Table(data=df_with_nlpd_stats)})
        wandb.log({"MSE": wandb.Table(data=df_with_mse_stats)})
        wandb.log({"Metrics": wandb.Table(data=df_all)})


if __name__ == "__main__":
    main()  # pyright: ignore
