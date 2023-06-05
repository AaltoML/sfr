#!/usr/bin/env python3
import logging
import os
import random


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import omegaconf
import torch
import wandb
from experiments.sl.bnn_predictive.experiments.scripts.imgclassification import (
    get_dataset,
    get_model,
)
from experiments.sl.new_inference import compute_metrics, evaluate
from experiments.sl.utils import set_seed_everywhere
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict_probs(
    dataloader: DataLoader, network: torch.nn.Module, device: str = "cpu"
):
    py = []
    for x, _ in dataloader:
        py.append(torch.softmax(network(x.to(device)), dim=-1))

    return torch.cat(py).cpu().numpy()


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def train(cfg: DictConfig):
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.double:
        logger.info("Using float64")
        torch.set_default_dtype(torch.double)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))

    ds_train, ds_test = get_dataset(
        dataset=cfg.dataset,
        double=cfg.double,
        dir="./",
        device=cfg.device,
        debug=cfg.debug,
    )

    n_classes = ds_train.K
    print("n_classes {}".format(n_classes))
    cfg.output_dim = n_classes

    network = get_model(model_name=cfg.model_name, ds_train=ds_train)
    network = network.to(cfg.device)
    sfr = hydra.utils.instantiate(cfg.sfr, model=network)

    if cfg.wandb.use_wandb:  # Initialise WandB
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        )

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=cfg.lr)

    @torch.no_grad()
    def map_pred_fn(x):
        return torch.softmax(sfr.network(x.to(cfg.device)), dim=-1)

    for epoch in tqdm(list(range(cfg.n_epochs))):
        for X, y in train_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
        if epoch % cfg.logging_epoch_freq == 0:
            train_metrics = compute_metrics(
                pred_fn=map_pred_fn,
                ds_test=ds_train,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            test_metrics = compute_metrics(
                pred_fn=map_pred_fn,
                ds_test=ds_test,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            wandb.log(train_metrics)
            wandb.log(test_metrics)
            wandb.log({"epoch": epoch})

    logger.info("Finished training")

    state = {"model": sfr.state_dict(), "optimizer": optimizer.state_dict()}

    logger.info("Saving model and optimiser etc...")
    fname = "ckpt_dict.pt"
    torch.save(state, os.path.join(run.dir, fname))
    logger.info("Finished saving model and optimiser etc")


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
