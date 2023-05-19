#!/usr/bin/env python3
import logging
import random
from functools import partial


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import laplace
import omegaconf
import src
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="./configs", config_name="main")
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
    # torch.backends.cudnn.determinstic = True
    # eval('setattr(torch.backends.cudnn, "benchmark", True)')
    # torch.backends.cudnn.benchmark = False
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    # torch.backends.cudnn.benchmark = False
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # cfg.device = "cpu"
    print("Using device: {}".format(cfg.device))

    ds_train, ds_test = src.sl.train.get_dataset(
        dataset=cfg.dataset, double=cfg.double, dir="./", device=cfg.device
    )
    n_classes = 10
    # TODO this is bad
    print("n_classes {}".format(n_classes))
    cfg.output_dim = n_classes

    network = get_model(model_name=cfg.model_name, ds_train=ds_train)
    network = network.to(cfg.device)
    # print("network {}".format(network))
    prior = hydra.utils.instantiate(cfg.prior, params=network.parameters)
    # print("prior {}".format(prior))
    sfr = hydra.utils.instantiate(cfg.sfr, prior=prior, network=network)
    # print("sfr {}".format(sfr))

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
    logger.info("cfg {}".format(cfg))

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=cfg.lr)

    for epoch in tqdm(list(range(cfg.n_epochs))):
        for X, y in train_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            # loss = torch.nn.CrossEntropyLoss(reduction="mean")(f, y)
            loss = sfr.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
        if epoch % cfg.logging_epoch_freq == 0:
            # tr_loss_sum, tr_loss_mean, tr_acc = evaluate(network, train_loader, cfg.device)
            # te_loss_sum, te_loss_mean, te_acc = evaluate(network, test_loader, cfg.device)
            criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            tr_loss, tr_acc, tr_nll = evaluate(
                network, train_loader, criterion, cfg.device
            )
            te_loss, te_acc, te_nll = evaluate(
                network, test_loader, criterion, cfg.device
            )
            wandb.log({"training/loss": tr_loss})
            wandb.log({"test/loss": te_loss})
            wandb.log({"training/nll": tr_nll})
            wandb.log({"test/nll": te_nll})
            # wandb.log({"training/loss_mean": tr_loss_mean})
            # wandb.log({"test/loss_mean": te_loss_mean})
            wandb.log({"training/acc": tr_acc})
            wandb.log({"test/acc": te_acc})
            wandb.log({"epoch": epoch})

    logger.info("Finished training")

    la = laplace.Laplace(
        network,
        "classification",
        subset_of_weights=cfg.subset_of_weights,
        hessian_structure=cfg.hessian_structure,
        prior_precision=prior.delta,
        backend=laplace.curvature.BackPackGGN,
    )
    if "cuda" in cfg.device:
        la.cuda()

    train_loader = DataLoader(ds_train, batch_size=len(ds_train))
    print("made train_loader {}".format(train_loader))

    logger.info("Fitting laplace...")
    la.fit(train_loader)
    logger.info("Finished fitting laplace")

    glm_pred_fn = partial(
        la.predictive_samples,
        pred_type="glm",
        n_samples=cfg.num_samples,
        diagonal_output=False,
        generator=cfg.random_seed,
    )

    bnn_pred_fn = partial(
        la.predictive_samples,
        pred_type="nn",
        n_samples=cfg.num_samples,
        diagonal_output=False,
        generator=cfg.random_seed,
    )

    logger.info("Finished making laplace pred")


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
