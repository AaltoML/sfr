#!/usr/bin/env python3

import logging
import os
import random
from functools import partial

import hydra
import laplace
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from preds.utils import ece, macc, nll_cls
from torch.distributions import Categorical, Normal
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from train import get_dataset, get_model, set_seed_everywhere


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_quick_loader(loader, device="cuda"):
    return [(X.to(device), y.to(device)) for X, y in loader]


def get_map_predictive(loader, model):
    ys, pstar = list(), list()
    for X, y in loader:
        X, y = X, y
        ys.append(y)
        pstar.append(torch.softmax(model(X), dim=-1).detach())
    ys = torch.cat(ys)
    pstar = torch.cat(pstar)
    return pstar, ys


def get_svgp_predictive(
    loader, svgp, likelihood, use_nn_out: bool = True, seeding: bool = False
):
    ys, ps = list(), list()
    for X, y in loader:  # tqdm(loader):
        X, y = X.cuda(), y.cuda()
        if seeding:
            torch.manual_seed(711)
        ps.append(
            sample_svgp(X, likelihood, svgp, use_nn_out, n_samples=100).mean(dim=0)
        )
        ys.append(y)
    ps = torch.cat(ps)
    ys = torch.cat(ys)
    return ps, ys


def get_la_predictive(loader, la_pred, seeding: bool = False):
    ys, ps = list(), list()
    for X, y in loader:  # tqdm(loader):
        X, y = X.cuda(), y.cuda()
        if seeding:
            torch.manual_seed(711)
        ps.append(
            la_pred(x=X).mean(dim=0)
            # sample_svgp(X, likelihood, svgp, use_nn_out, n_samples=100).mean(dim=0)
        )
        ys.append(y)
    ps = torch.cat(ps)
    ys = torch.cat(ys)
    return ps, ys


def sample_svgp(X, likelihood, svgp, use_nn_out: bool, n_samples: int):
    """Sample the SVGP, assumes a batched input."""
    n_data = X.shape[0]
    gp_means, gp_vars = svgp.predict_f(X)
    logits = svgp.network(X)
    if use_nn_out:
        dist = Normal(logits, torch.sqrt(gp_vars.clamp(10 ** (-32))))
    else:
        dist = Normal(gp_means, torch.sqrt(gp_vars.clamp(10 ** (-32))))
    logit_samples = dist.sample((n_samples,))
    out_dim = logit_samples.shape[-1]
    samples = likelihood.inv_link(logit_samples)
    # samples = samples.reshape(n_samples, n_data, out_dim)
    return samples


def evaluate(lh, yte, gstar_te, yva, gstar_va):
    res = dict()
    res["nll_te"] = nll_cls(gstar_te, yte, lh)
    res["nll_va"] = nll_cls(gstar_va, yva, lh)
    res["acc_te"] = macc(gstar_te, yte)
    res["acc_va"] = macc(gstar_va, yva)
    res["ece_te"] = ece(gstar_te, yte)
    res["ece_va"] = ece(gstar_va, yva)
    return res


@hydra.main(version_base="1.3", config_path="./configs", config_name="inference")
def main(cfg: DictConfig):
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    if cfg.double:
        torch.set_default_dtype(torch.double)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.determinstic = True
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    # torch.backends.cudnn.benchmark = False
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # cfg.device = "cpu"
    print("Using device: {}".format(cfg.device))

    ds_train, ds_test = get_dataset(
        dataset=cfg.dataset, double=True, dir="./", device=cfg.device, cfg=cfg
    )

    n_classes = ds_train.K
    cfg.output_dim = n_classes

    # Load the model and load on GPU
    network = get_model(model_name=cfg.model_name, ds_train=ds_train)
    checkpoint = torch.load(cfg.checkpoint)
    network.load_state_dict(checkpoint["model"])
    network = network.to(cfg.device)
    cfg.prior.delta = checkpoint["delta"]

    prior = hydra.utils.instantiate(cfg.prior, params=network.parameters)
    sfr = hydra.utils.instantiate(cfg.sfr, prior=prior, network=network)
    gp_subset = hydra.utils.instantiate(cfg.gp_subset, prior=prior, network=network)

    if cfg.wandb.use_wandb:  # Initialise WandB
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )
    logger.info("cfg {}".format(cfg))

    compute_metrics(sfr, gp_subset, ds_train, ds_test, cfg, checkpoint)


def compute_metrics(sfr, gp_subset, ds_train, ds_test, cfg, checkpoint):
    M = len(ds_test)
    n_inducing = cfg.sfr.num_inducing  # int(len(ds_train)*n_sparse)
    logging.info(f"Train set size: {len(ds_train)}")
    logging.info(f"Num inducing points: {n_inducing}")
    perm_ixs = torch.randperm(M)
    val_ixs, test_ixs = perm_ixs[: int(M / 2)], perm_ixs[int(M / 2) :]
    ds_val = Subset(ds_test, val_ixs)
    ds_test = Subset(ds_test, test_ixs)
    val_loader = get_quick_loader(
        DataLoader(ds_val, batch_size=cfg.batch_size), device=cfg.device
    )
    test_loader = get_quick_loader(
        DataLoader(ds_test, batch_size=cfg.batch_size), device=cfg.device
    )
    all_train = DataLoader(ds_train, batch_size=len(ds_train))
    (X_train, y_train) = next(iter(all_train))
    X_train = X_train.to(cfg.device)
    y_train = y_train.to(cfg.device)
    data = (X_train, y_train)

    if cfg.predictive_model == "map":
        # MAP
        conf_name = "map"
        logging.info("MAP performance")
        gstar_te, yte = get_map_predictive(test_loader, sfr.network)
        gstar_va, yva = get_map_predictive(val_loader, sfr.network)
        checkpoint["map"] = evaluate(sfr.likelihood, yte, gstar_te, yva, gstar_va)

        logging.info(checkpoint["map"])
        wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})
    elif cfg.predictive_model == "sfr":
        logging.info("SFR performance")

        sfr.set_data(data)

        conf_name = f"sfr_sparse{cfg.sfr.num_inducing}"

        logging.info(f"Computing {conf_name}")
        gstar_te, yte = get_svgp_predictive(
            test_loader, sfr, use_nn_out=False, likelihood=sfr.likelihood
        )
        gstar_va, yva = get_svgp_predictive(
            val_loader, sfr, use_nn_out=False, likelihood=sfr.likelihood
        )
        checkpoint[conf_name] = evaluate(sfr.likelihood, yte, gstar_te, yva, gstar_va)
        logging.info(checkpoint[conf_name])
        wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})

        conf_name = f"sfr_nn_sparse{cfg.sfr.num_inducing}"
        logging.info(f"Computing {conf_name}")
        gstar_te, yte = get_svgp_predictive(
            test_loader, sfr, use_nn_out=True, likelihood=sfr.likelihood
        )

        gstar_va, yva = get_svgp_predictive(
            val_loader, sfr, use_nn_out=True, likelihood=sfr.likelihood
        )
        checkpoint[conf_name] = evaluate(sfr.likelihood, yte, gstar_te, yva, gstar_va)
        logging.info(checkpoint[conf_name])
        wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})

    elif cfg.predictive_model == "gp_subset":
        # GP subset
        logging.info("GP subset")

        conf_name = f"gp_subset_nn_sparse{cfg.gp_subset.subset_size}"
        gp_subset.set_data(data)
        gstar_te, yte = get_svgp_predictive(
            test_loader, gp_subset, use_nn_out=True, likelihood=gp_subset.likelihood
        )
        gstar_va, yva = get_svgp_predictive(
            val_loader, gp_subset, use_nn_out=True, likelihood=gp_subset.likelihood
        )

        checkpoint[conf_name] = evaluate(
            gp_subset.likelihood, yte, gstar_te, yva, gstar_va
        )
        logging.info(checkpoint[conf_name])
        wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})

        conf_name = f"gp_subset_sparse{cfg.gp_subset.subset_size}"
        gstar_te, yte = get_svgp_predictive(
            test_loader, gp_subset, use_nn_out=False, likelihood=gp_subset.likelihood
        )
        gstar_va, yva = get_svgp_predictive(
            val_loader, gp_subset, use_nn_out=False, likelihood=gp_subset.likelihood
        )

        checkpoint[conf_name] = evaluate(
            gp_subset.likelihood, yte, gstar_te, yva, gstar_va
        )
        logging.info(checkpoint[conf_name])
        wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})
    elif cfg.predictive_model == "bnn" or cfg.predictive_model == "glm":
        la = laplace.Laplace(
            sfr.network,
            "classification",
            subset_of_weights=cfg.subset_of_weights,
            hessian_structure=cfg.hessian_structure,
            prior_precision=sfr.prior.delta,
            backend=laplace.curvature.asdl.AsdlGGN,
        )
        # la.to(cfg.device)

        train_loader = DataLoader(ds_train, batch_size=len(ds_train))
        print("made train_loader {}".format(train_loader))

        logger.info("Fitting laplace...")
        la.fit(train_loader)
        logger.info("Finished fitting laplace")

        # GLM predictive
        conf_name = "glm"
        logging.info("GLM")
        la_pred = partial(
            la.predictive_samples,
            pred_type="glm",
            n_samples=100,
            diagonal_output=False,
            generator=cfg.random_seed,
        )
        gstar_te, yte = get_la_predictive(test_loader, la_pred, seeding=True)
        gstar_va, yva = get_la_predictive(val_loader, la_pred, seeding=True)
        checkpoint[conf_name] = evaluate(
            gp_subset.likelihood, yte, gstar_te, yva, gstar_va
        )
        logging.info(checkpoint[conf_name])
        wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})

        # BNN predictive
        conf_name = "bnn"
        logging.info("BNN predictive")
        la_pred = partial(
            la.predictive_samples,
            pred_type="nn",
            n_samples=100,
            diagonal_output=False,
            generator=cfg.random_seed,
        )
        gstar_te, yte = get_la_predictive(test_loader, la_pred, seeding=True)
        gstar_va, yva = get_la_predictive(val_loader, la_pred, seeding=True)
        checkpoint[conf_name] = evaluate(
            gp_subset.likelihood, yte, gstar_te, yva, gstar_va
        )
        logging.info(checkpoint[conf_name])
        wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})

    res_dir = "./saved_inference_results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    fname = (
        "./"
        + "_".join([cfg.dataset, cfg.model_name, str(cfg.random_seed)])
        + f"_{cfg.prior.delta:.1e}.pt"
    )
    torch.save(checkpoint, os.path.join(res_dir, fname))


if __name__ == "__main__":
    main()
