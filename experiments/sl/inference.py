#!/usr/bin/env python3
import logging
import os
import random
import shutil
from pprint import pprint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import laplace
import numpy as np
import src
import torch
import wandb
from experiments.sl.bnn_predictive.experiments.scripts.imgclassification import (
    get_dataset,
    get_model,
)
from experiments.sl.utils import compute_metrics, set_seed_everywhere, train_val_split
from hydra.utils import get_original_cwd
from laplace.utils import get_nll, validate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


def optimize_prior_precision_base(
    model,
    pred_type,
    method="marglik",
    n_steps=100,
    lr=1e-1,
    init_prior_prec=1.0,
    val_loader=None,
    loss=get_nll,
    log_prior_prec_min=-4,
    log_prior_prec_max=4,
    grid_size=100,
    link_approx="probit",
    n_samples=100,
    verbose=False,
    cv_loss_with_var=False,
):
    """Optimize the prior precision post-hoc using the `method`
    specified by the user.

    Parameters
    ----------
    pred_type : {'glm', 'nn', 'gp'}, default='glm'
        type of posterior predictive, linearized GLM predictive or neural
        network sampling predictive or Gaussian Process (GP) inference.
        The GLM predictive is consistent with the curvature approximations used here.
    method : {'marglik', 'CV'}, default='marglik'
        specifies how the prior precision should be optimized.
    n_steps : int, default=100
        the number of gradient descent steps to take.
    lr : float, default=1e-1
        the learning rate to use for gradient descent.
    init_prior_prec : float, default=1.0
        initial prior precision before the first optimization step.
    val_loader : torch.data.utils.DataLoader, default=None
        DataLoader for the validation set; each iterate is a training batch (X, y).
    loss : callable, default=get_nll
        loss function to use for CV.
    cv_loss_with_var: bool, default=False
        if true, `loss` takes three arguments `loss(output_mean, output_var, target)`,
        otherwise, `loss` takes two arguments `loss(output_mean, target)`
    log_prior_prec_min : float, default=-4
        lower bound of gridsearch interval for CV.
    log_prior_prec_max : float, default=4
        upper bound of gridsearch interval for CV.
    grid_size : int, default=100
        number of values to consider inside the gridsearch interval for CV.
    link_approx : {'mc', 'probit', 'bridge'}, default='probit'
        how to approximate the classification link function for the `'glm'`.
        For `pred_type='nn'`, only `'mc'` is possible.
    n_samples : int, default=100
        number of samples for `link_approx='mc'`.
    verbose : bool, default=False
        if true, the optimized prior precision will be printed
        (can be a large tensor if the prior has a diagonal covariance).
    """
    if method == "marglik":
        model.prior_precision = init_prior_prec
        log_prior_prec = model.prior_precision.log()
        log_prior_prec.requires_grad = True
        optimizer = torch.optim.Adam([log_prior_prec], lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            prior_prec = log_prior_prec.exp()
            neg_log_marglik = -model.log_marginal_likelihood(prior_precision=prior_prec)
            neg_log_marglik.backward()
            optimizer.step()
        model.prior_precision = log_prior_prec.detach().exp()
    elif method == "CV":
        if val_loader is None:
            raise ValueError("CV requires a validation set DataLoader")
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        model.prior_precision = _gridsearch(
            model,
            loss,
            interval,
            val_loader,
            pred_type=pred_type,
            link_approx=link_approx,
            n_samples=n_samples,
            loss_with_var=cv_loss_with_var,
        )
    else:
        raise ValueError("For now only marglik and CV is implemented.")
    if verbose:
        print(f"Optimized prior precision is {model.prior_precision}.")


def _gridsearch(
    model,
    loss,
    interval,
    val_loader,
    pred_type,
    link_approx="probit",
    n_samples=100,
    loss_with_var=False,
):
    results = list()
    prior_precs = list()
    for prior_prec in interval:
        logger.info(f"Prior prec: {prior_prec}")
        model.prior_precision = prior_prec
        try:
            out_dist, targets = validate(
                model,
                val_loader,
                pred_type=pred_type,
                link_approx=link_approx,
                n_samples=n_samples,
            )
            if model.likelihood == "regression":
                out_mean, out_var = out_dist
                if loss_with_var:
                    result = loss(out_mean, out_var, targets).item()
                else:
                    result = loss(out_mean, targets).item()
            else:
                result = loss(out_dist, targets).item()
        except RuntimeError:
            result = np.inf
        results.append(result)
        logger.info(f"result {result}\n")
        prior_precs.append(prior_prec)
    return prior_precs[np.argmin(results)]


@hydra.main(version_base="1.3", config_path="./configs", config_name="inference")
def main(cfg: DictConfig):
    ckpt_cfg = OmegaConf.load(
        os.path.join(get_original_cwd(), cfg.checkpoint, "files/config.yaml")
    )
    pprint("Loaded cfg from {}: {}".format(cfg.checkpoint, ckpt_cfg))

    cfg.dataset = ckpt_cfg.dataset.value

    try:  # Make experiment reproducible
        set_seed_everywhere(ckpt_cfg.random_seed.value)
        cfg.random_seed = ckpt_cfg.random_seed.value
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)
        cfg.random_seed = random_seed

    if cfg.double:
        torch.set_default_dtype(torch.double)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))
    cfg.output_dim = ckpt_cfg.output_dim.value

    # ds_train, ds_test = get_dataset(
    #     dataset=ckpt_cfg.dataset.value,
    #     # double=True,
    #     double=False,
    #     dir=get_original_cwd(),  # don't nest wandb inside hydra dir
    #     device=cfg.device,
    #     debug=ckpt_cfg.debug.value,
    # )

    # Load the data with train/val/test split
    ds_train, ds_val, ds_test = hydra.utils.instantiate(
        ckpt_cfg.dataset.value, dir=os.path.join(get_original_cwd(), "data")
    )
    cfg.output_dim = ds_train.output_dim

    # Create data loaders
    # ds_train.data = ds_train.data.to(torch.double)
    # ds_val.data = ds_val.data.to(torch.double)
    # ds_test.data = ds_test.data.to(torch.double)
    train_loader = DataLoader(dataset=ds_train, shuffle=True, batch_size=cfg.batch_size)
    val_loader = DataLoader(dataset=ds_val, shuffle=False, batch_size=cfg.batch_size)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    # Instantiate the neural network
    network = hydra.utils.instantiate(ckpt_cfg.network.value, ds_train=ds_train)
    print("made network")
    network = network.to(cfg.device)
    network = network.double()
    sfr = hydra.utils.instantiate(ckpt_cfg.sfr.value, model=network)
    print("made SFR")

    # Load checkpoint
    ckpt_fname = os.path.join(
        get_original_cwd(), cfg.checkpoint, "files/best_ckpt_dict.pt"
    )
    checkpoint = torch.load(ckpt_fname)
    print("loaded ckpt")
    sfr.load_state_dict(checkpoint["model"])
    print("loaded state dict")
    sfr = sfr.double()
    # print(f"new delta: {sfr.prior.delta}")
    sfr.eval()

    if cfg.wandb.use_wandb:  # Initialise WandB
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        shutil.copytree(
            os.path.abspath(".hydra"),
            os.path.join(os.path.join(get_original_cwd(), wandb.run.dir), "hydra"),
        )
        wandb.save("hydra")

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x)
        return sfr.likelihood.inv_link(f)
        # return torch.softmax(sfr.network(x.to(cfg.device)), dim=-1)
    map_metrics = compute_metrics(
        pred_fn=map_pred_fn, data_loader=test_loader, device=cfg.device
    )
    logger.info(f"map_metrics {map_metrics}")
    wandb.log({"map": map_metrics})

    # model = laplace.Laplace(
    #     likelihood="classification",
    #     subset_of_weights="all",
    #     hessian_structure="full",
    #     backend=laplace.curvature.asdl.AsdlGGN,
    # )
    print("making inference model")
    model = hydra.utils.instantiate(cfg.inference_strategy.model, model=sfr.network)
    if isinstance(model, laplace.BaseLaplace):
        # model.prior_precision = sfr.prior.delta
        # TODO change this back maybe
        model.prior_precision = cfg.delta_post_hoc
    elif isinstance(model, src.SFR):
        # model.prior.delta = sfr.prior.delta
        model.prior.delta = cfg.delta_post_hoc
        model = model.double()
    logger.info("Starting inference...")
    model.fit(train_loader)
    logger.info("Finished inference")

    print("building pred fn")
    for pred_cfg in cfg.inference_strategy.pred:
        print("pred_cfg {}".format(pred_cfg))
        pred_fn = hydra.utils.instantiate(pred_cfg, model=model)
        print("Made pred fn")
        metrics = compute_metrics(
            pred_fn=pred_fn, data_loader=test_loader, device=cfg.device,
            inference_strategy=cfg.inference_strategy.name
        )
        print(f"Computed metrics {metrics}")
        if isinstance(model, laplace.BaseLaplace):
            name = (
                cfg.inference_strategy.name
                + "."
                + pred_cfg.pred_type
                + "."
                + pred_cfg.link_approx
            )
        else:
            name = cfg.inference_strategy.name + "." + pred_cfg.pred_type
        wandb.log({name: metrics})

    if cfg.inference_strategy.optimize_prior_precision_kwargs is not None:
        # model.optimize_prior_precision(
        # optimize_prior_precision_base(
        #     model=model,

        #     **cfg.inference_strategy.optimize_prior_precision_kwargs,
        #     val_loader=val_loader
        # )

        for pred_cfg in cfg.inference_strategy.pred:
            print(pred_cfg)
            print(cfg.inference_strategy.optimize_prior_precision_kwargs)
            # model.optimize_prior_precision(
            optimize_prior_precision_base(
                model=model,
                pred_type=pred_cfg.pred_type,
                **cfg.inference_strategy.optimize_prior_precision_kwargs,
                val_loader=val_loader,
            )
            torch.cuda.empty_cache()
            print("pred_cfg {}".format(pred_cfg))
            pred_fn = hydra.utils.instantiate(pred_cfg, model=model)
            print("Made pred fn")
            metrics = compute_metrics(
                pred_fn=pred_fn, data_loader=test_loader, device=cfg.device
            )
            print("Computed metrics")
            if isinstance(model, laplace.BaseLaplace):
                name = (
                    cfg.inference_strategy.name
                    + "."
                    + pred_cfg.pred_type
                    + "."
                    + pred_cfg.link_approx
                    + ".posthoc"
                )
            else:
                name = (
                    cfg.inference_strategy.name + "." + pred_cfg.pred_type + ".posthoc"
                )
            wandb.log({name: metrics})


def sfr_pred(
    model: src.SFR,
    pred_type: str = "gp",  # "gp" or "nn"
    num_samples: int = 100,
):
    @torch.no_grad()
    def pred_fn(x, idx=None):
        return model(x, idx, pred_type=pred_type, num_samples=num_samples)[0]

    return pred_fn


def la_pred(
    model: laplace.BaseLaplace,
    pred_type: str = "glm",  # "glm" or "nn"
    link_approx: str = "probit",  # 'mc', 'probit', 'bridge', 'bridge_norm'
    num_samples: int = 100,  # num_samples for link_approx="mc"
):
    @torch.no_grad()
    def pred_fn(x):
        return model(
            x, pred_type=pred_type, link_approx=link_approx, n_samples=num_samples
        )

    return pred_fn


if __name__ == "__main__":
    main()
