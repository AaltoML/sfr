#!/usr/bin/env python3
import logging
import os
import random
import shutil


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import logging
import warnings
from copy import deepcopy

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from experiments.sl.bnn_predictive.experiments.scripts.imgclassification import (
    get_dataset,
    get_model,
)
from experiments.sl.configs.schema import TrainConfig
from experiments.sl.utils import compute_metrics, set_seed_everywhere, train_val_split
from hydra.utils import get_original_cwd
from laplace import Laplace
from laplace.curvature import AsdlGGN
from laplace.utils import expand_prior_precision
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


def marglik_training(
    model,
    train_loader,
    likelihood="classification",
    hessian_structure="kron",
    backend=AsdlGGN,
    optimizer_cls=Adam,
    optimizer_kwargs=None,
    scheduler_cls=None,
    scheduler_kwargs=None,
    n_epochs=300,
    lr_hyp=1e-1,
    prior_structure="layerwise",
    n_epochs_burnin=0,
    n_hypersteps=10,
    marglik_frequency=1,
    prior_prec_init=1.0,
    sigma_noise_init=1.0,
    temperature=1.0,
    enable_backprop=False,
):
    """Marginal-likelihood based training (Algorithm 1 in [1]).
    Optimize model parameters and hyperparameters jointly.
    Model parameters are optimized to minimize negative log joint (train loss)
    while hyperparameters minimize negative log marginal likelihood.
    This method replaces standard neural network training and adds hyperparameter
    optimization to the procedure.

    The settings of standard training can be controlled by passing `train_loader`,
    `optimizer_cls`, `optimizer_kwargs`, `scheduler_cls`, `scheduler_kwargs`, and `n_epochs`.
    The `model` should return logits, i.e., no softmax should be applied.
    With `likelihood='classification'` or `'regression'`, one can choose between
    categorical likelihood (CrossEntropyLoss) and Gaussian likelihood (MSELoss).
    As in [1], we optimize prior precision and, for regression, observation noise
    using the marginal likelihood. The prior precision structure can be chosen
    as `'scalar'`, `'layerwise'`, or `'diagonal'`. `'layerwise'` is a good default
    and available to all Laplace approximations. `lr_hyp` is the step size of the
    Adam hyperparameter optimizer, `n_hypersteps` controls the number of steps
    for each estimated marginal likelihood, `n_epochs_burnin` controls how many
    epochs to skip marginal likelihood estimation, `marglik_frequency` controls
    how often to estimate the marginal likelihood (default of 1 re-estimates
    after every epoch, 5 would estimate every 5-th epoch).
    References
    ----------
    [1] Immer, A., Bauer, M., Fortuin, V., Rätsch, G., Khan, EM.
    [*Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning*](https://arxiv.org/abs/2104.04975).
    ICML 2021.
    Parameters
    ----------
    model : torch.nn.Module
        torch neural network model (needs to comply with Backend choice)
    train_loader : DataLoader
        pytorch dataloader that implements `len(train_loader.dataset)` to obtain number of data points
    likelihood : str, default='classification'
        'classification' or 'regression'
    hessian_structure : {'diag', 'kron', 'full'}, default='kron'
        structure of the Hessian approximation
    backend : Backend, default=AsdlGGN
        Curvature subclass, e.g. AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    optimizer_cls : torch.optim.Optimizer, default=Adam
        optimizer to use for optimizing the neural network parameters togeth with `train_loader`
    optimizer_kwargs : dict, default=None
        keyword arguments for `optimizer_cls`, for example to change learning rate or momentum
    scheduler_cls : torch.optim.lr_scheduler._LRScheduler, default=None
        optionally, a scheduler to use on the learning rate of the optimizer.
        `scheduler.step()` is called after every batch of the standard training.
    scheduler_kwargs : dict, default=None
        keyword arguments for `scheduler_cls`, e.g. `lr_min` for CosineAnnealingLR
    n_epochs : int, default=300
        number of epochs to train for
    lr_hyp : float, default=0.1
        Adam learning rate for hyperparameters
    prior_structure : str, default='layerwise'
        structure of the prior. one of `['scalar', 'layerwise', 'diagonal']`
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int, default=10
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
        `marglik_frequency=1` would be every epoch,
        `marglik_frequency=5` would be every 5 epochs.
    prior_prec_init : float, default=1.0
        initial prior precision
    sigma_noise_init : float, default=1.0
        initial observation noise (for regression only)
    temperature : float, default=1.0
        factor for the likelihood for 'overcounting' data. Might be required for data augmentation.
    enable_backprop : bool, default=False
        make the returned Laplace instance backpropable---useful for e.g. Bayesian optimization.
    Returns
    -------
    lap : laplace
        fit Laplace approximation with the best obtained marginal likelihood during training
    model : torch.nn.Module
        corresponding model with the MAP parameters
    margliks : list
        list of marginal likelihoods obtained during training (to monitor convergence)
    losses : list
        list of losses (log joints) obtained during training (to monitor convergence)
    """

    # get device, data set size N, number of layers H, number of parameters P
    device = parameters_to_vector(model.parameters()).device
    num_data = len(train_loader.dataset)
    # H = len(list(model.parameters()))
    num_params = len(parameters_to_vector(model.parameters()))

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec_init = np.log(temperature * prior_prec_init)
    log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    log_prior_prec.requires_grad = True
    hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == "classification":
        criterion = CrossEntropyLoss(reduction="mean")
        sigma_noise = 1.0
    elif likelihood == "regression":
        criterion = MSELoss(reduction="mean")
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)

    # Set up model optimizer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr_model)

    # # set up learning rate scheduler
    # if scheduler_cls is not None:
    #     if scheduler_kwargs is None:
    #         scheduler_kwargs = dict()
    #     scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    # Set up hyperparameter optimizer
    hyper_optimizer = torch.optim.Adam(hyperparameters, lr=lr_hyp)

    best_marglik = np.inf
    best_model_dict = None
    best_precision = None
    losses = list()
    margliks = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # standard NN training per batch
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if likelihood == "regression":
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = temperature / (2 * sigma_noise.square())
            else:
                crit_factor = temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            theta = parameters_to_vector(model.parameters())
            delta = expand_prior_precision(prior_prec, model)
            f = model(X)
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
            epoch_loss += loss.cpu().item() * len(y)
            if likelihood == "regression":
                epoch_perf += (f.detach() - y).square().sum()
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item()
            if scheduler_cls is not None:
                scheduler.step()

        # if epoch % cfg.logging_epoch_freq == 0:
        #     val_loss = loss_fn(val_loader)
        #     wandb.log({"val_loss": val_loss})
        #     train_metrics = compute_metrics(
        #         pred_fn=map_pred_fn,
        #         ds_test=ds_train,
        #         batch_size=cfg.batch_size,
        #         device=cfg.device,
        #     )
        #     val_metrics = compute_metrics(
        #         pred_fn=map_pred_fn,
        #         ds_test=ds_val,
        #         batch_size=cfg.batch_size,
        #         device=cfg.device,
        #     )
        #     test_metrics = compute_metrics(
        #         pred_fn=map_pred_fn,
        #         ds_test=ds_test,
        #         batch_size=cfg.batch_size,
        #         device=cfg.device,
        #     )
        #     wandb.log({"train/": train_metrics})
        #     wandb.log({"val/": val_metrics})
        #     wandb.log({"test/": test_metrics})
        #     wandb.log({"epoch": epoch})

        losses.append(epoch_loss / N)

        # compute validation error to report during training
        logging.info(
            f"MARGLIK[epoch={epoch}]: network training. Loss={losses[-1]:.3f}."
            + f"Perf={epoch_perf/N:.3f}"
        )

        # only update hyperparameters every marglik_frequency steps after burnin
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # optimizer hyperparameters by differentiating marglik
        # 1. fit laplace approximation
        sigma_noise = (
            1 if likelihood == "classification" else torch.exp(log_sigma_noise)
        )
        prior_prec = torch.exp(log_prior_prec)
        lap = Laplace(
            model,
            likelihood,
            hessian_structure=hessian_structure,
            sigma_noise=sigma_noise,
            prior_precision=prior_prec,
            temperature=temperature,
            backend=backend,
            subset_of_weights="all",
        )
        lap.fit(train_loader)

        # 2. differentiate wrt. hyperparameters for n_hypersteps
        for _ in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            if likelihood == "classification":
                sigma_noise = None
            elif likelihood == "regression":
                sigma_noise = torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise)
            marglik.backward()
            hyper_optimizer.step()
            margliks.append(marglik.item())

        # early stopping on marginal likelihood
        if margliks[-1] < best_marglik:
            best_model_dict = deepcopy(model.state_dict())
            best_precision = deepcopy(prior_prec.detach())
            best_sigma = (
                1 if likelihood == "classification" else deepcopy(sigma_noise.detach())
            )
            best_marglik = margliks[-1]
            logging.info(
                f"MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.2f}. "
                + "Saving new best model."
            )
        else:
            logging.info(
                f"MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.2f}."
                + f"No improvement over {best_marglik:.2f}"
            )

    logging.info("MARGLIK: finished training. Recover best model and fit Laplace.")
    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
    lap = Laplace(
        model,
        likelihood,
        hessian_structure=hessian_structure,
        sigma_noise=sigma_noise,
        prior_precision=prior_prec,
        temperature=temperature,
        backend=backend,
        subset_of_weights="all",
        enable_backprop=enable_backprop,
    )
    lap.fit(train_loader)
    return lap, model, margliks, losses


@torch.no_grad()
def predict_probs(
    dataloader: DataLoader, network: torch.nn.Module, device: str = "cpu"
):
    py = []
    for x, _ in dataloader:
        py.append(torch.softmax(network(x.to(device)), dim=-1))

    return torch.cat(py).cpu().numpy()


@hydra.main(version_base="1.3", config_path="./configs", config_name="marglik_train")
def train(cfg: DictConfig):
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    if "gpu" in cfg.device:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.double:
        logger.info("Using float64")
        torch.set_default_dtype(torch.double)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))

    # Load the data
    ds_train, ds_test = get_dataset(
        dataset=cfg.dataset,
        double=cfg.double,
        dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        device=cfg.device,
        debug=cfg.debug,
    )
    cfg.output_dim = ds_train.K

    # Instantiate SFR
    network = get_model(model_name=cfg.model_name, ds_train=ds_train).to(cfg.device)
    sfr = hydra.utils.instantiate(cfg.sfr, model=network)

    # Split train data set into train and validation
    print("num train {}".format(len(ds_train)))
    ds_train, ds_val, ds_test = train_val_split(ds_train=ds_train,  
                                                ds_test=ds_test, 
                                                val_from_test=cfg.val_from_test, 
                                                val_split=cfg.val_split)
    train_loader = DataLoader(dataset=ds_train, shuffle=True, batch_size=cfg.batch_size)
    val_loader = DataLoader(dataset=ds_val, shuffle=False, batch_size=cfg.batch_size)
    print("train_loader {}".format(train_loader))
    print("val_loader {}".format(val_loader))
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    # Initialise WandB
    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        shutil.copytree(
            os.path.abspath(".hydra"),
            os.path.join(os.path.join(get_original_cwd(), wandb.run.dir), "hydra"),
        )
        wandb.save("hydra")

    optimizer = torch.optim.Adam([{"params": sfr.network.parameters()}], lr=cfg.lr)

    # Hyperparameters (prior precision)
    hyperparameters = []
    # log_prior_prec_init = np.log(cfg.temperature * cfg.prior_prec_init)
    # log_prior_prec = log_prior_prec_init * torch.ones(1, device=cfg.device)
    # prior_prec = torch.exp(log_prior_prec)
    prior_prec = torch.tensor(cfg.prior_prec_init, requires_grad=True)
    # prior_prec.requires_grad = True
    sfr.prior.delta = prior_prec
    hyperparameters.append(sfr.prior.delta)

    hyper_optimizer = torch.optim.Adam(hyperparameters, lr=cfg.lr_hyp)
    # log_prior_prec.requires_grad = False

    @torch.no_grad()
    def map_pred_fn(x):
        return torch.softmax(sfr.network(x.to(cfg.device)), dim=-1)

    def loss_fn(data_loader: DataLoader):
        cum_loss = 0
        for X, y in data_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss
        return cum_loss

    for epoch in tqdm(list(range(cfg.n_epochs))):
        for X, y in train_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})

        if epoch % cfg.logging_epoch_freq == 0:
            val_loss = loss_fn(val_loader)
            wandb.log({"val_loss": val_loss})
            train_metrics = compute_metrics(
                pred_fn=map_pred_fn,
                ds_test=ds_train,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            val_metrics = compute_metrics(
                pred_fn=map_pred_fn,
                ds_test=ds_val,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            test_metrics = compute_metrics(
                pred_fn=map_pred_fn,
                ds_test=ds_test,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            wandb.log({"train/": train_metrics})
            wandb.log({"val/": val_metrics})
            wandb.log({"test/": test_metrics})
            wandb.log({"epoch": epoch})
            wandb.log({"delta_sfr": sfr.prior.delta})
            wandb.log({"delta": prior_prec})

        # only update hyperparameters every marglik_frequency steps after burnin
        if (epoch % cfg.marglik_frequency) != 0 or epoch < cfg.num_epochs_burnin:
            print("doing marg lik training")
            continue
        print("YEP DOING marg lik training")

        # optimizer hyperparameters by differentiating marglik
        # 1. fit laplace approximation
        # sigma_noise = (
        #     1 if likelihood == "classification" else torch.exp(log_sigma_noise)
        # )
        #
        # log_prior_prec.requires_grad = True
        # sigma_noise = 1.0
        lap = hydra.utils.instantiate(
            cfg.inference_strategy.model,
            model=sfr.network,
            prior_precision=sfr.prior.delta,
        )
        # lap = Laplace(
        #     sfr.network,
        #     cfg.inference_strategy.likelihood,
        #     hessian_structure=cfg.inference_strategy.hessian_structure,
        #     sigma_noise=sigma_noise,
        #     prior_precision=prior_prec,
        #     # temperature=cfg.temperature,
        #     backend=cfg.inference_strategy.backend,
        #     subset_of_weights=cfg.inference_strategy.subset_of_weights,
        # )
        lap.fit(train_loader)

        # 2. differentiate wrt. hyperparameters for n_hypersteps
        for _ in range(cfg.num_hypersteps):
            hyper_optimizer.zero_grad()
            sigma_noise = None
            # prior_prec = torch.exp(log_prior_prec)
            neg_log_marglik = -lap.log_marginal_likelihood(sfr.prior.delta, sigma_noise)
            neg_log_marglik.backward()
            hyper_optimizer.step()
            wandb.log({"neg_log_marglik": neg_log_marglik})
            wandb.log({"delta_sfr": sfr.prior.delta})
            wandb.log({"delta": prior_prec})
            # margliks.append(marglik.item())

        # # early stopping on marginal likelihood
        # if margliks[-1] < best_marglik:
        #     best_model_dict = deepcopy(model.state_dict())
        #     best_precision = deepcopy(prior_prec.detach())
        #     best_sigma = (
        #         1 if likelihood == "classification" else deepcopy(sigma_noise.detach())
        #     )
        #     best_marglik = margliks[-1]
        #     logging.info(
        #         f"MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.2f}. "
        #         + "Saving new best model."
        #     )
        # else:
        #     logging.info(
        #         f"MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.2f}."
        #         + f"No improvement over {best_marglik:.2f}"
        #     )

    logger.info("Finished training")

    state = {"model": sfr.state_dict(), "optimizer": optimizer.state_dict()}

    logger.info("Saving model and optimiser etc...")
    fname = "ckpt_dict.pt"
    torch.save(state, os.path.join(run.dir, fname))
    logger.info("Finished saving model and optimiser etc")


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
