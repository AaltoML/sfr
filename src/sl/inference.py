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

# from src.sl.datasets import CIFAR10, FMNIST, MNIST
# from src.sl.networks import CIFAR10Net, CIFAR100Net, MLPS
from preds.datasets import CIFAR10, FMNIST, MNIST
from preds.models import CIFAR10Net, CIFAR100Net, MLPS

# from train import get_dataset, get_model, set_seed_everywhere
from torch.distributions import Categorical, Normal
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm


# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def set_seed_everywhere(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(cfg.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    # pl.seed_everything(random_seed)


def get_dataset(dataset, double, dir, cfg, device=None):
    if dataset == "MNIST":
        # Download training data from open datasets.
        ds_train = MNIST(train=True, double=double, root=dir)
        ds_test = MNIST(train=False, double=double, root=dir)
    elif dataset == "FMNIST":
        ds_train = FMNIST(train=True, double=double, root=dir)
        ds_test = FMNIST(train=False, double=double, root=dir)
    elif dataset == "CIFAR10":
        ds_train = CIFAR10(train=True, double=double, root=dir)
        ds_test = CIFAR10(train=False, double=double, root=dir)
    else:
        raise ValueError("Invalid dataset argument")
    if device is not None:
        if cfg.debug:
            ds_train.data = ds_train.data[:500]
            ds_train.targets = ds_train.targets[:500]
            ds_test.data = ds_test.data[:500]
            ds_test.targets = ds_test.targets[:500]
        return QuickDS(ds_train, device), QuickDS(ds_test, device)
    else:
        return ds_train, ds_test


def get_model(model_name, ds_train):
    if model_name == "MLP":
        input_size = ds_train.pixels**2 * ds_train.channels
        hidden_sizes = [1024, 512, 256, 128]
        output_size = ds_train.K
        return MLPS(input_size, hidden_sizes, output_size, "tanh", flatten=True)
    elif model_name == "SmallMLP":
        input_size = ds_train.pixels**2 * ds_train.channels
        hidden_sizes = [128, 128]
        output_size = ds_train.K
        return MLPS(input_size, hidden_sizes, output_size, "tanh", flatten=True)
    elif model_name == "CNN":
        return CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True)
    elif model_name == "AllCNN":
        return CIFAR100Net(ds_train.channels, ds_train.K)
    else:
        raise ValueError("Invalid model name")


def get_quick_loader(loader, device="cuda"):
    return [(X.to(device).to(torch.float64), y.to(device)) for X, y in loader]


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
        # X, y = X.cuda(), y.cuda() TODO put this back
        X, y = X, y
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
        # X, y = X.cuda(), y.cuda()
        # X, y = X.cuda(), y.cuda() TODO put this back
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
    print("inside sample svgp")
    n_data = X.shape[0]
    gp_means, gp_vars = svgp.predict_f(X)
    print("predeicted mean")
    logits = svgp.network(X)
    print("predeicted logits")
    if use_nn_out:
        dist = Normal(logits, torch.sqrt(gp_vars.clamp(10 ** (-32))))
    else:
        dist = Normal(gp_means, torch.sqrt(gp_vars.clamp(10 ** (-32))))
    print("made dist")
    logit_samples = dist.sample((n_samples,))
    print("logit samples")
    out_dim = logit_samples.shape[-1]
    samples = likelihood.inv_link(logit_samples)
    # samples = samples.reshape(n_samples, n_data, out_dim)
    return samples


def evaluate(lh, yte, gstar_te, yva, gstar_va):
    print("inside evaluate")
    res = dict()
    res["nll_te"] = nll_cls(gstar_te, yte, lh)
    print("after nll_cls")
    # res["nll_va"] = nll_cls(gstar_va, yva, lh)
    res["acc_te"] = macc(gstar_te, yte)
    print("after macc")
    # res["acc_va"] = macc(gstar_va, yva)
    res["ece_te"] = ece(gstar_te, yte)
    print("after ece")
    # res["ece_va"] = ece(gstar_va, yva)
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
    n_inducing = sfr.num_inducing  # int(len(ds_train)*n_sparse)
    logging.info(f"Train set size: {len(ds_train)}")
    logging.info(f"Num inducing points: {n_inducing}")
    perm_ixs = torch.randperm(M)
    val_ixs, test_ixs = perm_ixs[: int(M / 2)], perm_ixs[int(M / 2) :]
    ds_val = Subset(ds_test, val_ixs)
    ds_test = Subset(ds_test, test_ixs)
    val_loader = get_quick_loader(
        DataLoader(ds_val, batch_size=cfg.inference_batch_size), device=cfg.device
    )
    test_loader = get_quick_loader(
        DataLoader(ds_test, batch_size=cfg.inference_batch_size), device=cfg.device
    )
    all_train = DataLoader(ds_train, batch_size=len(ds_train))
    (X_train, y_train) = next(iter(all_train))
    X_train = X_train.to(cfg.device)
    y_train = y_train.to(cfg.device)
    data = (X_train.to(torch.float64), y_train)

    # if cfg.predictive_model == "map":
    # MAP
    conf_name = "map"
    logging.info("MAP performance")
    gstar_te, yte = get_map_predictive(test_loader, sfr.network)
    gstar_va, yva = get_map_predictive(val_loader, sfr.network)
    checkpoint["map"] = evaluate(sfr.likelihood, yte, gstar_te, yva, gstar_va)

    logging.info(checkpoint["map"])
    wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})
    if cfg.predictive_model == "sfr":
        logging.info("SFR performance")

        sfr.set_data(data)

        conf_name = f"sfr_sparse{sfr.num_inducing}"

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

        conf_name = f"sfr_nn_sparse{sfr.num_inducing}"
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

        conf_name = f"gp_subset_nn_sparse{gp_subset.subset_size}"
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

        conf_name = f"gp_subset_sparse{gp_subset.subset_size}"
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
    elif cfg.predictive_model == "glm":
        la = laplace.Laplace(
            sfr.network,
            "classification",
            subset_of_weights=cfg.subset_of_weights,
            hessian_structure=cfg.hessian_structure,
            prior_precision=sfr.prior.delta,
            backend=laplace.curvature.asdl.AsdlGGN,
        )
        # la.to(cfg.device)

        print("Making train_loader fo LA...")
        # train_loader = DataLoader(ds_train, batch_size=cfg.inference_batch_size)
        train_loader_double = DataLoader(
            TensorDataset(*data), batch_size=cfg.inference_batch_size
        )
        print("made train_loader {}".format(train_loader_double))

        logger.info("Fitting laplace...")
        la.fit(train_loader_double)
        logger.info("Finished fitting laplace")

        # GLM predictive
        conf_name = "glm"
        logging.info("GLM")

        def la_pred(x):
            # ys = []
            # for i in range(100):
            #     print("sample {}".format(i))
            #     ys.append(
            #         la.predictive_samples(
            #             x=x,
            #             pred_type="glm",
            #             n_samples=1,
            #             # diagonal_output=False,
            #             # generator=cfg.random_seed,
            #         )
            #     )
            #     torch.cuda.empty_cache()
            # return torch.stack(ys, 0)
            return la.predictive_samples(x=x, pred_type="glm", n_samples=100)

        # la_pred = partial(
        #     la.predictive_samples,
        #     pred_type="glm",
        #     n_samples=100,
        #     diagonal_output=False,
        #     # generator=cfg.random_seed,
        # )
        gstar_te, yte = get_la_predictive(test_loader, la_pred, seeding=True)
        gstar_va, yva = get_la_predictive(val_loader, la_pred, seeding=True)
        checkpoint[conf_name] = evaluate(
            gp_subset.likelihood, yte, gstar_te, yva, gstar_va
        )
        logging.info(checkpoint[conf_name])
        wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})

    elif cfg.predictive_model == "bnn":
        la = laplace.Laplace(
            sfr.network,
            "classification",
            subset_of_weights=cfg.subset_of_weights,
            hessian_structure=cfg.hessian_structure,
            prior_precision=sfr.prior.delta,
            backend=laplace.curvature.asdl.AsdlGGN,
        )
        # la.to(cfg.device)

        print("Making train_loader fo LA...")
        # train_loader = DataLoader(ds_train, batch_size=cfg.inference_batch_size)
        train_loader_double = DataLoader(
            TensorDataset(*data), batch_size=cfg.inference_batch_size
        )

        print("made train_loader {}".format(train_loader_double))

        logger.info("Fitting laplace...")
        la.fit(train_loader_double)
        logger.info("Finished fitting laplace")

        # BNN predictive
        conf_name = "bnn"
        logging.info("BNN predictive")

        def la_pred(x):
            ys = []
            # for i in range(100):
            #     print("sample {}".format(i))
            #     try:
            #         p = la.predictive_samples(
            #             x=x,
            #             pred_type="nn",
            #             n_samples=1,
            #             # diagonal_output=False,
            #             # generator=cfg.random_seed,
            #         )
            #     except Exception as e:
            #         print("e {}".format(e))
            #     print("p {}".format(p.shape))
            #     ys.append(p)
            #     torch.cuda.empty_cache()
            # return torch.stack(ys, 0)
            return la.predictive_samples(x=x, pred_type="nn", n_samples=100)

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
