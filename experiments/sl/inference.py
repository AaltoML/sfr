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
import src
import torch
import wandb
from experiments.sl.bnn_predictive.experiments.scripts.imgclassification import (
    get_dataset,
    get_model,
)
from experiments.sl.utils import compute_metrics, set_seed_everywhere, train_val_split
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


@hydra.main(version_base="1.3", config_path="./configs", config_name="inference")
def main(cfg: DictConfig):
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    ckpt_cfg = OmegaConf.load(
        os.path.join(get_original_cwd(), cfg.checkpoint, "files/config.yaml")
    )
    pprint("Loaded cfg from {}: {}".format(cfg.checkpoint, ckpt_cfg))

    if cfg.double:
        torch.set_default_dtype(torch.double)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))
    cfg.output_dim = ckpt_cfg.output_dim.value

    ds_train, ds_test = get_dataset(
        dataset=ckpt_cfg.dataset.value,
        # double=True,
        double=False,
        dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        device=cfg.device,
        debug=ckpt_cfg.debug.value,
    )

    # Instantiate the model and update from checkpoint
    ckpt_fname = os.path.join(get_original_cwd(), cfg.checkpoint, "files/ckpt_dict.pt")
    checkpoint = torch.load(ckpt_fname)
    network = get_model(model_name=ckpt_cfg.model_name.value, ds_train=ds_train)
    # torch.set_default_dtype(torch.double)
    # network = network.to(cfg.device).to(torch.double)
    network = network.to(cfg.device)
    # network = network.double()
    sfr = hydra.utils.instantiate(ckpt_cfg.sfr.value, model=network)
    # print(f"old delta: {sfr.prior.delta}")
    # print(checkpoint["model"])
    sfr.load_state_dict(checkpoint["model"])
    sfr = sfr.double()
    # print(f"new delta: {sfr.prior.delta}")
    sfr.eval()

    ds_train, ds_val = train_val_split(ds_train=ds_train, split=1 / 6)
    print("num train {}".format(len(ds_train)))
    print("num val {}".format(len(ds_val)))
    train_loader = DataLoader(dataset=ds_train, shuffle=True, batch_size=cfg.batch_size)
    val_loader = DataLoader(dataset=ds_val, shuffle=False, batch_size=cfg.batch_size)
    print("train_loader {}".format(train_loader))
    print("val_loader {}".format(val_loader))
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)
    # train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    # test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)
    # print("made train_loader {}".format(train_loader_double))

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
    def map_pred_fn(x):
        return torch.softmax(sfr.network(x.to(cfg.device)), dim=-1)

    map_metrics = compute_metrics(
        pred_fn=map_pred_fn, data_loader=test_loader, device=cfg.device
    )
    wandb.log({"map": map_metrics})

    # model = laplace.Laplace(
    #     likelihood="classification",
    #     subset_of_weights="all",
    #     hessian_structure="full",
    #     backend=laplace.curvature.asdl.AsdlGGN,
    # )
    print("making inference model")
    model = hydra.utils.instantiate(cfg.inference_strategy.model, model=sfr.network)
    print(f"model delta: {model.prior.delta}")
    if isinstance(model, laplace.BaseLaplace):
        model.prior_precision = sfr.prior.delta
    elif isinstance(model, src.SFR):
        model.prior.delta = sfr.prior.delta
    print(f"sfr delta: {sfr.prior.delta}")
    print(f"model delta: {model.prior.delta}")
    logger.info("Starting inference...")
    model.fit(train_loader)
    logger.info("Finished inference")

    print("building pred fn")
    for pred_cfg in cfg.inference_strategy.pred:
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
            )
        else:
            name = cfg.inference_strategy.name + "." + pred_cfg.pred_type
        wandb.log({name: metrics})

    if cfg.inference_strategy.optimize_prior_precision_kwargs is not None:
        model.optimize_prior_precision(
            **cfg.inference_strategy.optimize_prior_precision_kwargs
        )

        for pred_cfg in cfg.inference_strategy.pred:
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
    def pred_fn(x):
        return model(x, pred_type=pred_type, num_samples=num_samples)[0]

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


# def compute_metrics_(sfr, ds_train, ds_test, cfg):
#     # def compute_metrics(sfr, ds_train, ds_test, cfg, checkpoint):
#     # Split the test data set into test and validation sets
#     num_test = len(ds_test)
#     perm_ixs = torch.randperm(num_test)
#     val_ixs, test_ixs = perm_ixs[: int(num_test / 2)], perm_ixs[int(num_test / 2) :]
#     ds_val = Subset(ds_test, val_ixs)
#     ds_test = Subset(ds_test, test_ixs)
#     val_loader = get_quick_loader(
#         DataLoader(ds_val, batch_size=cfg.batch_size), device=cfg.device
#     )
#     test_loader = get_quick_loader(
#         DataLoader(ds_test, batch_size=cfg.batch_size), device=cfg.device
#     )
#     all_train = DataLoader(ds_train, batch_size=len(ds_train))
#     (X_train, y_train) = next(iter(all_train))
#     X_train = X_train.to(cfg.device)
#     y_train = y_train.to(cfg.device)
#     data = (X_train.to(torch.float64), y_train)

#     model = sfr

#     # MAP
#     @torch.no_grad()
#     def predict_probs(dataloader: DataLoader, map: bool = False):
#         py = []
#         for x, _ in dataloader:
#             if map:
#                 py.append(torch.softmax(sfr.network(x.to(cfg.device)), dim=-1))
#             else:
#                 py.append(model(x.to(cfg.device)))

#         return torch.cat(py).cpu().numpy()

#     # logging.info("MAP performance")
#     map_metrics = evaluate(
#         test_loader=test_loader, predict_probs=partial(predict_probs, map=True)
#     )
#     print("map_metrics {}".format(map_metrics))
#     wandb.log({"map": map_metrics})

#     # conf_name = "map"
#     # logging.info("MAP performance")
#     # gstar_te, yte = get_map_predictive(test_loader, sfr.network)
#     # gstar_va, yva = get_map_predictive(val_loader, sfr.network)
#     # checkpoint["map"] = evaluate(sfr.likelihood, yte, gstar_te, yva, gstar_va)
#     # logging.info(checkpoint["map"])
#     # wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})
#     # print("checkpoint[map] {}".format(checkpoint["map"]))

#     # logging.info("SFR performance")

#     # model.fit(data)
#     # # gp_subset.set_data(data)
#     # # sfr.set_data(data)
#     # # la.fit(data)

#     # conf_name = "predict"
#     # # conf_name = f"{cfg.model_name.num_inducing}"

#     # # logging.info(f"Computing {conf_name}")
#     # gstar_te, yte = predict_fn(test_loader)
#     # gstar_va, yva = predict_fn(val_loader)
#     # checkpoint[conf_name] = evaluate(sfr.likelihood, yte, gstar_te, yva, gstar_va)
#     # logging.info(checkpoint[conf_name])
#     # wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})

#     model = hydra.utils.instantiate(
#         cfg.inference_strategy, model=sfr.network, prior_precision=sfr.prior.delta
#     )
#     print("bnn {}".format(model))
#     # la = laplace.Laplace(
#     #     sfr.network,
#     #     "classification",
#     #     subset_of_weights=cfg.subset_of_weights,
#     #     hessian_structure=cfg.hessian_structure,
#     #     prior_precision=sfr.prior.delta,
#     #     backend=laplace.curvature.asdl.AsdlGGN,
#     # )

#     # print("Making train_loader fo LA...")
#     # # train_loader = DataLoader(ds_train, batch_size=cfg.inference_batch_size)
#     train_loader_double = DataLoader(TensorDataset(*data), batch_size=cfg.batch_size)
#     print("made train_loader {}".format(train_loader_double))

#     logger.info("Starting inference...")
#     # la.fit(train_loader_double)
#     model.fit(train_loader_double)
#     logger.info("Finished inference")

#     # GLM predictive
#     # conf_name = "glm"
#     logging.info("GLM")
#     glm_metrics = evaluate(test_loader=test_loader, predict_probs=predict_probs)
#     print("glm_metrics {}".format(glm_metrics))
#     wandb.log({"glm": glm_metrics})
#     print("delta {}".format(model.prior_precision))

#     model.optimize_prior_precision(method="marglik")
#     glm_with_prior_opt_metrics = evaluate(
#         test_loader=test_loader, predict_probs=predict_probs
#     )
#     print("delta {}".format(model.prior_precision))
#     print("glm_with_prior_opt_metrics {}".format(glm_with_prior_opt_metrics))
#     wandb.log({"glm_posthoc_opt": glm_with_prior_opt_metrics})

# def glm_predictive(x):
#     def la_pred(x):
#         return la.predictive_samples(x=x, pred_type="glm", n_samples=100)

#     gstar_te, yte = get_lap_predictive(test_loader, la_pred, seeding=True)

# gstar_te, yte = get_lap_predictive(test_loader, la_pred, seeding=True)
# gstar_va, yva = get_lap_predictive(val_loader, la_pred, seeding=True)
# checkpoint[conf_name] = evaluate(sfr.likelihood, yte, gstar_te, yva, gstar_va)
# logging.info(checkpoint[conf_name])
# wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})
# print("checkpoint[glm] {}".format(checkpoint["glm"]))

# la = laplace.Laplace(
#     sfr.network,
#     "classification",
#     subset_of_weights=cfg.subset_of_weights,
#     hessian_structure=cfg.hessian_structure,
#     prior_precision=sfr.prior.delta,
#     backend=laplace.curvature.asdl.AsdlGGN,
# )

# print("Making train_loader fo LA...")
# train_loader_double = DataLoader(
#     TensorDataset(*data), batch_size=cfg.inference_batch_size
# )

# print("made train_loader {}".format(train_loader_double))

# logger.info("Fitting laplace...")
# la.fit(train_loader_double)
# logger.info("Finished fitting laplace")

# # BNN predictive
# conf_name = "bnn"
# logging.info("BNN predictive")

# def la_pred(x):
#     return la.predictive_samples(x=x, pred_type="nn", n_samples=100)

# gstar_te, yte = get_la_predictive(test_loader, la_pred, seeding=True)
# gstar_va, yva = get_la_predictive(val_loader, la_pred, seeding=True)
# checkpoint[conf_name] = evaluate(gp_subset.likelihood, yte, gstar_te, yva, gstar_va)
# logging.info(checkpoint[conf_name])
# wandb.log({f"{conf_name}_{k}": v for k, v in checkpoint[conf_name].items()})

#

# res_dir = "./saved_inference_results"
# if not os.path.exists(res_dir):
#     os.makedirs(res_dir)
# fname = (
#     "./"
#     + "_".join([ckpt_cfg.dataset, cfg.model_name, str(cfg.random_seed)])
#     + f"_{cfg.prior.delta:.1e}.pt"
# )
# torch.save(checkpoint, os.path.join(res_dir, fname))


if __name__ == "__main__":
    main()
