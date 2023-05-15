import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.distributions import Categorical, Normal
from tqdm import tqdm
import logging
from itertools import chain

# from preds.laplace import Laplace, FunctionaLaplace
from preds.laplace import FunctionaLaplace
from src import CategoricalLh, NTKSVGP, NN2GPSubset
import src as ntksvgp
from preds.utils import nll_cls, macc, ece
from preds.datasets import MNIST, FMNIST, SVHN
from imgclassification import get_model, get_dataset


def get_ood_dataset(dataset):
    if dataset == "MNIST":
        return FMNIST(train=False)
    elif dataset == "FMNIST":
        return MNIST(train=False)
    elif dataset == "CIFAR10":
        return SVHN(train=False)
    else:
        raise ValueError("No OOD available.")


def get_map_predictive(loader, model):
    ys, pstar = list(), list()
    for X, y in loader:
        X, y = X, y
        ys.append(y)
        pstar.append(torch.softmax(model(X), dim=-1).detach())
    ys = torch.cat(ys)
    pstar = torch.cat(pstar)
    return pstar, ys


def get_lap_predictive(loader, lap):
    ys, ps = list(), list()
    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        ps.append(lap.predictive_samples_glm(X, n_samples=100).mean(dim=0))
        ys.append(y)
    ps = torch.cat(ps)
    ys = torch.cat(ys)
    return ps, ys


def get_nn_predictive(loader, lap, seeding=False):
    ys, ps = list(), list()
    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        if seeding:
            torch.manual_seed(711)
        ps.append(lap.predictive_samples_bnn(X, n_samples=100).mean(dim=0))
        ys.append(y)
    ps = torch.cat(ps)
    ys = torch.cat(ys)
    return ps, ys


def get_svgp_predictive(
    loader, svgp, likelihood, use_nn_out: bool = True, seeding: bool = False
):
    ys, ps = list(), list()
    output_dim = svgp.output_dim
    for X, y in loader:
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


def evaluate_one(lh, y, g):
    res = dict()
    res["nll"] = nll_cls(g, y, lh)
    res["acc"] = macc(g, y)
    res["ece"] = ece(g, y)
    return res


def predictive_entropies(gstar_te, gstar_od):
    te_dist = Categorical(probs=gstar_te)
    te_ents = te_dist.entropy().cpu().numpy()
    od_dist = Categorical(probs=gstar_od)
    od_ents = od_dist.entropy().cpu().numpy()
    return dict(test=te_ents, ood=od_ents)


def get_perf_dict():
    return dict(
        nll_tr=list(),
        nll_te=list(),
        acc_tr=list(),
        acc_te=list(),
        ece_tr=list(),
        ece_te=list(),
    )


def get_quick_loader(loader, device="cuda"):
    return [(X.to(device), y.to(device)) for X, y in loader]

def update_with_ood(svgp, likelihood,  ds_ood, update_size=100, batch_size=512):

    perm_ixs = torch.randperm(len(ds_ood))
    update_ixs, test_ixs = perm_ixs[:update_size], perm_ixs[int(M / 2) :]
    ds_ood_update = Subset(ds_ood, val_ixs)
    ds_ood_test = Subset(ds_ood, test_ixs)

    test_ood_loader = get_quick_loader(DataLoader(ds_ood_test, batch_size=batch_size))

    update_X, update_y = ds_ood
    svgp.update(update_X, update_y)
    gstar_ood, yte = get_svgp_predictive(
            test_ood_loader, svgp, use_nn_out=False, likelihood=likelihood
        )
    state[f"svgp_ntk_{name}_ood"] = evaluate_one(lh, yte, gstar_ood)

def retrain_and_update(state, svgp, likelihood, ds_test, ds_train, name='sparse_500',
    update_size=100, batch_size=500, device='cpu'):
    # get update dataset
    perm_ixs = torch.randperm(len(test))
    update_ixs, test_ixs = perm_ixs[:update_size], perm_ixs[update_size:]
    ds_update = Subset(ds_test, update_ixs)
    ds_update_test = Subset(ds_test, test_ixs)

    # update SVGP
    all_update = DataLoader(ds_update, batch_size=len(ds_update))
    batched_update = DataLoader(ds_update, batch_size=batch_size)
    (X_update, y_update) = next(iter(all_update))
    svgp.update(update_X, update_y)

    gstar_update, yte = get_svgp_predictive(
            batched_update, svgp, use_nn_out=False, likelihood=likelihood
        )
    state[f"svgp_ntk_{name}_updated"] = evaluate_one(likelihood, yte, gstar_update)

    # Retrain network
    train_update_loader = ds_train + ds_update

    model = retrain_nn(svgp, train_update_loader, delta=svgp.prior.delta)
    gstar_update, yte = get_map_predictive(model=model, loader=batched_update)
    state[f'nn_retrain_map'] = evaluate_one(likelihood, yte, gstar_update)


def retrain_nn(svgp, train_loader, delta, lr=1e-3, device='cpu', n_epochs=500):
    optim = Adam(svgp.network.parameters(), lr=lr, weight_decay=delta)
    scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1/(epoch // 10 + 1))
    for epoch in tqdm(list(range(n_epochs))):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            loss = svgp.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return svgp.network



def main(
    dataset_name,
    ds_train,
    ds_test,
    model_name,
    rerun,
    batch_size,
    seed,
    n_sparse,
    name,
    n_inducing,
    logdelta_min=-5,
    logdelta_max=8,
    res_dir="experiments/results",
    run_update=False,
    device="cuda",
):
    lh = CategoricalLh(EPS=0)  #0.000000000000000001)

    eligible_files = list()
    deltas = list()
    for file in os.listdir(os.path.join(res_dir, "models")):
        # strip off filename ending using indexing
        print(file)
        ds, m, s, delta = file[:-3].split("_")
        if (
            ds == dataset_name
            and m == model_name
            and seed == int(s)
            and (np.log10(float(delta)) >= delta_min)
            and (np.log10(float(delta)) <= delta_max)
        ):
            eligible_files.append(os.path.join(res_dir, "models/" + file))
            deltas.append(float(delta))
    logging.info(f"Deltas: {deltas}")
    # start with smallest delta and continue
    ixlist = np.argsort(deltas)
    deltas = np.array(deltas)[ixlist]
    eligible_files = list(np.array(eligible_files)[ixlist])

    train_loader = DataLoader(ds_train, batch_size=256)
    all_train = DataLoader(ds_train, batch_size=len(ds_train))
    (X_train, y_train) = next(iter(all_train))
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    torch.manual_seed(seed)
    M = len(ds_test)
    n_inducing = n_inducing_points  # int(len(ds_train)*n_sparse)
    logging.info(f"Train set size: {len(ds_train)}")
    logging.info(f"Num inducing points: {n_inducing}")
    print(X_train.shape)
    perm_ixs = torch.randperm(M)
    val_ixs, test_ixs = perm_ixs[: int(M / 2)], perm_ixs[int(M / 2) :]
    ds_val = Subset(ds_test, val_ixs)
    ds_test = Subset(ds_test, test_ixs)
    val_loader = get_quick_loader(
        DataLoader(ds_val, batch_size=batch_size), device=device
    )
    test_loader = get_quick_loader(
        DataLoader(ds_test, batch_size=batch_size), device=device
    )

    for f, delta in tqdm(list(zip(eligible_files, deltas))):
        logging.info(f"inference for delta={delta}")
        state = torch.load(f)
        # if 'map' in state and not rerun:
        #     # do not recompute the metrics
        #     print("map in state")
        #     continue

        model = get_model(model_name, ds_train)
        model.load_state_dict(state["model"])
        model = model.to(device)

        if device == "cuda":
            model = model.cuda()

        # MAP
        logging.info("MAP performance")
        gstar_te, yte = get_map_predictive(test_loader, model)
        gstar_va, yva = get_map_predictive(val_loader, model)
        state["map"] = evaluate(lh, yte, gstar_te, yva, gstar_va)
        logging.info(state['map'])

        # SVGP
        logging.info("SVGP performance")

        data = (X_train, y_train)
        prior = ntksvgp.priors.Gaussian(params=model.parameters, delta=delta)
        output_dim = model(X_train[:10].to(device)).shape[-1]
        svgp = NTKSVGP(
            network=model,
            prior=prior,
            output_dim=output_dim,
            likelihood=lh,
            num_inducing=n_inducing,
            dual_batch_size=batch_size,
            jitter=0,#10**(-9),
            device=device,
        )
        svgp.set_data(data)
        gstar_te, yte = get_svgp_predictive(
            test_loader, svgp, use_nn_out=False, likelihood=lh
        )
        gstar_va, yva = get_svgp_predictive(
            val_loader, svgp, use_nn_out=False, likelihood=lh
        )
        state[f"svgp_ntk_{name}"] = evaluate(lh, yte, gstar_te, yva, gstar_va)
        logging.info(state[f"svgp_ntk_{name}"])

        gstar_te, yte = get_svgp_predictive(
            test_loader, svgp, use_nn_out=True, likelihood=lh
        )
        gstar_va, yva = get_svgp_predictive(
            val_loader, svgp, use_nn_out=True, likelihood=lh
        )
        state[f"svgp_ntk_nn_{name}"] = evaluate(lh, yte, gstar_te, yva, gstar_va)
        logging.info(state[f"svgp_ntk_nn_{name}"])


        # GP subset
        logging.info("GP subset")
        prior = ntksvgp.priors.Gaussian(params=model.parameters, delta=delta)
        svgp_subset = NN2GPSubset(
            network=model,
            prior=prior,
            output_dim=output_dim,
            likelihood=lh,
            subset_size=n_inducing,
            dual_batch_size=batch_size,
            jitter=0,
            device=device,
        )
        svgp_subset.set_data(data)
        gstar_te, yte = get_svgp_predictive(
            test_loader, svgp_subset, use_nn_out=False, likelihood=lh
        )
        gstar_va, yva = get_svgp_predictive(
            val_loader, svgp_subset, use_nn_out=False, likelihood=lh
        )
        state[f"gp_subset_{name}"] = evaluate(lh, yte, gstar_te, yva, gstar_va)
        logging.info(state[f"gp_subset_{name}"])

        # retrain and update
        if run_update:
           state = retrain_and_update(state, svgp, likelihood, ds_test, ds_train, name=name,
                                        update_size=1000, batch_size=batch_size, device=device)


        torch.save(state, f)


def ood(
    dataset_name,
    ds_train,
    ds_test,
    ds_ood,
    model_name,
    batch_size,
    seed,
    n_inducing,
    name,
    res_dir="experiments/results",
    device="cuda",
    run_update=False
):
    lh = CategoricalLh(EPS=0.0)

    perf_keys = ["map", f"svgp_ntk_nn_{name}", f"svgp_ntk_{name}"]
    eligible_files = list()
    perfs = list()
    for file in os.listdir(os.path.join(res_dir, "models")):
        # strip off filename ending using indexing
        ds, m, s, delta = file[:-3].split("_")
        if (
            ds == dataset_name
            and m == model_name
            and float(delta) > 0
            and seed == int(s)
        ):

            state = torch.load(os.path.join(res_dir, "models/" + file))
            if 'map' not in state:
                continue
            eligible_files.append(os.path.join(res_dir, "models/" + file))
            perfs.append({k: state[k]["nll_va"] for k in perf_keys})

    train_loader = DataLoader(ds_train, batch_size=256)
    test_loader = get_quick_loader(DataLoader(ds_test, batch_size=batch_size))
    ood_loader = get_quick_loader(DataLoader(ds_ood, batch_size=batch_size))

    pred_ents = dict()

    # MAP
    model = get_model(model_name, ds_train)
    state = torch.load(eligible_files[np.argmin([e["map"] for e in perfs])])
    logging.info(f'MAP inference - best delta={state["delta"]}')
    model.load_state_dict(state["model"])
    model = model.cuda()
    gstar_te, _ = get_map_predictive(test_loader, model)
    gstar_od, _ = get_map_predictive(ood_loader, model)
    pred_ents["map"] = predictive_entropies(gstar_te, gstar_od)

    # Get train data for SVGP
    all_train = DataLoader(ds_train, batch_size=len(ds_train))
    (X_train, y_train) = next(iter(all_train))
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # SVGP
    model = get_model(model_name, ds_train)
    state = torch.load(eligible_files[np.argmin([e[f"svgp_ntk_{name}"] for e in perfs])])
    logging.info(f'SVGP predictive - best delta={state["delta"]}')
    model.load_state_dict(state["model"])
    model = model.to(device)

    data = (X_train, y_train)
    prior = ntksvgp.priors.Gaussian(params=model.parameters, delta=float(delta))
    output_dim = model(X_train[:10].to(device)).shape[-1]
    svgp = NTKSVGP(
        network=model,
        prior=prior,
        output_dim=output_dim,
        likelihood=lh,
        num_inducing=n_inducing,
        dual_batch_size=batch_size,
        jitter=0,
        device=device,
    )
    svgp.set_data(data)
    gstar_te, _ = get_svgp_predictive(
        test_loader, svgp, use_nn_out=False, likelihood=lh
    )
    gstar_va, _ = get_svgp_predictive(ood_loader, svgp, use_nn_out=False, likelihood=lh)
    pred_ents["svgp_ntk"] = predictive_entropies(gstar_te, gstar_od)

    # SVGP NN
    model = get_model(model_name, ds_train)
    state = torch.load(eligible_files[np.argmin([e[f"svgp_ntk_nn_{name}"] for e in perfs])])
    logging.info(f'SVGP predictive - best delta={state["delta"]}')
    model.load_state_dict(state["model"])
    model = model.to(device)

    gstar_te, _ = get_svgp_predictive(
        test_loader, svgp, use_nn_out=True, likelihood=lh
    )
    gstar_va, _ = get_svgp_predictive(ood_loader, svgp, use_nn_out=False, likelihood=lh)
    pred_ents["svgp_ntk"] = predictive_entropies(gstar_te, gstar_od)


    # SVGP update
    if run_update:
        pred_ents = update_with_ood(pred_ents, svgp, likelihood,  ds_ood, update_size=100, batch_size=512)

    # # # Laplace Kron GLM
    # model = get_model(model_name, ds_train)
    # state = torch.load(eligible_files[np.argmin([e['lap_kron'] for e in perfs])])
    # logging.info(f'Laplace Kron GLM inference - best delta={state["delta"]}')
    # model.load_state_dict(state['model'])
    # model = model.cuda()
    # lap = Laplace(model, state['delta'], lh)
    # lap.infer(train_loader, cov_type='kron')
    # mstar_te, _ = get_lap_predictive(tqdm(test_loader), lap)
    # mstar_od, _ = get_lap_predictive(tqdm(ood_loader), lap)
    # pred_ents['lap_kron'] = predictive_entropies(mstar_te, mstar_od)
    #
    # # Laplace Kron NN
    # model = get_model(model_name, ds_train)
    # state = torch.load(eligible_files[np.argmin([e['lap_kron_nn'] for e in perfs])])
    # logging.info(f'Laplace Kron NN inference - best delta={state["delta"]}')
    # model.load_state_dict(state['model'])
    # model = model.cuda()
    # lap = Laplace(model, state['delta'], lh)
    # lap.infer(train_loader, cov_type='kron')
    # mstar_te, _ = get_nn_predictive(tqdm(test_loader), lap)
    # mstar_od, _ = get_nn_predictive(tqdm(ood_loader), lap)
    # pred_ents['lap_kron_nn'] = predictive_entropies(mstar_te, mstar_od)
    #
    # # Laplace Damp
    # model = get_model(model_name, ds_train)
    # state = torch.load(eligible_files[np.argmin([e['lap_kron_damp'] for e in perfs])])
    # logging.info(f'Laplace Kron damp GLM inference - best delta={state["delta"]}')
    # model.load_state_dict(state['model'])
    # model = model.cuda()
    # lap = Laplace(model, state['delta'], lh)
    # lap.infer(train_loader, cov_type='kron', dampen_kron=True)
    # mstar_te, _ = get_lap_predictive(tqdm(test_loader), lap)
    # mstar_od, _ = get_lap_predictive(tqdm(ood_loader), lap)
    # pred_ents['lap_kron_damp'] = predictive_entropies(mstar_te, mstar_od)
    #
    # # Laplace NN Damp
    # model = get_model(model_name, ds_train)
    # state = torch.load(eligible_files[np.argmin([e['lap_kron_dampnn'] for e in perfs])])
    # logging.info(f'Laplace Kron damp NN inference - best delta={state["delta"]}')
    # model.load_state_dict(state['model'])
    # model = model.cuda()
    # lap = Laplace(model, state['delta'], lh)
    # lap.infer(train_loader, cov_type='kron', dampen_kron=True)
    # mstar_te, _ = get_nn_predictive(tqdm(test_loader), lap)
    # mstar_od, _ = get_nn_predictive(tqdm(ood_loader), lap)
    # pred_ents['lap_kron_dampnn'] = predictive_entropies(mstar_te, mstar_od)
    #
    # # Laplace Diag
    # model = get_model(model_name, ds_train)
    # state = torch.load(eligible_files[np.argmin([e['lap_diag'] for e in perfs])])
    # logging.info(f'Laplace diag GLM inference - best delta={state["delta"]}')
    # model.load_state_dict(state['model'])
    # model = model.cuda()
    # lap = Laplace(model, state['delta'], lh)
    # lap.infer(train_loader, cov_type='diag')
    # mstar_te, _ = get_lap_predictive(tqdm(test_loader), lap)
    # mstar_od, _ = get_lap_predictive(tqdm(ood_loader), lap)
    # pred_ents['lap_diag'] = predictive_entropies(mstar_te, mstar_od)
    #
    # # Laplace Diag NN
    # model = get_model(model_name, ds_train)
    # state = torch.load(eligible_files[np.argmin([e['lap_diag_nn'] for e in perfs])])
    # logging.info(f'Laplace diag NN inference - best delta={state["delta"]}')
    # model.load_state_dict(state['model'])
    # model = model.cuda()
    # lap = Laplace(model, state['delta'], lh)
    # lap.infer(train_loader, cov_type='diag')
    # mstar_te, _ = get_nn_predictive(tqdm(test_loader), lap)
    # mstar_od, _ = get_nn_predictive(tqdm(ood_loader), lap)
    # pred_ents['lap_diag_nn'] = predictive_entropies(mstar_te, mstar_od)

    fname = f"{res_dir}/{dataset_name}_{model_name}_{seed}_{name}_ood.pkl"
    logging.info(f"save {fname}")
    with open(fname, "wb") as f:
        pickle.dump(pred_ents, f)


def gp(dataset_name, ds_train, ds_test, ds_ood, model_name, batch_size, seed):
    lh = CategoricalLh()
    eligible_files = list()
    perfs = list()
    for file in os.listdir("models"):
        # strip off filename ending using indexing
        ds, m, s, delta = file[:-3].split("_")
        if (
            ds == dataset_name
            and m == model_name
            and float(delta) > 0
            and seed == int(s)
        ):
            eligible_files.append("models/" + file)
            state = torch.load("models/" + file)
            if "map" not in state:
                continue
            perfs.append(state["map"]["nll_va"])
    best_file = eligible_files[np.argmin(perfs)]
    model = get_model(model_name, ds_train)
    state = torch.load(best_file)
    model.load_state_dict(state["model"])
    model = model.cuda()

    train_loader = DataLoader(ds_train, batch_size=512, shuffle=False)
    ood_loader = DataLoader(ds_ood, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    N = len(ds_test)
    perm_ixs = torch.randperm(N)
    val_ixs, test_ixs = perm_ixs[: int(N / 2)], perm_ixs[int(N / 2) :]
    ds_val = Subset(ds_test, val_ixs)
    ds_test = Subset(ds_test, test_ixs)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    val, test = int(N / 2), N
    get_val_test_ood = lambda: chain(val_loader, test_loader, ood_loader)

    yte = torch.cat([e[1] for e in test_loader])
    yva = torch.cat([e[1] for e in val_loader])

    random_ixs = torch.randperm(len(ds_train))
    # use sum(p - p^2)=tr(lam) as in FROMP
    # NOTE: maximized if uniform prediction
    diag_lam = list()
    for X, _ in train_loader:
        p = torch.softmax(model(X.cuda()).detach(), dim=-1)
        lam = p - p.square()
        diag_lam.append(lam.cpu())
    trace_lam = torch.sum(torch.cat(diag_lam), dim=-1)

    gp_perf = dict(delta=state["delta"])

    for method in ["random", "topk"]:
        logging.info(f"Running for method {method}")
        for M in [50, 100, 200, 400, 800, 1600, 3200]:
            logging.info(f"Running for M={M} and method {method}")

            if method == "random":
                subset_ixs = random_ixs[:M]
            elif method == "topk":
                subset_ixs = torch.topk(trace_lam, M)[1]
            else:
                raise ValueError("Invalid method")

            # create corresponding subset of training data
            ds_train_sub = Subset(ds_train, subset_ixs)
            train_loader_sub = DataLoader(
                ds_train_sub, shuffle=False, batch_size=batch_size
            )

            # infer for corrected delta
            delta_sub = state["delta"] * len(ds_train) / M
            lap = FunctionaLaplace(model, delta_sub, lh)
            lap.infer(
                train_loader_sub,
                get_val_test_ood(),
                print_progress=False,
                max_ram_gb=12,
                batch_size_per_gpu=batch_size,
            )

            for delta_factor in np.logspace(-1, 3, 13):
                logging.info(
                    f"Running for M={M} and method {method} with deltafac={delta_factor}"
                )
                key = method + "-" + str(M) + "-" + str(delta_factor)
                lap.prior_prec = delta_sub * delta_factor
                pred_vto = lap.predictive_samples().cpu()
                pred_val, pred_test, pred_ood = (
                    pred_vto[:val],
                    pred_vto[val:test],
                    pred_vto[test:],
                )

                gp_perf[key] = {
                    "ood": predictive_entropies(pred_test, pred_ood),
                    "perf": evaluate(lh, yte, pred_test, yva, pred_val),
                }

    fname = f"results/{dataset_name}_{model_name}_{seed}_{name}_GP.pkl"
    logging.info(f"save {fname}")
    with open(fname, "wb") as f:
        pickle.dump(gp_perf, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    datasets = ["MNIST", "FMNIST", "CIFAR10"]
    models = ["CNN", "AllCNN", "MLP"]
    parser.add_argument("-d", "--dataset", help="dataset", choices=datasets)
    parser.add_argument("-m", "--model", help="which model to train", choices=models)
    parser.add_argument(
        "-r", "--rerun", help="recompute for models", action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size", help="Jac/Kernel batch size", type=int, default=1000
    )
    parser.add_argument("-s", "--seed", help="randomness seed", default=117, type=int)
    parser.add_argument("--gp", help="functional inference", action="store_true")
    parser.add_argument("--ood", help="out of distribution", action="store_true")
    parser.add_argument("--res_folder", help="Result folder", default="test")
    parser.add_argument("--logdelta_min", type=float, default=1e-7)
    parser.add_argument("--logdelta_max", type=float, default=1e7)
    parser.add_argument("--n_sparse", type=float, default=0.5)


    parser.add_argument(
        "--run_update", help="on/off switch for posterior refinement", type=int
    )
    parser.add_argument(
        "--n_inducing_points", type=int, default=1000
    )  # TODO: use 3200 (Immer et al.) as default
    parser.add_argument(
        "--name",
        type=str,
        default="testlocal",
        help="Name of run to be saved in dict key",
    )
    parser.add_argument("--loginfo", action="store_true", help="log info")
    parser.add_argument("--root_dir", help="Root directory", default="../")
    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model
    rerun = args.rerun
    n_sparse = args.n_sparse
    n_inducing_points = args.n_inducing_points
    root_dir = args.root_dir
    res_folder = args.res_folder
    run_update = args.run_update
    run_update = bool(run_update)
    log_delta = args.log_delta
    name = args.name

    data_dir = os.path.join(root_dir, "data")
    res_dir = os.path.join(root_dir, "experiments", "results", dataset, res_folder)

    print(f"Writing results to {res_dir}")
    print(f"Reading data from {data_dir}")
    print(f'Experiment name: {name}')
    print(f"Dataset: {dataset}")
    print(f"Seed: {args.seed}")


    torch.set_default_dtype(torch.double)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    logging.basicConfig(level=logging.INFO if args.loginfo else logging.WARNING)
    ds_train, ds_test = get_dataset(dataset, False, data_dir)
    # ds_train.data = ds_train.data[:1000]
    # ds_train.targets = ds_train.targets[:1000]
    if args.gp:
        # does inference and OOD at once!
        logging.info(f"Run GP inference with {dataset} using {model_name}")
        ds_ood = get_ood_dataset(dataset)
        # batch size * 10 due to classes
        gp(
            dataset,
            ds_train,
            ds_test,
            ds_ood,
            model_name,
            args.batch_size * 10,
            args.seed,
        )
    elif args.ood:
        logging.info(f"Run OOD with {dataset} using {model_name}")
        ds_ood = get_ood_dataset(dataset)
        ood(
            dataset,
            ds_train,
            ds_test,
            ds_ood,
            model_name,
            args.batch_size,
            args.seed,
            n_inducing_points,
            name=name,
            res_dir=res_dir,
            device=device,
            run_update=run_update
        )
    else:
        logging.info(f"Run inference with {dataset} using {model_name}")
        main(
            dataset,
            ds_train,
            ds_test,
            model_name,
            rerun,
            args.batch_size,
            args.seed,
            n_sparse,
            name,
            n_inducing_points,
            args.logdelta_min,
            args.logdelta_max,
            res_dir,
            run_update=run_update,
            device=device,
        )
