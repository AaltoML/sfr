import logging

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import VisionDataset
import torchvision.datasets as datasets_torch
from torchvision.transforms import ToTensor
from tqdm import tqdm
from src import NTKSVGP, CategoricalLh
import src as ntksvgp
import os

from preds.models import CIFAR10Net,  MLPS, CIFAR100Net
from preds.datasets import MNIST, FMNIST, CIFAR10

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class QuickDS(VisionDataset):

    def __init__(self, ds, device):
        self.D = [(ds[i][0].to(device), torch.tensor(ds[i][1]).to(device))
                  for i in range(len(ds))]
        self.K = ds.K
        self.channels = ds.channels
        self.pixels = ds.pixels

    def __getitem__(self, index):
        return self.D[index]

    def __len__(self):
        return len(self.D)


def get_dataset(dataset, double, dir, device=None):
    if dataset == 'MNIST':
        # Download training data from open datasets.
        ds_train = MNIST(train=True, double=double, root=dir)
        ds_test = MNIST(train=False, double=double, root=dir)
    elif dataset == 'FMNIST':
        ds_train = FMNIST(train=True, double=double, root=dir)
        ds_test = FMNIST(train=False, double=double, root=dir)
    elif dataset == 'CIFAR10':
        ds_train = CIFAR10(train=True, double=double, root=dir)
        ds_test = CIFAR10(train=False, double=double, root=dir)
    else:
        raise ValueError('Invalid dataset argument')
    if device is not None:
        return QuickDS(ds_train, device), QuickDS(ds_test, device)
    else:
        return ds_train, ds_test


def get_model(model_name, ds_train):
    if model_name == 'MLP':
        input_size = ds_train.pixels ** 2 * ds_train.channels
        hidden_sizes = [1024, 512, 256, 128]
        output_size = ds_train.K
        return MLPS(input_size, hidden_sizes, output_size, 'tanh', flatten=True)
    elif model_name == 'CNN':
        return CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True)
    elif model_name == 'AllCNN':
        return CIFAR100Net(ds_train.channels, ds_train.K)
    else:
        raise ValueError('Invalid model name')


def evaluate(model, data_loader, criterion, device):
    model.eval()
    loss, acc = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            fs = model(X)
            acc += (torch.argmax(fs, dim=-1) == y).sum().cpu().float().item()
            loss += criterion(fs, y).item()
    return loss / len(data_loader.dataset), acc / len(data_loader.dataset)


def main(ds_train, ds_test, model_name, seed, n_epochs, batch_size, lr, deltas, device, fname, res_dir):
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    for delta in deltas:
        torch.manual_seed(seed)
        model = get_model(model_name, ds_train).to(device)
        optim = Adam(model.parameters(), lr=lr, weight_decay=delta)
        scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1/(epoch // 10 + 1))
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        losses = list()
        N = len(ds_train)
        # training
        for epoch in tqdm(list(range(n_epochs))):
            running_loss = 0.0
            for X, y in train_loader:
                # X, y = X.to(device), y.to(device)
                M = len(y)
                optim.zero_grad()
                fs = model(X)
                loss = N / M * criterion(fs, y)
                loss.backward()
                optim.step()
                p = parameters_to_vector(model.parameters()).detach()
                running_loss += loss.item() + (1/2 * delta * p.square().sum()).item()
            loss_avg = running_loss / len(train_loader)
            losses.append(loss_avg)
            scheduler.step()
        # evaluation
        tr_loss, tr_acc = evaluate(model, train_loader, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        metrics = {'test_loss': te_loss, 'test_acc': te_acc,
                   'train_loss': tr_loss, 'train_acc': tr_acc}

        state = {'model': model.state_dict(), 'optimizer': optim.state_dict(),
                 'losses': losses, 'metrics': metrics, 'delta': delta}
        torch.save(state, os.path.join(res_dir, fname.format(delta=delta)))

def main_new(ds_train, ds_test, model_name, seed, n_epochs, batch_size, lr, deltas, device, fname, res_dir):
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    likelihood = CategoricalLh()
    n_classes = next(iter(ds_train))[0].shape[-1]
    for delta in deltas:
        torch.manual_seed(seed)
        model = get_model(model_name, ds_train).to(device)
        #f_out = model(next(iter(ds_train))[0])
        prior = ntksvgp.priors.Gaussian(params=model.parameters, delta=delta)
        svgp = NTKSVGP(network=model, likelihood=likelihood, output_dim=n_classes, prior=prior, num_inducing=0)
        optimizer = torch.optim.Adam([{"params": svgp.parameters()}], lr=lr)
        losses = list()
        # training
        for epoch in tqdm(list(range(n_epochs))):
            running_loss = 0.0
            for X, y in train_loader:
                # X, y = X.to(device), y.to(device)
                loss = svgp.loss(X, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() 
            loss_avg = running_loss / len(train_loader)
            losses.append(loss_avg)
        # evaluation
        model = svgp.network
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        tr_loss, tr_acc = evaluate(model, train_loader, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        logging.info(f"train loss:\t{tr_loss}")
        logging.info(f"train acc.:\t{tr_acc}")
        logging.info(f"test loss:\t{te_loss}")
        logging.info(f"test acc.:\t{te_acc}")
        metrics = {'test_loss': te_loss, 'test_acc': te_acc,
                   'train_loss': tr_loss, 'train_acc': tr_acc}

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'losses': losses, 'metrics': metrics, 'delta': delta}
        torch.save(state, os.path.join(res_dir, fname.format(delta=delta)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    models = ['CNN', 'AllCNN', 'MLP']
    parser.add_argument('-d', '--dataset', help='dataset', choices=datasets)
    parser.add_argument('-m', '--model', help='which model to train', choices=models)
    parser.add_argument('-s', '--seed', help='randomness seed', default=117, type=int)
    parser.add_argument('--n_epochs', help='epochs training neural network', default=500, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', help='neural network learning rate', default=1e-3, type=float)
    parser.add_argument('--n_deltas', help='number of deltas to try', default=16, type=int)
    parser.add_argument('--logd_min', help='min log delta', default=-2.0, type=float)
    parser.add_argument('--logd_max', help='max log delta', default=3.0, type=float)
    parser.add_argument("--res_folder", help="Result folder", default="test")
    parser.add_argument('--double', help='double precision', action='store_true')
    parser.add_argument('--root_dir', help='Root directory', default='../')
    args = parser.parse_args()
    dataset = args.dataset
    double = args.double
    model_name = args.model
    seed = args.seed
    n_epochs = args.n_epochs
    lr = args.lr
    batch_size = args.batch_size
    n_deltas = args.n_deltas
    logd_min, logd_max = args.logd_min, args.logd_max
    root_dir = args.root_dir
    res_folder = args.res_folder

    data_dir = os.path.join(root_dir, 'data')
    res_dir = os.path.join(root_dir, 'experiments', 'results', dataset, res_folder)

    print(f'Writing results to {res_dir}')
    print(f'Reading data from {data_dir}')
    print(f'Model name: {model_name}')
    print(f'Dataset: {dataset}')
    print(f'Seed: {seed}')

    if double:
        torch.set_default_dtype(torch.double)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_train, ds_test = get_dataset(dataset, double, data_dir, device=device)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        print(res_dir)
    
    if not os.path.exists(os.path.join(res_dir, 'models')):
        os.makedirs(os.path.join(res_dir, 'models'))


    # naming convention: dataset_model_seed_delta
    fname = 'models/' + '_'.join([dataset, model_name, str(seed)]) + '_{delta:.1e}.pt'
    deltas = np.logspace(logd_min, logd_max, n_deltas)
 #   deltas = np.insert(deltas, 0, 0)  # add unregularized network
    print(f'Train set size: {len(ds_train)}')
    deltas = deltas / len(ds_train)

    main_new(ds_train, ds_test, model_name, seed, n_epochs, batch_size, lr, deltas, device, fname, res_dir)
