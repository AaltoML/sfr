from argparse import Namespace
from typing import Tuple
import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
# from backbone.CifarNet import CifarNet

def base_path():
    return "/tmp/datasets/"

# from utils.conf import base_path_dataset as base_path
# from datasets.utils.continual_dataset import (ContinualDataset, 
#                                               store_masked_loaders)

# class SequentialCIFAR(ContinualDataset):
#     NAME = "seq-cifar"
#     SETTING = "class-il"
#     N_TASKS = 6
#     N_CLASSES = 10 * N_TASKS
#     TRANSFORM = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(), 
#         transforms.ToTensor()
#         # Note: no normalization here
#     ])
    
#     def __init__(self, args: Namespace) -> None:
#         super().__init__(args)

#         base_transform = transforms.ToTensor()

#         train_cifar10 = CIFAR10(base_path() + "CIFAR10", train=True, 
#                                 download=True, transform=base_transform)
#         test_cifar10 = CIFAR10(base_path() + "CIFAR10", train=False,
#                                download=True, transform=base_transform)

#         train_cifar100 = CIFAR100(base_path() + "CIFAR100", train=True,
#                                   download=True, transform=base_transform)
#         test_cifar100 = CIFAR100(base_path() + "CIFAR100", train=False,
#                                  download=True, transform=base_transform)
        
#         train_cifar10_loader = DataLoader(train_cifar10, batch_size=len(train_cifar10))
#         train_cifar10_data, train_cifar10_targets = next(iter(train_cifar10_loader))
#         test_cifar10_loader = DataLoader(test_cifar10, batch_size=len(test_cifar10))
#         test_cifar10_data, test_cifar10_targets = next(iter(test_cifar10_loader))

#         train_cifar100_loader = DataLoader(train_cifar100, batch_size=len(train_cifar100))
#         train_cifar100_data, train_cifar100_targets = next(iter(train_cifar100_loader))
#         train_cifar100_targets += 10
#         test_cifar100_loader = DataLoader(test_cifar100, batch_size=len(test_cifar100))
#         test_cifar100_dataa, test_cifar100_targets = next(iter(test_cifar100_loader))
#         test_cifar100_targets += 10

#         print()


#     def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
#         transform = self.TRANSFORM
#         test_transform = transforms.ToTensor()
        


if __name__ == "__main__":
    # seq_cifar = SequentialCIFAR()
    base_transform = transforms.ToTensor()

    train_cifar10 = CIFAR10(base_path() + "CIFAR10", train=True, 
                            download=True, transform=base_transform)
    test_cifar10 = CIFAR10(base_path() + "CIFAR10", train=False,
                            download=True, transform=base_transform)

    train_cifar100 = CIFAR100(base_path() + "CIFAR100", train=True,
                                download=True, transform=base_transform)
    test_cifar100 = CIFAR100(base_path() + "CIFAR100", train=False,
                                download=True, transform=base_transform)
    
    train_cifar10_loader = DataLoader(train_cifar10, batch_size=len(train_cifar10))
    train_cifar10_data, train_cifar10_targets = next(iter(train_cifar10_loader))
    test_cifar10_loader = DataLoader(test_cifar10, batch_size=len(test_cifar10))
    test_cifar10_data, test_cifar10_targets = next(iter(test_cifar10_loader))

    train_cifar100_loader = DataLoader(train_cifar100, batch_size=len(train_cifar100))
    train_cifar100_data, train_cifar100_targets = next(iter(train_cifar100_loader))
    train_cifar100_targets += 10
    test_cifar100_loader = DataLoader(test_cifar100, batch_size=len(test_cifar100))
    test_cifar100_data, test_cifar100_targets = next(iter(test_cifar100_loader))
    test_cifar100_targets += 10

    train_cifar_data = torch.vstack((train_cifar10_data, train_cifar100_data))
    test_cifar_data = torch.vstack/((test_cifar10_data, test_cifar100_data))
    print()
