from argparse import Namespace
from typing import Callable, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy

from typing import Any, Tuple
import os
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset
from backbone.CifarNet import CifarNet

from utils.conf import base_path_dataset as base_path
from datasets.utils.continual_dataset import (ContinualDataset, 
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val


class MyCIFAR(VisionDataset):
    def __init__(self, root: str, 
                 transform: Callable[..., Any] | None = None, 
                 target_transform: Callable[..., Any] | None = None,
                 train: bool = True,
                 download: bool = True, 
                 ) -> None:
        super().__init__(root, transform, target_transform)
        self.train = train
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        cifar10 = CIFAR10(os.path.join(root, "CIFAR10"), 
                           train=train,
                           download=download,
                           transform=self.not_aug_transform)
        cifar100 = CIFAR100(os.path.join(root, "CIFAR100"), 
                             train=train,
                             download=download, 
                             transform=self.not_aug_transform)
        
        cifar10_loader = DataLoader(cifar10, batch_size=len(cifar10), shuffle=False)
        cifar10_data, cifar10_targets = next(iter(cifar10_loader))
        cifar100_loader = DataLoader(cifar100, batch_size=len(cifar100), shuffle=False)
        cifar100_data, cifar100_targets = next(iter(cifar100_loader))
        cifar100_targets += 10

        self.data = torch.vstack((cifar10_data, cifar100_data))
        self.targets = torch.concatenate((cifar10_targets, cifar100_targets))
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        if not self.train:
            return img, target

        if self.transform is not None:
            aug_img = self.transform(img)
        else:
            aug_img = img.clone()
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # if hasattr(self, 'logits'):
        #     return aug_img, target, img, self.logits[index]
        
        return aug_img, target, img

    def __len__(self):
        return len(self.targets)


class SequentialCIFAR(ContinualDataset):
    NAME = "seq-cifar"
    SETTING = "class-il"
    N_TASKS = 6
    N_CLASSES = 10 * N_TASKS
    N_CLASSES_PER_TASK = 10
    TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        # transforms.ToTensor()
    ])
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.train_dataset = MyCIFAR(base_path(), train=True, download=True)
        self.test_dataset = MyCIFAR(base_path(), train=False, download=True)


    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        transform = self.TRANSFORM
        test_transform = transforms.ToTensor()

        if self.args.validation: 
            train_dataset, test_dataset = get_train_val(self.train_dataset,
                                                        test_transform, 
                                                        self.NAME)
            train, test = store_masked_loaders(
                train_dataset,
                test_dataset,
                self)
        else:
            train, test = store_masked_loaders(
                deepcopy(self.train_dataset),
                deepcopy(self.test_dataset),
                self)
        return train, test
    
    @staticmethod
    def get_transform():
        return SequentialCIFAR.TRANSFORM
    
    def get_backbone(self):
        return CifarNet(n_classes=self.N_CLASSES)
    
    @staticmethod
    def get_loss():
        return F.cross_entropy
    
    @staticmethod
    def get_normalization_transform():
        return nn.Identity()
    
    @staticmethod
    def get_denormalization_transform():
        return nn.Identity()
    
    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR.get_batch_size()


if __name__ == "__main__":
    # seq_cifar = SequentialCIFAR()
    base_transform = transforms.ToTensor()
    root = base_path()
    # train_cifar10 = CIFAR10(root + "CIFAR10", train=True, 
    #                         download=True, transform=base_transform)
    # test_cifar10 = CIFAR10(root + "CIFAR10", train=False,
    #                         download=True, transform=base_transform)

    # train_cifar100 = CIFAR100(root + "CIFAR100", train=True,
    #                             download=True, transform=base_transform)
    # test_cifar100 = CIFAR100(root + "CIFAR100", train=False,
    #                             download=True, transform=base_transform)
    
    # train_cifar10_loader = DataLoader(train_cifar10, batch_size=len(train_cifar10))
    # train_cifar10_data, train_cifar10_targets = next(iter(train_cifar10_loader))
    # test_cifar10_loader = DataLoader(test_cifar10, batch_size=len(test_cifar10))
    # test_cifar10_data, test_cifar10_targets = next(iter(test_cifar10_loader))

    # train_cifar100_loader = DataLoader(train_cifar100, batch_size=len(train_cifar100))
    # train_cifar100_data, train_cifar100_targets = next(iter(train_cifar100_loader))
    # train_cifar100_targets += 10
    # test_cifar100_loader = DataLoader(test_cifar100, batch_size=len(test_cifar100))
    # test_cifar100_data, test_cifar100_targets = next(iter(test_cifar100_loader))
    # test_cifar100_targets += 10

    # train_cifar_data = torch.vstack((train_cifar10_data, train_cifar100_data))
    # test_cifar_data = torch.vstack((test_cifar10_data, test_cifar100_data))

    # train_cifar_targets = torch.concatenate((train_cifar10_targets, train_cifar100_targets))
    # test_cifar_targets = torch.concatenate((test_cifar10_targets, test_cifar100_targets))
    
    train_dataset = MyCIFAR(root, download=True, train=True) 
    test_dataset = MyCIFAR(root, download=True, train=False)

    print(f"{train_dataset.data.shape}")
    print(f"{train_dataset.targets.shape}")
    print(f"{test_dataset.data.shape}")
    print(f"{test_dataset.targets.shape}")