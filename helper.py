#!/usr/bin/env python3
from collections import OrderedDict

import numpy as np
import torch.nn as nn


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val = np.inf

    def __call__(self, val):
        if val < self.min_val:
            self.min_val = val
            self.counter = 0
        elif val > (self.min_val + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CIFAR10Net(nn.Module):
    def __init__(self, in_channels: int = 3, n_out: int = 10, use_tanh: bool = False):
        super().__init__()
        self.output_size = n_out
        activ = nn.Tanh if use_tanh else nn.ReLU

        self.cnn_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=64,
                            kernel_size=(5, 5),
                            stride=(1, 1),
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "maxpool1",
                        nn.Sequential(
                            nn.ZeroPad2d((0, 1, 0, 1)),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        ),
                    ),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=96,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    (
                        "maxpool2",
                        nn.Sequential(
                            nn.ZeroPad2d((0, 1, 0, 1)),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        ),
                    ),
                    (
                        "conv3",
                        nn.Conv2d(
                            in_channels=96,
                            out_channels=128,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                    ),
                    ("relu3", nn.ReLU()),
                    (
                        "maxpool3",
                        nn.Sequential(
                            nn.ZeroPad2d((1, 1, 1, 1)),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        ),
                    ),
                ]
            )
        )
        self.lin_block = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("dense1", nn.Linear(in_features=3 * 3 * 128, out_features=512)),
                    ("activ1", activ()),
                    ("dense2", nn.Linear(in_features=512, out_features=256)),
                    ("activ2", activ()),
                    (
                        "dense3",
                        nn.Linear(in_features=256, out_features=self.output_size),
                    ),
                ]
            )
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.cnn_block(x)
        out = self.lin_block(x)
        return out
