#!/usr/bin/env python3
import logging
from typing import Any, Callable, NamedTuple, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim
from custom_types import (
    Action,
    DeltaStateMean,
    DeltaStateVar,
    NoiseVar,
    Prediction,
    State,
    StateMean,
    StateVar,
)
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec, ReplayBuffer, UnboundedContinuousTensorSpec
from torchrl.envs import ModelBasedEnvBase
from torchrl.modules import SafeModule, WorldModelWrapper
from torchtyping import TensorType


class TransitionModel(nn.Module):
    predict: Callable[
        [InputData, Optional[Data]], Tuple[DeltaStateMean, DeltaStateVar, NoiseVar]
    ]
    train: Callable[[DataLoader], TensorType[float, ""]]

    def __call__(
        self,
        state_mean: StateMean,
        state_var: StateVar,
        noise_var: NoiseVar,
        action: Action,
    ) -> Tuple[StateMean, StateVar, NoiseVar]:
        """Returns next state mean/var and noise var"""

    def forward(
        self, x, data_new: Optional = None
    ) -> Tuple[DeltaStateMean, DeltaStateVar, NoiseVar]:
        """Returns change in state"""
        raise NotImplementedError

    def train(self, replay_buffer: ReplayBuffer):
        raise NotImplementedError
