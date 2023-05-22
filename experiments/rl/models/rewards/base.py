#!/usr/bin/env python3
import abc
import logging
from typing import Any, Callable, NamedTuple, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from experiments.rl.custom_types import Action, Data, RewardPrediction, State
from torch.utils.data import DataLoader


# class RewardModel(NamedTuple):
#     predict: Callable[[State, Action], RewardPrediction]
#     train: Callable[[DataLoader], dict]
#     update: Callable[[Data], None]


class RewardModel:
    @abc.abstractmethod
    def predict(self, state: State, action: Action) -> RewardPrediction:
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, data: DataLoader):
        # TODO this should be Tuple[InputData,OutputData]
        raise NotImplementedError

    def update(self, data_new: Data):
        raise NotImplementedError
