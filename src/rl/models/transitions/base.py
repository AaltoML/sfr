#!/usr/bin/env python3
import logging
import abc
from typing import Callable, NamedTuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.rl.custom_types import Action, StatePrediction, State, Data
from torch.utils.data import DataLoader


# class TransitionModel(NamedTuple):
#     predict: Callable[[State, Action], StatePrediction]
#     train: Callable[[DataLoader], dict]
#     update: Callable[[Data], None]


class TransitionModel:
    @abc.abstractmethod
    def predict(self, state: State, action: Action) -> StatePrediction:
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, data: DataLoader):
        raise NotImplementedError

    def update(self, data: Data):
        pass
