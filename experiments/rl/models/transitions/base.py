#!/usr/bin/env python3
import abc
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from experiments.rl.custom_types import Action, Data, State, StatePrediction
from torch.utils.data import DataLoader


class TransitionModel:
    @abc.abstractmethod
    def predict(self, state: State, action: Action) -> StatePrediction:
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, data: DataLoader):
        # TODO this should be Tuple[InputData,OutputData]
        raise NotImplementedError

    def update(self, data_new: Data):
        pass
