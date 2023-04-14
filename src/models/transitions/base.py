#!/usr/bin/env python3
import logging
from typing import Callable, NamedTuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from custom_types import Action, State, StatePrediction
from torch.utils.data import DataLoader


class TransitionModel(NamedTuple):
    predict: Callable[[State, Action], StatePrediction]
    train: Callable[[DataLoader], dict]
