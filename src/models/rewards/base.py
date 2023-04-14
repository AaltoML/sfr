#!/usr/bin/env python3
import logging
from typing import Any, Callable, NamedTuple, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from custom_types import Action, RewardPrediction, State
from torch.utils.data import DataLoader


class RewardModel(NamedTuple):
    predict: Callable[[State, Action], RewardPrediction]
    train: Callable[[DataLoader], dict]
