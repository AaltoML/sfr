#!/usr/bin/env python3
from typing import Callable, NamedTuple, Optional, Any

from src.custom_types import Action, EvalMode, State, T0
from torch.utils.data import DataLoader

Data = Any


class Agent(NamedTuple):
    select_action: Callable[[State, EvalMode, T0], Action]
    train: Callable[[DataLoader], Optional[dict]]
    update: Callable[[Data], None]
