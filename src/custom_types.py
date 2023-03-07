#!/usr/bin/env python3
from collections import deque, namedtuple
from typing import Any, Iterator, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as td
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import IterableDataset
from torchtyping import TensorType


Tensor = Any
# Observation = Tensor
State = Tensor
Action = Tensor

ActionSequence = Any
Prediction = Any

ActionTrajectory = TensorType["horizon", "action_dim"]


class Prediction(NamedTuple):
    latent_dist: td.Distribution  # p(f_{\theta}(s, a) \mid (s, a), \mathcal{D})
    output_dist: td.Distribution  # p(\Delta s' \mid (s, a), \mathcal{D})
    noise_var: Optional[
        Union[td.Distribution, TensorType[""]]
    ] = 0.0  # p(\Delta s' \mid f(s, a), \Sigma_n(s, a))
    reward_dist: td.Distribution = 0.0  # p(r_t \mid (s_t, a_t), \mathcal{D})


# def ReplayBuffer_to_dynamics_DataLoader(
#     replay_buffer: ReplayBuffer,
#     batch_size: int = 64,
#     shuffle=True,
#     delta_state: Optional[bool] = True,
#     num_workers: Optional[int] = 1,
# ):
#     dataset = ReplayBuffer_to_dynamics_TensorDataset(
#         replay_buffer=replay_buffer, delta_state=delta_state
#     )
#     train_loader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
#     )
#     return train_loader


# def ReplayBuffer_to_dynamics_TensorDataset(
#     replay_buffer, delta_state: Optional[bool] = True
# ) -> TensorDataset:
#     transitions = replay_buffer.sample(len(replay_buffer))
#     observation_action_inputs = torch.concat(
#         [transitions.observations, transitions.actions], -1
#     )
#     if delta_state:
#         delta_observations = transitions.next_observations - transitions.observations
#         dataset = (observation_action_inputs, delta_observations)
#     else:
#         dataset = (observation_action_inputs, transitions.next_observations)
#     dataset = TensorDataset(*dataset)
#     return dataset
