#!/usr/bin/env python3
import logging
import random
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from dm_control import suite
from torch.utils.data import DataLoader, TensorDataset

from .buffer import ReplayBuffer


def to_torch(xs, device, dtype=torch.float32):
    return tuple(torch.as_tensor(x, device=device, dtype=dtype) for x in xs)


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.min_validation_loss = np.inf


def set_seed_everywhere(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(cfg.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    # pl.seed_everything(random_seed)


def make_data_loader(
    replay_buffer: ReplayBuffer, batch_size: int = 64, num_workers: int = 1
):
    samples = replay_buffer.sample(len(replay_buffer))
    # print("samples")
    # print(samples)
    state = samples["observation_vector"]
    action = samples["action"]
    reward = samples["reward"]
    # print(reward.shape)
    state_action_input = torch.concat([state, action], -1)
    next_state = samples["next"]["observation_vector"]
    state_diff = next_state - state
    # state_diff_reward = torch.concat([state_diff, reward], -1)
    next_state_reward = torch.concat([next_state, reward], -1)
    # data = (state_action_input, state_diff)
    # data = (state_action_input, state_diff_reward)
    data = (state_action_input, next_state_reward)
    data = TensorDataset(*data)
    train_loader = DataLoader(
        data,
        batch_sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(data),
            batch_size=batch_size,
            drop_last=False,
            # drop_last=True,
        ),
        # batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader


# def rollout_agent_and_populate_replay_buffer(
#     env: gym.Env,
#     policy: Policy,
#     replay_buffer: ReplayBuffer,
#     num_episodes: int,
#     # rollout_horizon: Optional[int] = 1,
#     rollout_horizon: Optional[int] = None,
# ) -> ReplayBuffer:
#     logger.info(f"Collecting {num_episodes} episodes from env")

#     observation, info = env.reset()
#     for episode in range(num_episodes):
#         terminated, truncated = False, False
#         timestep = 0
#         while not terminated or truncated:
#             if rollout_horizon is not None:
#                 if timestep >= rollout_horizon:
#                     break
#             action = policy(observation=observation)
#             next_observation, reward, terminated, truncated, info = env.step(action)
#             replay_buffer.push(
#                 observation=observation,
#                 action=action,
#                 next_observation=next_observation,
#                 reward=reward,
#                 terminated=terminated,
#                 truncated=truncated,
#             )
#             observation = next_observation

#             timestep += 1

#         observation, info = env.reset()

#     return replay_buffer


def make_env(env_name, seed, action_repeat):
    """
    Make environment for TD-MPC experiments.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    domain, task = str(env_name).replace("-", "_").split("_", 1)
    domain = dict(cup="ball_in_cup").get(domain, domain)

    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain, task, task_kwargs={"random": seed}, visualize_reward=False
        )
    else:
        import os
        import sys

        sys.path.append("..")
        import custom_dmc_tasks as cdmc

        env = cdmc.make(
            domain, task, task_kwargs={"random": seed}, visualize_reward=False
        )

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = ExtendedTimeStepWrapper(env)
    env = ConcatObsWrapper(env)
    # env = TimeStepToGymWrapper(env, domain, task, action_repeat, 'state')

    return env


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # elif isinstance(m, EnsembleLinear):
    #     for w in m.weight.data:
    #         nn.init.orthogonal_(w)
    #     if m.bias is not None:
    #         for b in m.bias.data:
    #             nn.init.zeros_(b)
    elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        # nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
