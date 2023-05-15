#!/usr/bin/env python3
import logging
import random
from collections import deque, namedtuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from src.rl.custom_types import Action, State


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayBuffer(object):
    def __init__(self, capacity: int, batch_size: int, device):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state: State, action: Action, next_state: State, reward):
        """Save a transition"""
        self.memory.append(
            Transition(state=state, action=action, next_state=next_state, reward=reward)
        )
        # self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        samples = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        for sample in samples:
            # print("sample {}".format(sample))
            states.append(torch.Tensor(sample.state).to(self.device))
            actions.append(torch.Tensor(sample.action).to(self.device))
            rewards.append(torch.Tensor(sample.reward).to(self.device))
            next_states.append(torch.Tensor(sample.next_state))
        states = torch.stack(states, 0).to(self.device)
        actions = torch.stack(actions, 0).to(self.device)
        rewards = torch.stack(rewards, 0).to(self.device)
        next_states = torch.stack(next_states, 0).to(self.device)
        return {
            "state": states,
            "action": actions,
            "reward": rewards,
            "next_state": next_states,
        }
        # return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
