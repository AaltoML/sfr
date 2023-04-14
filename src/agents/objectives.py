#!/usr/bin/env python3
from src.custom_types import State
from src.models import RewardModel, TransitionModel
import torch


def greedy(
    actor: Actor,
    critic: Critic,
    transition_model: TransitionModel,
    reward_model: RewardModel,
    horizon: int = 5,
    std: float = 0.1,
    std_clip: float = 0.3,
    gamma: float = 0.99,
):
    def greedy_fn(start_state: State, actions):
        """Estimate value of a trajectory starting at state and executing given actions."""
        state = start_state
        G, discount = 0, 1
        print("state.shape {}".format(start_state.shape))
        print("actions.shape {}".format(actions.shape))
        for t in range(horizon):
            next_state = transition_model.predict(state, actions[t])
            reward = reward_model.predict(state, actions[t])
            print("reward at t={}: ".format(t, reward))
            # print("state")
            # print(state.shape)
            G += discount * reward
            discount *= gamma
            state = next_state["state_mean"]
        G += discount * torch.min(
            *critic(state, actor(state, std).sample(clip=std_clip))
        )
        return G

    return greedy_fn
