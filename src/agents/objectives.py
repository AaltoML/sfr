#!/usr/bin/env python3
import torch
from src.custom_types import State
from src.models import RewardModel, TransitionModel

from .ddpg import Actor, Critic


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
        # print("state.shape {}".format(start_state.shape))
        # print("actions.shape {}".format(actions.shape))
        for t in range(horizon):
            # TODO this isn't integrating uncertainty
            next_state_prediction = transition_model.predict(state, actions[t])
            # print("next_state_prediction")
            # print(next_state_prediction.state_mean.shape)
            reward_prediction = reward_model.predict(state, actions[t])
            # print("reward at t={}: ".format(t, reward_prediction.reward_mean.shape))
            # print(reward_prediction.reward_mean.shape)
            # print("state")
            # print(state.shape)
            G += discount * reward_prediction.reward_mean
            # print("G {}".format(G.shape))
            discount *= gamma
            state = next_state_prediction.state_mean
        # print("final state {}".format(state.shape))
        G_final = discount * torch.min(
            *critic(
                state,
                actor(state, std).sample(clip=std_clip)
                # state[None, ...], actor(state[None, ...], std).sample(clip=std_clip)
            )
        )
        # print("G_final {}".format(G_final.shape))
        return G[..., None] + G_final

    return greedy_fn
