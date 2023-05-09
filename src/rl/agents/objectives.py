#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.distributions as td
from src.rl.custom_types import ActionTrajectory, State, StateTrajectory
from src.rl.models import RewardModel, TransitionModel
from torchtyping import TensorType

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
    unc_prop_strategy: str = "mean",
    sample_actor: bool = True,
    bootstrap: bool = True,
):
    if unc_prop_strategy == "mean":
        # def greedy_fn(start_state: State, actions: ActionTrajectory) -> TensorType[""]:
        #     """Estimate value of a trajectory starting at state and executing given actions."""
        #     state = start_state
        #     G, discount = 0, 1
        #     for t in range(horizon):
        #         next_state_prediction = transition_model.predict(state, actions[t])
        #         reward_prediction = reward_model.predict(state, actions[t])
        #         G += discount * reward_prediction.reward_mean
        #         discount *= gamma
        #         state = next_state_prediction.state_mean
        #     final_action_dist = actor(state, std)
        #     if sample_actor:
        #         final_action = final_action_dist.sample(clip=std_clip)
        #     else:
        #         final_action = final_action_dist.mean
        #     G_final = discount * torch.min(*critic(state, final_action))
        #     return G[..., None] + G_final
        def rollout(start_state: State, actions: ActionTrajectory) -> StateTrajectory:
            state = start_state
            state_trajectory = state[None, ...]
            for t in range(horizon):
                state = transition_model.predict(state, actions[t]).state_mean
                state_trajectory = torch.concatenate(
                    [state_trajectory, state[None, ...]], 0
                )
            return state_trajectory

    elif unc_prop_strategy == "sample":

        def rollout(start_state: State, actions: ActionTrajectory) -> StateTrajectory:
            state = start_state
            state_trajectory = state[None, ...]
            for t in range(horizon):
                next_state_prediction = transition_model.predict(state, actions[t])
                state_dist = td.Normal(
                    next_state_prediction.state_mean,
                    next_state_prediction.state_var + next_state_prediction.noise_var,
                )
                state = state_dist.sample()
                # TODO draw more than one sample?
                # print("sample state {}".format(state.shape))
                state_trajectory = torch.concatenate(
                    [state_trajectory, state[None, ...]], 0
                )
            return state_trajectory

    def greedy_fn(start_state: State, actions: ActionTrajectory) -> TensorType[""]:
        """Estimate value of a trajectory starting at state and executing given actions."""
        # state = start_state
        # print("start_state: {}".format(start_state.shape))
        G, discount = 0, 1
        state_trajectory = rollout(start_state=start_state, actions=actions)
        # print("state_trajectory: {}".format(state_trajectory.shape))
        # next_state_prediction = transition_model.predict(state, actions[t])
        # reward_prediction = reward_model.predict(state_trajectory, actions).reward_mean
        # if data_new["reward"] is not None:
        #     try:
        #         # print("data_new[reward] {}".format(data_new["reward"].shape))
        #         print("data_new[reward] {}".format(type(data_new["reward"])))
        #         print(
        #             "data_new[reward] {} {}".format(
        #                 data_new["reward"][0].shape, data_new["reward"][1].shape
        #             )
        #         )
        #     except:
        #         print("data_new[reward] {}".format("none"))
        # else:
        #     print("data_new[reward] {}".format("none"))
        # reward_model.dual_update(data_new=data_new["reward"])
        for t in range(horizon):
            G += (
                discount
                * reward_model.predict(state_trajectory[t], actions[t]).reward_mean
            )
            discount *= gamma

        # print("state_trajectory[-1, :] {}".format(state_trajectory[-1, :].shape))
        # reward = reward_model.predict(state_trajectory[t], actions[t]).reward_mean
        # print("reward_mean {}".format(reward.shape))
        # print("G {}".format(G.shape))
        final_action_dist = actor(state_trajectory[-1, :], std)
        if sample_actor:
            final_action = final_action_dist.sample(clip=std_clip)
        else:
            final_action = final_action_dist.mean
        G_final = discount * torch.min(*critic(state_trajectory[-1, :], final_action))
        # print("G_ginal {}".format(G_final.shape))
        # logger.info("discount {}".format(discount))
        # logger.info("G 0:H {}".format(torch.mean(G, 0)))
        # logger.info("G bootstrap {}".format(torch.mean(G_final, 0)))
        if bootstrap:
            return G[..., None] + G_final
        else:
            return G[..., None]

    return greedy_fn


# def greedy(
#     actor: Actor,
#     critic: Critic,
#     transition_model: TransitionModel,
#     reward_model: RewardModel,
#     horizon: int = 5,
#     std: float = 0.1,
#     std_clip: float = 0.3,
#     gamma: float = 0.99,
# ):
#     def greedy_fn(start_state: State, actions):
#         """Estimate value of a trajectory starting at state and executing given actions."""
#         state = start_state
#         G, discount = 0, 1
#         for t in range(horizon):
#             next_state_prediction = transition_model.predict(state, actions[t])
#             reward_prediction = reward_model.predict(state, actions[t])
#             G += discount * reward_prediction.reward_mean
#             discount *= gamma
#             state = next_state_prediction.state_mean
#         G_final = discount * torch.min(
#             *critic(state, actor(state, std).sample(clip=std_clip))
#         )
#         return G[..., None] + G_final

#     return greedy_fn


# def greedy(
#     actor: Actor,
#     critic: Critic,
#     transition_model: TransitionModel,
#     reward_model: RewardModel,
#     horizon: int = 5,
#     std: float = 0.1,
#     std_clip: float = 0.3,
#     gamma: float = 0.99,
# ):
#     def greedy_fn(start_state: State, actions):
#         """Estimate value of a trajectory starting at state and executing given actions."""
#         state = start_state
#         G, discount = 0, 1
#         # print("state.shape {}".format(start_state.shape))
#         # print("actions.shape {}".format(actions.shape))
#         for t in range(horizon):
#             # TODO this isn't integrating uncertainty
#             next_state_prediction = transition_model.predict(state, actions[t])
#             # print("next_state_prediction")
#             # print(next_state_prediction.state_mean.shape)
#             reward_prediction = reward_model.predict(state, actions[t])
#             # print("reward at t={}: ".format(t, reward_prediction.reward_mean.shape))
#             # print(reward_prediction.reward_mean.shape)
#             # print("state")
#             # print(state.shape)
#             G += discount * reward_prediction.reward_mean
#             # print("G {}".format(G.shape))
#             discount *= gamma
#             state = next_state_prediction.state_mean
#         # print("final state {}".format(state.shape))
#         G_final = discount * torch.min(
#             *critic(
#                 state,
#                 actor(state, std).sample(clip=std_clip)
#                 # state[None, ...], actor(state[None, ...], std).sample(clip=std_clip)
#             )
#         )
#         # print("G_final {}".format(G_final.shape))
#         return G[..., None] + G_final

#     return greedy_fn
