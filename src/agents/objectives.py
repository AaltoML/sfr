#!/usr/bin/env python3
import torch
import torch.distributions as td
from src.custom_types import ActionTrajectory, State, StateTrajectory
from src.models import RewardModel, TransitionModel
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
        def rollout(
            start_state: State, actions: ActionTrajectory, data_new=None
        ) -> StateTrajectory:
            state = start_state
            state_trajectory = state[None, ...]
            for t in range(horizon):
                state = transition_model.predict(
                    state, actions[t], data_new=data_new
                ).state_mean
                state_trajectory = torch.concatenate(
                    [state_trajectory, state[None, ...]], 0
                )
            return state_trajectory

    elif unc_prop_strategy == "sample":

        def rollout(start_state: State, actions: ActionTrajectory, data_new=None):
            state = start_state
            state_trajectory = state[None, ...]
            for t in range(horizon):
                next_state_prediction = transition_model.predict(
                    state, actions[t], data_new=data_new
                )
                state_dist = td.Normal(
                    next_state_prediction.state_mean,
                    next_state_prediction.state_var + next_state_prediction.noise_var,
                )
                state = state_dist.sample()
                state_trajectory = torch.concatenate(
                    [state_trajectory, state[None, ...]], 0
                )
            return state_trajectory

    def greedy_fn(
        start_state: State, actions: ActionTrajectory, data_new=None
    ) -> TensorType[""]:
        """Estimate value of a trajectory starting at state and executing given actions."""
        state = start_state
        # print("start_state: {}".format(start_state.shape))
        G, discount = 0, 1
        state_trajectory = rollout(
            start_state=start_state, actions=actions, data_new=data_new
        )
        # print("state_trajectory: {}".format(state_trajectory.shape))
        # next_state_prediction = transition_model.predict(state, actions[t])
        # reward_prediction = reward_model.predict(state_trajectory, actions).reward_mean
        for t in range(horizon):
            G += (
                discount
                * reward_model.predict(
                    state_trajectory[t], actions[t], data_new=data_new
                ).reward_mean
            )
            discount *= gamma

        final_action_dist = actor(state, std)
        if sample_actor:
            final_action = final_action_dist.sample(clip=std_clip)
        else:
            final_action = final_action_dist.mean
        G_final = discount * torch.min(*critic(state, final_action))
        return G[..., None] + G_final

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
