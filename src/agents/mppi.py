#!/usr/bin/env python3
import logging
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src.agents.utils as util
import torch
import wandb
from custom_types import State
from src.models import RewardModel, TransitionModel
from torch.utils.data import DataLoader

from .agent import Agent
from .ddpg import Actor, Critic, update_actor, update_q


def init(
    transition_model: TransitionModel,
    reward_model: RewardModel,
    state_dim: int,
    action_dim: int,
    mlp_dims: List[int] = [512, 512],
    learning_rate: float = 3e-4,
    num_iterations: int = 100,
    # std_schedule: str = "linear(1.0, 0.1, 50)",
    std_clip: float = 0.3,
    nstep: int = 1,
    gamma: float = 0.99,
    tau: float = 0.005,
    horizon: int = 5,
    num_samples: int = 512,
    mixture_coef: float = 0.05,
    num_topk: int = 64,
    temperature: int = 0.5,
    momentum: float = 0.1,
    device: str = "cuda",
) -> Agent:
    actor = Actor(state_dim, mlp_dims, action_dim).to(device)
    critic = Critic(state_dim=state_dim, mlp_dims=mlp_dims, action_dim=action_dim).to(
        device
    )
    critic_target = Critic(
        state_dim=state_dim, mlp_dims=mlp_dims, action_dim=action_dim
    ).to(device)

    # Init optimizer
    optim_actor = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=learning_rate)

    ddpg_agent = src.agents.ddpg.init_from_actor_critic(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        optim_actor=optim_actor,
        optim_critic=optim_critic,
        num_iterations=num_iterations,
        std_clip=std_clip,
        nstep=nstep,
        gamma=gamma,
        tau=tau,
        horizon=horizon,
        num_samples=num_samples,
        mixture_coef=mixture_coef,
        num_topk=num_topk,
        temperature=temperature,
        momentum=momentum,
        device=device,
    )

    # def train_fn(data_loader: DataLoader):
    def train_fn(replay_iter) -> dict:
        std = 0.1
        info = {"std": std}
        # data_iter = iter(data_loader)
        # for batch in data_loader:
        for i in range(num_iterations):
            # i = epoch * num_iterations + batch_idx
            # for i, batch in enumerate(data_loader):
            # std = linear_schedule(std_schedule, i)  # linearly udpate std
            # info = {"std": std}
            std = 0.1
            info = {"std": std}

            # std_schedule, (num_iterations - 10) * episode_idx + i

            batch = next(replay_iter)
            state, action, reward, discount, next_state = util.to_torch(
                batch, device, dtype=torch.float32
            )
            # swap the batch and horizon dimension -> [H, B, _shape]
            action, reward, discount, next_state = (
                torch.swapaxes(action, 0, 1),
                torch.swapaxes(reward, 0, 1),
                torch.swapaxes(discount, 0, 1),
                torch.swapaxes(next_state, 0, 1),
            )

            # form n-step samples
            _reward, _discount = 0, 1
            for t in range(nstep):
                _reward += _discount * reward[t]
                _discount *= gamma

            info.update(
                update_q_fn(state, action[0], _reward, _discount, next_state, std)
            )
            info.update(update_actor_fn(state, std))

            # Update target network
            util.soft_update_params(critic, critic_target, tau=tau)
            # if i % 10 == 0:
            #     print(
            #         "DDPG: Iteration {} | Q Loss {} | Actor Loss {}".format(
            #             i, info["q_loss"], info["actor_loss"]
            #         )
            #     )
            # if early_stop(loss_value):
            #     print("SAC loss early stop criteria met, exiting training loop")
            #     break
            wandb.log(info)

        return info

    @torch.no_grad()
    def select_action_fn(state, eval_mode: bool = False, t0=True):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # sample policy trajectories
        num_pi_trajs = int(mixture_coef) * num_samples
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, action_dim, device=device)
            state = state.repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = actor(state, std).sample()
                state, _ = transition_model(state, pi_actions[t])

        # Initialize state and parameters
        state = state.repeat(num_samples + num_pi_trajs, 1)
        mean = torch.zeros(horizon, action_dim, device=device)
        std = 2 * torch.ones(horizon, action_dim, device=device)
        # TODO implememnt prev_mean
        # if not t0 and hasattr(self, "_prev_mean"):
        #     mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(num_iterations):
            logger.info("MPPI iteration: {}".format(i))
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(
                    horizon,
                    num_samples,
                    action_dim,
                    device=std.device,
                ),
                -1,
                1,
            )
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = estimate_value(state, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), num_topk, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9
            )
            _std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
                / (score.sum(0) + 1e-9)
            )
            _std = _std.clamp_(0.1, 2)
            mean, std = (momentum * mean + (1 - momentum) * _mean, _std)

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        # self._prev_mean = mean TODO implement prev_mean
        mean, std = actions[0], _std[0]
        action = mean
        if not eval_mode:
            action += std * torch.randn(action_dim, device=std.device)
        return action

    return Agent(select_action=select_action_fn, train=train_fn)


def estimate_value(
    state: State,
    actions,
    actor: Actor,
    critic: Critic,
    transition_model,
    reward_model,
    horizon: int,
    gamma: float,
    std: float,
    std_clip: float,
):
    """Estimate value of a trajectory starting at state and executing given actions."""
    G, discount = 0, 1
    print("state.shape {}".format(state.shape))
    print("actions.shape {}".format(actions.shape))
    for t in range(horizon):
        transition = transition_model(state, actions[t])
        reward = reward_model(state, actions[t])
        print("reward at t={}: ".format(t, reward))
        # print("state")
        # print(state.shape)
        G += discount * reward
        discount *= gamma
    G += discount * torch.min(*critic(state, actor(state, std).sample(clip=std_clip)))
    return G
