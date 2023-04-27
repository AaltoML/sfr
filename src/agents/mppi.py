#!/usr/bin/env python3
import logging
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src
import torch
from src.custom_types import State
from src.models import RewardModel, TransitionModel
from torchrl.data import ReplayBuffer

from .agent import Agent
from .ddpg import Actor, Critic


def init(
    transition_model: TransitionModel,
    reward_model: RewardModel,
    state_dim: int,
    action_dim: int,
    mlp_dims: List[int] = [512, 512],
    learning_rate: float = 3e-4,
    max_ddpg_iterations: int = 100,  # for training DDPG
    # std_schedule: str = "linear(1.0, 0.1, 50)",
    std: float = 0.1,  # TODO make this schedule
    std_clip: float = 0.3,
    # nstep: int = 1,
    gamma: float = 0.99,
    tau: float = 0.005,
    horizon: int = 5,
    num_mppi_iterations: int = 5,
    num_samples: int = 512,
    mixture_coef: float = 0.05,
    num_topk: int = 64,
    temperature: int = 0.5,
    momentum: float = 0.1,
    unc_prop_strategy: str = "mean",
    sample_actor: bool = True,
    bootstrap: bool = True,
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
        max_ddpg_iterations=max_ddpg_iterations,
        std_clip=std_clip,
        nstep=1,
        gamma=gamma,
        tau=tau,
        device=device,
    )
    # std = torch.Tensor([0.5], device=device)
    # std = 0.5
    std = 0.1

    estimate_value = src.agents.objectives.greedy(
        actor=actor,
        critic=critic,
        transition_model=transition_model,
        reward_model=reward_model,
        horizon=horizon,
        std=std,
        std_clip=std_clip,
        gamma=gamma,
        unc_prop_strategy=unc_prop_strategy,
        sample_actor=sample_actor,
        bootstrap=bootstrap,
    )

    def train_fn(replay_buffer: ReplayBuffer) -> dict:
        # TODO these could be run in parallel. How to do that?
        logger.info("Starting training DDPG...")
        info = ddpg_agent.train(replay_buffer)
        logger.info("Finished training DDPG")

        logger.info("Starting training transition model...")
        transition_model.train(replay_buffer)
        # info.update(transition_model.train(replay_buffer))
        logger.info("Finished training transition model")

        logger.info("Starting training reward model...")
        reward_model.train(replay_buffer)
        # info.update(reward_model.train(replay_buffer))
        logger.info("Finished training reward model")

        return info

    _prev_mean = torch.zeros(horizon, action_dim, device=device)

    @torch.no_grad()
    def select_action_fn(
        state: State,
        eval_mode: bool = False,
        t0: bool = True,
    ):
        # estimate_value_compiled = torch.compile(estimate_value)
        # if isinstance(state, np.ndarray):
        #     state = torch.from_numpy(state).to(device).float()
        # print("state: {}".format(state))
        global _prev_mean
        if isinstance(state, np.ndarray):
            # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # TODO should this be float64 or float32?
            # TODO set dynamically using type of NN params
            state = torch.tensor(state, dtype=torch.float64, device=device).unsqueeze(0)
        # print("state: {}".format(state))

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
        action_mean = torch.zeros(horizon, action_dim, device=device)
        action_std = 2 * torch.ones(horizon, action_dim, device=device)
        # TODO implememnt prev_mean
        # if not t0 and hasattr(self, "_prev_mean"):
        if not t0:
            # print("USING PREV_MEAN {}".format(_prev_mean[1:]))
            action_mean[:-1] = _prev_mean[1:]

        # Iterate CEM
        # for i in range(num_iterations):
        # print("before loop")
        for i in range(num_mppi_iterations):
            # logger.info("MPPI iteration: {}".format(i))
            actions = torch.clamp(
                action_mean.unsqueeze(1)
                + action_std.unsqueeze(1)
                * torch.randn(
                    horizon,
                    num_samples,
                    action_dim,
                    device=device,
                    # TODO what device should this be on?
                    # device=std.device,
                ),
                -1,
                1,
            )
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = estimate_value(state, actions).nan_to_num_(0)
            # value = estimate_value_compiled(state, actions).nan_to_num_(0)
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
            action_mean = momentum * action_mean + (1 - momentum) * _mean
            action_std = _std.clamp_(0.1, 2)

        # print("after loop")
        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        _prev_mean = action_mean  # TODO implement prev_mean
        mean, std = actions[0], action_std[0]
        if eval_mode:
            return mean
        else:
            return mean + std * torch.randn(
                action_dim,
                device=device,
                # TODO what device should this be on?
                # device=std.device,
            )

    def update_fn(data_new):
        state_action_input, state_diff_output, reward_output = data_new
        transition_model.update(data_new=(state_action_input, state_diff_output))
        reward_model.update(data_new=(state_action_input, reward_output))

    return Agent(select_action=select_action_fn, train=train_fn, update=update_fn)
