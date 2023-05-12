#!/usr/bin/env python3
import logging
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src
import torch
from src.rl.custom_types import State, Action, ActionTrajectory
from src.rl.models import RewardModel, TransitionModel
from torchrl.data import ReplayBuffer

from .agent import Agent
from .ddpg import Actor, Critic, DDPGAgent


class MPPIAgent(Agent):
    def __init__(
        self,
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
        unc_prop_strategy: str = "mean",  # "mean" or "sample", "sample" require transition_model to use SVGP prediction type
        sample_actor: bool = True,
        bootstrap: bool = True,
        device: str = "cuda",
    ):
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        # learning_rate: float = 3e-4,
        # max_ddpg_iterations: int = 100,  # for training DDPG
        # # std_schedule: str = "linear(1.0, 0.1, 50)",
        self.std = std
        self.std_clip = std_clip
        self.gamma = gamma
        self.tau = tau
        self.horizon = horizon
        self.num_mppi_iterations = num_mppi_iterations
        self.num_samples = num_samples
        self.mixture_coef = mixture_coef
        self.num_topk = num_topk
        self.temperature = temperature
        self.momentum = momentum
        self.unc_prop_strategy = unc_prop_strategy
        self.sample_actor = sample_actor
        self.bootstrap = bootstrap
        self.device = device = device

        # self.actor = Actor(state_dim, mlp_dims, action_dim).to(device)
        # self.critic = Critic(
        #     state_dim=state_dim, mlp_dims=mlp_dims, action_dim=action_dim
        # ).to(device)
        # self.critic_target = Critic(
        #     state_dim=state_dim, mlp_dims=mlp_dims, action_dim=action_dim
        # ).to(device)

        # # Init optimizer
        # optim_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        # optim_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.ddpg_agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            mlp_dims=mlp_dims,
            learning_rate=learning_rate,
            max_ddpg_iterations=max_ddpg_iterations,
            std=std,
            std_clip=std_clip,
            nstep=1,
            gamma=gamma,
            tau=tau,
            device=device,
        )

        # self.ddpg_agent = src.agents.ddpg.init_from_actor_critic(
        #     actor=self.actor,
        #     critic=self.critic,
        #     critic_target=self.critic_target,
        #     optim_actor=optim_actor,
        #     optim_critic=optim_critic,
        #     max_ddpg_iterations=max_ddpg_iterations,
        #     std_clip=std_clip,
        #     nstep=1,
        #     gamma=gamma,
        #     tau=tau,
        #     device=device,
        # )
        # std = torch.Tensor([0.5], device=device)
        # std = 0.5
        # std = 0.1

        self._estimate_value_fn = src.agents.objectives.greedy(
            actor=self.ddpg_agent.actor,
            critic=self.ddpg_agent.critic,
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
        self._prev_mean = torch.zeros(horizon, action_dim, device=device)

    def train(self, replay_buffer: ReplayBuffer) -> dict:
        # TODO these could be run in parallel. How to do that?
        logger.info("Starting training DDPG...")
        info = self.ddpg_agent.train(replay_buffer)
        logger.info("Finished training DDPG")

        logger.info("Starting training transition model...")
        self.transition_model.train(replay_buffer)
        # info.update(transition_model.train(replay_buffer))
        logger.info("Finished training transition model")

        logger.info("Starting training reward model...")
        self.reward_model.train(replay_buffer)
        # info.update(reward_model.train(replay_buffer))
        logger.info("Finished training reward model")

        return info

    @torch.no_grad()
    def select_action(
        self,
        state: State,
        eval_mode: bool = False,
        t0: bool = True,
    ):
        # estimate_value_compiled = torch.compile(estimate_value)
        # if isinstance(state, np.ndarray):
        #     state = torch.from_numpy(state).to(device).float()
        # print("state: {}".format(state))
        # global _prev_mean
        if isinstance(state, np.ndarray):
            # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # TODO should this be float64 or float32?
            # TODO set dynamically using type of NN params
            state = torch.tensor(
                state, dtype=torch.float64, device=self.device
            ).unsqueeze(0)
        # print("state: {}".format(state))

        # sample policy trajectories
        num_pi_trajs = int(self.mixture_coef) * self.num_samples
        if num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.horizon, num_pi_trajs, self.action_dim, device=self.device
            )
            state = state.repeat(num_pi_trajs, 1)
            for t in range(self.horizon):
                pi_actions[t] = self.ddpg_agent.actor(state, self.std).sample()
                state, _ = self.transition_model(state, pi_actions[t])

        # Initialize state and parameters
        state = state.repeat(self.num_samples + num_pi_trajs, 1)
        action_mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        action_std = 2 * torch.ones(self.horizon, self.action_dim, device=self.device)
        # TODO implememnt prev_mean
        # if not t0 and hasattr(self, "_prev_mean"):
        if not t0:
            # print("USING PREV_MEAN {}".format(_prev_mean[1:]))
            action_mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        # for i in range(num_iterations):
        # print("before loop")
        for i in range(self.num_mppi_iterations):
            # logger.info("MPPI iteration: {}".format(i))
            actions = torch.clamp(
                action_mean.unsqueeze(1)
                + action_std.unsqueeze(1)
                * torch.randn(
                    self.horizon,
                    self.num_samples,
                    self.action_dim,
                    device=self.device,
                    # TODO what device should this be on?
                    # device=std.device,
                ),
                -1,
                1,
            )
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self._estimate_value(state, actions).nan_to_num_(0)
            # value = estimate_value_compiled(state, actions).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.num_topk, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.temperature * (elite_value - max_value))
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
            action_mean = self.momentum * action_mean + (1 - self.momentum) * _mean
            action_std = _std.clamp_(0.1, 2)

        # print("after loop")
        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = action_mean  # TODO implement prev_mean
        mean, std = actions[0], action_std[0]
        if eval_mode:
            return mean
        else:
            return mean + std * torch.randn(
                self.action_dim,
                device=self.device,
                # TODO what device should this be on?
                # device=std.device,
            )

    @torch.no_grad()
    def update(self, data_new):
        state_action_input, state_diff_output, reward_output = data_new
        self.transition_model.update(data_new=(state_action_input, state_diff_output))
        self.reward_model.update(data_new=(state_action_input, reward_output))

    @torch.no_grad()
    def _estimate_value(self, start_state: State, actions: ActionTrajectory):
        return self._estimate_value_fn(start_state=start_state, actions=actions)
