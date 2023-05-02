#!/usr/bin/env python3
from typing import Optional

import models
import numpy as np
import torch
import torchrl
from custom_types import Action, State
from tensordict.tensordict import TensorDictBase


# class MPPI(Actor):
# class MPPI(torch.nn.Module):
class MPPI:
    def __init__(
        self,
        model: models.DynamicModel,
        proof_env: torchrl.envs.EnvBase,
        advantage_module: nn.Module,
        horizon: int,
        num_samples: int,
        mixture_coef: float,
        num_iterations: int,
        num_topk: int,
        temperature: int,
        momentum: float,
        gamma: float,
        device,
        eval_mode: bool = False,
        t0: bool = True,
    ):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.mixture_coef = mixture_coef
        self.num_iterations = num_iterations
        self.num_topk = num_topk
        self.temperature = temperature
        self.momentum = momentum
        self.gamma = gamma
        self.device = device

        self.env = proof_env
        self.action_spec = self.env.action_spec
        # self.state_dim = self.env.observation_spec
        # print("self.state_dim")
        # print(self.state_dim)

        self.eval_mode = eval_mode
        self.t0 = t0

    # def estimate_value(self, start_state, actions, horizon: int):
    #     """Estimate value of a trajectory starting at latent state z and executing given actions."""
    #     G, discount = 0, 1
    #     for t in range(horizon):
    #         z, reward = self.model(state, actions[t])
    #         G += discount * reward
    #         discount *= self.gamma
    #     G += discount * torch.min(
    #         *self.critic(z, self.actor(z, self.std).sample(clip=self.std_clip))
    #     )
    #     return G
    # def optimistic_value(self, start_state: State, actions, horizon: int):
    def estimate_value(self, start_state: State, actions, horizon: int):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        # reward = self.env._task.get_reward(self.env._physics)
        G, discount = 0, 1
        state = start_state
        # print("state.shape")
        # print(state.shape)
        # print("actions.shape")
        # print(actions.shape)
        # print(actions[0].shape)
        # actions = actions[:, 0 : -self.state_dim]
        for t in range(horizon):
            # state, reward = self.model(state, actions[t])
            prediction = self.model(state, actions[t])
            # print("prediction")
            # print(prediction)
            reward = prediction.reward_dist.mean
            # print("reward at {}".format(t))
            # print(reward.shape)
            state = prediction.output_dist.mean
            # print("state")
            # print(state.shape)
            G += discount * reward
            discount *= self.gamma
        # G += discount * torch.min(
        #     *self.critic(state, self.actor(z, self.std).sample(clip=self.std_clip))
        # )
        return G

    # def forward(
    #     self, state: Optional[State] = None, eval_mode: bool = False, t0: bool = False
    # ) -> Action:
    #     return self(state)
    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        state = td.get("observation_vector")
        # print("state before shaping")
        # print(state.shape)
        # sample policy trajectories
        num_pi_trajs = int(self.mixture_coef) * self.num_samples
        if num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.horizon, num_pi_trajs, self.action_shape[0], device=self.device
            )
            state = state.repeat(num_pi_trajs, 1)
            for t in range(self.horizon):
                pi_actions[t] = self.actor(state, self.std).sample()
                state, _ = self.model.latent_trans(state, pi_actions[t])

        # Initialize state and parameters
        state = state.repeat(self.num_samples + num_pi_trajs, 1)
        # print("state after shaping before loop")
        # print(state.shape)
        mean = torch.zeros(self.horizon, self.action_shape[0], device=self.device)
        std = 2 * torch.ones(self.horizon, self.action_shape[0], device=self.device)
        if not self.t0 and hasattr(self, "_prev_mean"):
            # TODO implement prev_mean
            mean[:-1] = self._prev_mean[1:]

        # Iterate MPPI
        for i in range(self.num_iterations):
            logger.info("MPPI iteration: {}".format(i))
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(
                    self.horizon,
                    self.num_samples,
                    self.action_shape[0],
                    device=std.device,
                ),
                -1,
                1,
            )
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)
            # print("action inside loop")
            # print(actions.shape)

            # Compute elite actions
            value = self.estimate_value(state, actions, self.horizon).nan_to_num_(0)
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
            _std = _std.clamp_(0.1, 2)
            mean, std = (self.momentum * mean + (1 - self.momentum) * _mean, _std)

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        action = mean
        if not self.eval_mode:
            action += std * torch.randn(self.action_shape[0], device=std.device)
        return td.set("action", action)

    @property
    def action_shape(self):
        return self.action_spec.shape
