#!/usr/bin/env python3
from typing import Optional

import numpy as np
import torch
from src.custom_types import Action, State
from torchrl.data.replay_buffers import ReplayBuffer

from torchrl.modules import Actor

# from .base import Policy


# class MPCPolicy(Policy):
#     def __init__(
#         self, state_dim: int, action_dim: int, trajectory_optimizer, *args, **kwargs
#     ):
#         super().__init__(state_dim=state_dim, action_dim=action_dim)
#         self.trajectory_optimizer = trajectory_optimizer

#     @torch.jit.export
#     def update(self, replay_buffer: ReplayBuffer, *args, **kwargs):
#         """Update parameters."""
#         pass

#     @torch.jit.export
#     def reset(self):
#         """Reset parameters."""
#         pass

#     @torch.jit.export
#     def forward(self, state: Optional[state] = None) -> Action:
#         if self._steps % self.solver_frequency == 0 or self.action_sequence is None:
#             self.action_sequence = self.solver(state)
#         else:
#             self.trajectory_optimizer.initialize_actions(state.shape[:-1])

#         action = self.action_sequence[self._steps % self.solver_frequency, ..., :]
#         self._steps += 1
#         return action, torch.zeros(self.dim_action[0], self.dim_action[0])

#     def reset(self):
#         """Reset trajectory optimizer."""
#         self._steps = 0
#         self.trajectory_optimizer.reset()


class MPPI(Actor):
    def __init__(
        self,
        horizon: int,
        num_samples: int,
        mixture_coef: float,
        num_iterations: int,
        num_topk: int,
        temperature: int,
        momentum: float,
        device,
        # in_keys: List[str] = ["observation_vector"],
        # out_keys: List[str] = ["action"],
    ):
        # super().__init__(in_keys=in_keys, out_keys=out_keys, spec=spec)
        self.horizon = horizon
        self.num_samples = num_samples
        self.mixture_coef = mixture_coef
        self.num_iterations = num_iterations
        self.num_topk = num_topk
        self.temperature = temperature
        self.momentum = momentum
        self.device = device

    def __call__(
        self, state: Optional[State] = None, eval_mode: bool = False, t0: bool = False
    ) -> Action:
        # sample policy trajectories
        num_pi_trajs = int(self.mixture_coef) * self.num_samples
        if num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.horizon, num_pi_trajs, self.action_shape[0], device=self.device
            )
            z = state.repeat(num_pi_trajs, 1)
            for t in range(self.horizon):
                pi_actions[t] = self.actor(z, self.std).sample()
                z, _ = self.model.latent_trans(z, pi_actions[t])

        # Initialize state and parameters
        z = state.repeat(self.num_samples + num_pi_trajs, 1)
        mean = torch.zeros(self.horizon, self.action_shape[0], device=self.device)
        std = 2 * torch.ones(self.horizon, self.action_shape[0], device=self.device)
        if not t0 and hasattr(self, "_prev_mean"):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.num_iterations):
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

            # Compute elite actions
            value = self.estimate_value(z, actions, self.horizon).nan_to_num_(0)
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
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.action_shape[0], device=std.device)
        return a
