#!/usr/bin/env python3
from abc import ABCMeta
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from torchtyping import TensorType

from ..custom_types import ActionTrajectory


def exponential_moving_average(new, old, momentum: float = 0.1):
    return momentum * old + (1.0 - momentum) * new


class TrajectoryOptimizer(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        objective_fn: Callable[[ActionTrajectory], TensorType["1"]],
        initial_solution: ActionTrajectory,  # [horizon, action_dim]
        replan_freq: int = 1,
        warm_start: bool = True,
    ):
        # self.initial_solution = self.initial_solution.repeat((planning_horizon, 1))
        self.initial_solution = initial_solution
        self.previous_solution = self.initial_solution.clone()
        self.objective_fn = objective_fn
        self.replan_freq = replan_freq
        self.warm_start = warm_start

    def optimize(
        self, initial_solution: Optional[ActionTrajectory] = None, **kwargs
    ) -> ActionTrajectory:
        """Run optimization."""
        raise NotImplementedError

    def reset(self):
        """Reset previous solution to the initial solution"""
        self.previous_solution = self.initial_solution.clone()

    @property
    def horizon(self):
        return self.initial_solution.shape[0]

    @property
    def action_dim(self):
        return self.initial_solution.shape[1]


class CEMTrajectoryOptimizer(TrajectoryOptimizer):
    def __init__(
        self,
        objective_fn: Callable[[ActionTrajectory], TensorType["1"]],
        initial_solution: ActionTrajectory,  # [horizon, action_dim]
        replan_freq: int = 1,
        warm_start: bool = True,
        num_iterations: int = 5,
        population_size: int = 5,
        elite_ratio: float = 0.2,
        momentum: float = 0
        # num_samples: int = 5,
        # num_topk: int = 5,
    ):
        assert initial_solution.ndim == 2
        # Add mapping over leading dim for aciton samples
        self.objective_fn = torch.vmap(objective_fn)
        super().__init__(
            objective_fn=objective_fn,
            initial_solution=initial_solution,
            replan_freq=replan_freq,
            warm_start=warm_start,
        )
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.num_topk = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        self.momentum = momentum
        # self.num_samples = num_samples
        # self.num_topk = self.population_size

    def optimize(
        self, initial_solution: Optional[ActionTrajectory] = None
    ) -> ActionTrajectory:
        if initial_solution is not None:
            assert initial_solution.ndim == 2
            self.previous_solution = self.initial_solution.clone()

        # Initialize action dists
        # action_means = torch.zeros((self.plan_horizon, self.action_dim))
        action_means = self.previous_solution
        action_stds = torch.ones_like(action_means)
        actions_dists = td.Normal(loc=action_means, scale=action_stds)

        best_actions = torch.empty_like(action_means)
        best_value = -np.inf
        for _ in range(self.num_iterations):
            action_samples = actions_dists.sample(torch.Size([self.population_size]))
            print("action_samples {}".format(action_samples.shape))
            value_samples = self.objective_fn(action_samples)
            print("value_samples {}".format(value_samples.shape))
            # TODO should this be using dim=0 and largest=True
            elite_value_samples, elite_idxs = value_samples.topk(
                self.num_topk, dim=0, largest=True
            )
            print("elite_value_samples: {}".format(elite_value_samples.shape))
            print("elite_idxs: {}".format(elite_idxs.shape))
            elite_actions = action_samples[elite_idxs, ...]
            print("elite_actions.shape: {}".format(elite_actions.shape))

            # Update action dist
            new_action_means = torch.mean(elite_actions, dim=0)
            new_action_stds = torch.std(elite_actions, dim=0)
            actions_dists = td.Normal(
                loc=exponential_moving_average(
                    new_action_means, old=action_means, momentum=self.momentum
                ),
                scale=exponential_moving_average(
                    new_action_stds, old=action_stds, momentum=self.momentum
                ),
            )
            if elite_value_samples[0] > best_value:
                # check if the top elite action sample is best out of all iterations
                best_value = elite_value_samples[0]
                best_actions = action_samples[elite_idxs[0]].clone()

        if self.keep_last_solution:
            self._prev_action_means = actions_dists.mean()

        if self.return_mean_elites:
            return action_means
        else:
            return best_actions
        # return actions_dists
        # return mean, std
        # select the first action in the planed horizon
        # action, std = mean[0], std[0]

        # if not eval_mode:
        #     action += self.expl_noise * torch.randn(action.shape)

        # udpate the simulator state (if use simulator to do planning)
        # o, r, d, _ = self.model.step(action)
        # self.model.save_checkpoint()

        # return action
