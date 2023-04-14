#!/usr/bin/env python3
import os
import re
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


__REDUCE__ = lambda b: "mean" if b else "none"


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def soft_update_params(model, model_target, tau: float):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for params, params_target in zip(model.parameters(), model_target.parameters()):
            params_target.data.lerp_(params.data, tau)


def mlp(in_dim, mlp_dims: List[int], out_dim, act_fn=nn.ELU, out_act=nn.Identity):
    """Returns an MLP."""
    if isinstance(mlp_dims, int):
        raise ValueError("mlp dimensions should be list, but got int.")

    layers = [nn.Linear(in_dim, mlp_dims[0]), act_fn()]
    for i in range(len(mlp_dims) - 1):
        layers += [nn.Linear(mlp_dims[i], mlp_dims[i + 1]), act_fn()]

    layers += [nn.Linear(mlp_dims[-1], out_dim), out_act()]
    return nn.Sequential(*layers)


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # elif isinstance(m, EnsembleLinear):
    #     for w in m.weight.data:
    #         nn.init.orthogonal_(w)
    #     if m.bias is not None:
    #         for b in m.bias.data:
    #             nn.init.zeros_(b)
    elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        # nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)


def to_torch(xs, device, dtype=torch.float32):
    return tuple(torch.as_tensor(x, device=device, dtype=dtype) for x in xs)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Actor(nn.Module):
    def __init__(self, state_dim, mlp_dims, action_dim):
        super().__init__()
        # self.trunk = nn.Sequential(
        #     nn.Linear(latent_dim, mlp_dims[0]), nn.LayerNorm(mlp_dims[0]), nn.Tanh()
        # )
        self._actor = mlp(state_dim, mlp_dims, action_dim)
        self.apply(orthogonal_init)

    def forward(self, state, std):
        # feature = self.trunk(state)
        mu = self._actor(state)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim, mlp_dims, action_dim):
        super().__init__()
        # self.trunk = nn.Sequential(
        #     nn.Linear(latent_dim + action_shape[0], mlp_dims[0]),
        #     nn.LayerNorm(mlp_dims[0]),
        #     nn.Tanh(),
        # )
        self._critic1 = mlp(state_dim + action_dim, mlp_dims, 1)
        self._critic2 = mlp(state_dim + action_dim, mlp_dims, 1)
        self.apply(orthogonal_init)

    def forward(self, state, action):
        state_action_input = torch.cat([state, action], dim=-1)
        return self._critic1(state_action_input), self._critic2(state_action_input)


def update_q(
    optim: torch.optim.Optimizer,
    actor: Actor,
    critic: Critic,
    critic_target: Critic,
    std_clip: float = 0.3,
):
    def update_q_(state, act, rew, discount, next_state, std: float):
        with torch.no_grad():
            action = actor(next_state, std=std).sample(clip=std_clip)

            td_target = rew + discount * torch.min(*critic_target(next_state, action))

        q1, q2 = critic(state, act)
        q_loss = torch.mean(mse(q1, td_target) + mse(q2, td_target))

        optim.zero_grad(set_to_none=True)
        q_loss.backward()
        optim.step()

        return {"q": q1.mean().item(), "q_loss": q_loss.item()}

    return update_q_


def update_actor(
    optim: torch.optim.Optimizer, actor: Actor, critic: Critic, std_clip: float = 0.3
):
    def update_actor_(state, std: float):
        a = actor(state, std=std).sample(clip=std_clip)
        Q = torch.min(*critic(state, a))
        actor_loss = -Q.mean()

        optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        optim.step()

        return {"actor_loss": actor_loss.item()}

    return update_actor_


def train(
    actor: Actor,
    critic: Critic,
    critic_target: Critic,
    optim_actor,
    optim_critic,
    num_iterations: int,
    # std_schedule: str = "linear(1.0, 0.1, 50)",
    std_schedule: str = "linear(1.0, 0.1, 500)",
    std_clip: float = 0.3,
    nstep: int = 3,
    gamma: float = 0.99,
    tau: float = 0.005,
    device: str = "cuda",
):
    std = linear_schedule(std_schedule, 0)

    update_q_fn = update_q(
        optim=optim_critic,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        std_clip=std_clip,
    )
    update_actor_fn = update_actor(
        optim=optim_actor, actor=actor, critic=critic, std_clip=std_clip
    )

    def train_(replay_iter, episode_idx: int):
        for i in range(num_iterations):
            std = linear_schedule(
                std_schedule, (num_iterations - 10) * episode_idx + i
            )  # linearly udpate std
            # std = 0
            info = {"std": std}

            batch = next(replay_iter)
            state, action, reward, discount, next_state = to_torch(
                batch, device, dtype=torch.float32
            )
            # swap the batch and horizon dimension -> [H, B, _shape]
            action, reward, discount, next_state = (
                torch.swapaxes(action, 0, 1),
                torch.swapaxes(reward, 0, 1),
                torch.swapaxes(discount, 0, 1),
                torch.swapaxes(next_state, 0, 1),
            )
            # print("reward {}".format(reward.shape))
            # print("action {}".format(action.shape))
            # print("next_state{}".format(next_state.shape))

            # form n-step samples
            _rew, _discount = 0, 1
            for t in range(nstep):
                _rew += _discount * reward[t]
                _discount *= gamma

            info.update(update_q_fn(state, action[0], _rew, _discount, next_state, std))
            info.update(update_actor_fn(state, std))

            # print("Iteration {}".format(i))
            # print("info {}".format(info))
            # update target network
            soft_update_params(critic, critic_target, tau=tau)
            # if i % 10 == 0:
            #     print("SAC: Epoch {} | Loss {}".format(i, loss_value))
            # if early_stop(loss_value):
            #     print("SAC loss early stop criteria met, exiting training loop")
            #     break
            wandb.log(info)

        # print("SAC: Epoch {}".format(epoch))

        return info

    return train_
