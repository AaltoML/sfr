#!/usr/bin/env python3
import logging
import random
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import agents
import gpytorch
import hydra
import imageio
import models
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torchrl
import tqdm
import utils
import wandb
from dm_env import specs
from models import GaussianModelBaseEnv
from omegaconf import DictConfig, OmegaConf
from setuptools.dist import Optional
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from utils import EarlyStopper, set_seed_everywhere


@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def train(cfg: DictConfig):
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.episode_length = cfg.episode_length // cfg.env.action_repeat
    num_train_steps = cfg.num_train_episodes * cfg.episode_length

    env = hydra.utils.instantiate(cfg.env)
    eval_env = hydra.utils.instantiate(cfg.env, seed=cfg.env.seed + 42)

    cfg.state_dim = tuple(int(x) for x in env.observation_spec().shape)
    cfg.action_dim = tuple(int(x) for x in env.action_spec().shape)
    cfg.input_dim = cfg.state_dim + cfg.action_dim
    cfg.output_dim = cfg.state_dim

    num_iterations = 100
    # num_iterations = 5
    # num_iterations = cfg.episode_length // cfg.update_every_steps
    std_clip = 0.3
    nstep = 3
    gamma = 0.99
    tau = 0.005

    ###### set up workspace ######
    work_dir = (
        Path().cwd()
        / "logs"
        # / cfg.algo_name
        / cfg.name
        / cfg.env.env_name
        / cfg.env.task_name
        / str(cfg.random_seed)
    )
    if cfg.wandb.use_wandb:  # Initialise WandB
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            monitor_gym=True,
        )

    video_recorder = utils.VideoRecorder(work_dir)
    # video_recorder = VideoRecorder(work_dir) if cfg.save_video else None

    # Create replay buffer
    data_specs = (
        env.observation_spec(),
        env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    )
    replay_storage = utils.ReplayBufferStorage(data_specs, work_dir / "buffer")

    replay_loader = utils.make_replay_loader(
        replay_dir=work_dir / "buffer",
        max_size=int(num_train_steps),
        batch_size=int(cfg.batch_size),
        num_workers=4,
        save_snapshot=cfg.save_buffer,
        nstep=nstep,
        # nstep=int(cfg.planner.horizon),
        discount=1.0,
    )
    replay_iter = None

    lr = 3e-4
    mlp_dims = [512, 512]
    actor = agents.sac.Actor(cfg.state_dim[0], mlp_dims, cfg.action_dim[0]).to(
        cfg.device
    )
    critic = agents.sac.Critic(
        state_dim=cfg.state_dim[0], mlp_dims=mlp_dims, action_dim=cfg.action_dim[0]
    ).to(cfg.device)
    critic_target = agents.sac.Critic(
        state_dim=cfg.state_dim[0], mlp_dims=mlp_dims, action_dim=cfg.action_dim[0]
    ).to(cfg.device)

    # init optimizer
    optim_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=lr)

    train_sac = agents.sac.train(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        optim_actor=optim_actor,
        optim_critic=optim_critic,
        num_iterations=num_iterations,
        std_schedule="linear(1.0, 0.1, 50)",
        std_clip=std_clip,
        nstep=nstep,
        gamma=gamma,
        tau=tau,
        device=cfg.device,
    )

    global_step, start_time = 0, time.time()

    class MBRLAgent:
        def __init__(
            self,
            model,
            action_dim: int,
            horizon: int,
            num_samples: int,
            mixture_coef: float,
            num_iterations: int,
            num_topk: int,
            temperature: int,
            momentum: float,
            gamma: float,
            device: str = "cuda",
            eval_mode: bool = False,
            t0: bool = True,
        ):
            self.model = model
            self.action_dim = action_dim
            self.horizon = horizon
            self.num_samples = num_samples
            self.mixture_coef = mixture_coef
            self.num_iterations = num_iterations
            self.num_topk = num_topk
            self.temperature = temperature
            self.momentum = momentum
            self.gamma = gamma
            self.device = device

        def estimate_value(self, state, actions, horizon: int):
            """Estimate value of a trajectory starting at state and executing given actions."""
            G, discount = 0, 1
            print("state.shape {}".format(state.shape))
            print("actions.shape {}".format(actions.shape))
            for t in range(horizon):
                transition = self.model.transition(state, actions[t])
                reward = self.model.reward(state, actions[t])
                print("reward at t={}: ".format(t, reward))
                # print("state")
                # print(state.shape)
                G += discount * reward
                discount *= self.gamma
            G += discount * torch.min(
                *critic(state, actor(state, self.std).sample(clip=self.std_clip))
            )
            return G

        @torch.no_grad()
        def select_action(self, state, eval_mode: bool = False, t0=True):
            if isinstance(state, np.ndarray):
                state = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # sample policy trajectories
            num_pi_trajs = int(self.mixture_coef) * self.num_samples
            if num_pi_trajs > 0:
                pi_actions = torch.empty(
                    horizon, num_pi_trajs, self.action_dim, device=self.device
                )
                state = state.repeat(num_pi_trajs, 1)
                for t in range(horizon):
                    pi_actions[t] = self.actor(state, self.std).sample()
                    state, _ = self.model.latent_trans(state, pi_actions[t])

            # Initialize state and parameters
            z = self.model.encoder(state).repeat(num_samples + num_pi_trajs, 1)
            mean = torch.zeros(horizon, self.action_dim, device=self.device)
            std = 2 * torch.ones(horizon, self.action_dim, device=self.device)
            if not t0 and hasattr(self, "_prev_mean"):
                mean[:-1] = self._prev_mean[1:]

            # Iterate CEM
            for i in range(self.num_iterations):
                logger.info("MPPI iteration: {}".format(i))
                actions = torch.clamp(
                    mean.unsqueeze(1)
                    + std.unsqueeze(1)
                    * torch.randn(
                        self.horizon,
                        self.num_samples,
                        self.action_dim,
                        device=std.device,
                    ),
                    -1,
                    1,
                )
                if num_pi_trajs > 0:
                    actions = torch.cat([actions, pi_actions], dim=1)

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
            actions = elite_actions[
                :, np.random.choice(np.arange(score.shape[0]), p=score)
            ]
            self._prev_mean = mean
            mean, std = actions[0], _std[0]
            action = mean
            if not eval_mode:
                action += std * torch.randn(self.action_dim, device=std.device)
            return action

    class DDPGAgent:
        def __init__(self):
            pass

        @torch.no_grad()
        def select_action(self, state, eval_mode: bool = False, t0=None):
            state = torch.Tensor(state)
            dist = actor.forward(state, std=0)
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
            return action

    agent = DDPGAgent()

    for episode_idx in range(cfg.num_train_episodes):
        # Collect trajectory
        time_step = env.reset()
        replay_storage.add(time_step)
        episode_step, episode_reward = 0, 0
        while not time_step.last():
            if episode_idx < cfg.init_random_episodes:
                action = np.random.uniform(-1, 1, env.action_spec().shape).astype(
                    dtype=env.action_spec().dtype
                )
            else:
                action = agent.select_action(
                    time_step.observation, eval_mode=False, t0=time_step.first
                )
                action = action.cpu().numpy()

            time_step = env.step(action)
            replay_storage.add(time_step)

            global_step += 1
            episode_step += 1

        ###### update model ######
        if episode_idx >= cfg.init_random_episodes:
            # print("TRAINING SAC")
            # for _ in range(cfg.episode_length // cfg.update_every_steps):
            if replay_iter is None:
                replay_iter = iter(replay_loader)
            # print("replay_iter {}".format(replay_iter))
            # train_info = agent.update(
            #     ep, replay_iter, cfg.batch_size
            # )  # log training every episode
            # Update actor/critic with SAC
            # TODO remove episode_idx from train_sac if not needed for std_schedule
            train_sac(replay_iter, episode_idx=episode_idx)
            # print("Finished training SAC")

            ###### evaluation ######
            if episode_idx % cfg.eval_episode_freq == 0:
                # print("Evaluating {}".format(episode_idx))
                Gs = utils.evaluate(
                    eval_env,
                    agent,
                    episode_idx=episode_idx,
                    # num_episode=cfg.eval_episode_freq,
                    num_episodes=10,
                    video=video_recorder,
                )
                # print("DONE EVALUATING")
                episode_reward = np.mean(Gs)
                env_step = global_step * cfg.env.action_repeat
                eval_metrics = {
                    "episode": episode_idx,
                    "step": global_step,
                    "env_step": env_step,
                    # "time": time.time() - start_time,
                    "episode_reward": episode_reward,
                    # "eval_total_time": timer.total_time(),
                }
                logger.info(
                    "Episode: {} | Reward: {}".format(episode_idx, episode_reward)
                )
                if cfg.wandb.use_wandb:
                    wandb.log({"eval/": eval_metrics}, step=env_step)

        # num_new_inducing_points_per_episode = 0
        # logger.info("Training dynamic model")
        # print("Training world model")
        # print("Training transition model")
        # # world_model.transition_model.train(replay_buffer_model)
        # print("Training reward model")
        # # world_model.reward_model.train(replay_buffer_model)
        # print("DONE TRAINING WORLD MODEL")
        # logger.info("DONE TRAINING MODEL")

        # Log rewards/videos in eval env
        # eval_policy(logger=logger, name="planner", policy=planner, env=eval_env, step=i)
        # eval_policy(
        #     logger=logger, name="policy", policy=policy_module, env=eval_env, step=i
        # )


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
