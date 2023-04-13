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
import wandb
from dm_env import specs
from models import GaussianModelBaseEnv
from omegaconf import DictConfig, OmegaConf

# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import WandbLogger
from setuptools.dist import Optional
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import CompositeSpec, TensorDictReplayBuffer
from torchrl.data.postprocs import MultiStep
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    RandomSampler,
    SamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torchrl.envs import CatTensors, DoubleToFloat
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import ObservationNorm, RewardScaling, TransformedEnv
from torchrl.envs.utils import step_mdp
from torchrl.modules import (
    Actor,
    CEMPlanner,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.trainers import Recorder
from utils import EarlyStopper, set_seed_everywhere


class Actor(nn.Module):
    def __init__(self, state_dim, mlp_dims, action_shape):
        super().__init__()
        # self.trunk = nn.Sequential(
        #     nn.Linear(latent_dim, mlp_dims[0]), nn.LayerNorm(mlp_dims[0]), nn.Tanh()
        # )
        self._actor = net.mlp(mlp_dims[0], mlp_dims[1:], action_shape[0])
        self.apply(net.orthogonal_init)

    def forward(self, state, std):
        # feature = self.trunk(state)
        mu = self._actor(state)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return h.TruncatedNormal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim, mlp_dims, action_shape):
        super().__init__()
        # self.trunk = nn.Sequential(
        #     nn.Linear(latent_dim + action_shape[0], mlp_dims[0]),
        #     nn.LayerNorm(mlp_dims[0]),
        #     nn.Tanh(),
        # )
        self._critic1 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self._critic2 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self.apply(net.orthogonal_init)

    def forward(self, state, action):
        state_action_input = torch.cat([state, action], dim=-1)
        return self._critic1(state_action_input), self._critic2(state_action_input)


class Mss:
    def __init__(self, device):
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.actor = Actor(state_dim, mlp_dims, action_shape).to(self.device)

        self.critic = Critic(
            state_dim=state_dim, mlp_dims=mlp_dims, action_shape=action_shape
        ).to(self.device)
        self.critic_target = Critic(
            state_dim=state_dim, mlp_dims=mlp_dims, action_shape=action_shape
        ).to(self.device)

        # init optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)


def make_replay_buffer(
    buffer_size: int,
    device,
    buffer_scratch_dir: str = "/tmp/",
    batch_size: int = 64,
    # prefetch: int = 3,
):
    sampler = RandomSampler()
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(buffer_size, device=device),
        # storage=LazyMemmapStorage(
        #     buffer_size,
        #     scratch_dir=buffer_scratch_dir,
        #     device=device,
        # ),
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=False,
        # prefetch=prefetch,
    )
    return replay_buffer


def make_transformed_env(
    cfg: DictConfig,
    reward_scaling=1.0,
    init_env_steps: int = 1000,
    from_pixels: bool = False,
    pixels_only: bool = False,
    seed: int = 42,
):
    """Apply transforms to the env (such as reward scaling and state normalization)."""

    env = hydra.utils.instantiate(
        cfg.env,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
        # random=seed
    )

    double_to_float_list, double_to_float_inv_list = [], []
    if isinstance(env, DMControlEnv):
        # if isinstance(env, torchrl.envs.libs.dm_control.DMControlEnv):
        print("env is dm_env so making double precision")
        # if env_library is torchrl.envs.libs.dm_control.DMControlEnv:
        # DMControl requires double-precision
        double_to_float_list += ["reward", "action"]
        double_to_float_inv_list += ["action"]

    out_key = "state_vector"
    obs_keys = list(env.observation_spec.keys())
    # print("obs_keys {}".format(obs_keys))
    if from_pixels:
        obs_keys.remove("pixels")
    # print("obs_keys {}".format(obs_keys))

    double_to_float_list.append(out_key)
    env = TransformedEnv(
        env,
        torchrl.envs.transforms.Compose(
            RewardScaling(loc=0.0, scale=reward_scaling),
            CatTensors(in_keys=obs_keys, out_key=out_key),
            ObservationNorm(in_keys=[out_key]),
            DoubleToFloat(
                in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
            ),
        ),
    )
    env.transform[2].init_stats(init_env_steps)
    return env


def eval_policy(logger: WandbLogger, name: str, policy, env, step: int, fps: int = 15):
    env.reset()
    eval_rollout = env.rollout(max_steps=200, policy=policy, auto_reset=True).cpu()
    logger.log_scalar(
        name=name + " reward",
        value=eval_rollout["reward"].numpy().mean(),
        step=step,
    )
    logger.log_video(
        name=name + " video",
        video=eval_rollout["pixels_save"].permute(0, 3, 1, 2)[None, ...],
        fps=fps,
        step=step,
    )


def get_env_stats(cfg, seed: Optional[int] = 42):
    """Gets the stats of an environment."""
    proof_env = make_transformed_env(cfg, seed=seed)
    # print("proof_env {}".format(proof_env))
    transform_state_dict = proof_env.state_dict()
    proof_env.close()
    return transform_state_dict


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

    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )

    if cfg.wandb.use_wandb:  # Initialise WandB
        logger = WandbLogger(
            exp_name=cfg.wandb.run_name,
            # offline=False,
            # save_dir=None,
            # id=cfg.wandb.id,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        )

    env = make_transformed_env(cfg=cfg)

    world_model = hydra.utils.instantiate(cfg.model_new)
    print("World model {}".format(world_model))

    total_frames = 500000 // cfg.env.frame_skip
    frames_per_batch = 1000 // cfg.env.frame_skip

    max_grad_norm = 1.0
    num_cells = 256
    # num_cells = 64
    sub_batch_size = 64  # cardinality of the sub-samples gathered from data in the
    # sub_batch_size = 25  # cardinality of the sub-samples gathered from data in the
    clip_epsilon = 0.2
    entropy_eps = 1e-4

    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
    policy_module = ProbabilisticActor(
        module=TensorDictModule(
            actor_net, in_keys=["state_vector"], out_keys=["loc", "scale"]
        ),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.minimum,
            "max": env.action_spec.space.maximum,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )
    # actor_net = nn.Sequential(
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(env.action_spec.shape[-1], device=device),
    # )
    # policy_module = Actor(
    #     module=TensorDictModule(
    #         actor_net, in_keys=["state_vector"], out_keys=["action"]
    #     ),
    #     spec=env.action_spec,
    #     in_keys=["state_vector"],
    # )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(module=value_net, in_keys=["state_vector"])
    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))

    gamma = 0.99
    lmbda = 0.95
    lr = 5e-4
    # lr = 1e-4
    weight_decay = 0.0
    num_epochs = 7000
    # num_epochs = 500
    # patience = 500
    patience = 1000
    min_delta = 0.0
    # min_delta = 0.2

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    # from torchrl.objectives import DQNLoss, ValueEstimators

    # # loss_module = DQNLoss(actor)
    # loss_module = DQNLoss(value_network=value_module)
    # kwargs = {"gamma": 0.9, "lmbda": 0.9}
    # loss_module.make_value_estimator(ValueEstimators.TDLambda, **kwargs)

    loss_module = torchrl.objectives.ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        # critic=planner.advantage_module.value_network,
        advantage_key="advantage",
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        value_target_key=advantage_module.value_target_key,
        # value_target_key=planner.advantage_module.value_target_key,
        critic_coef=1.0,
        gamma=0.99,
        loss_critic_type="smooth_l1",
    )

    # # Get model
    # actor, actor_explore = make_model(test_env)
    # loss_module, target_net_updater = get_loss_module(actor, gamma)
    # target_net_updater.init_()
    print("loss_module {}".format(loss_module))

    # optim = torch.optim.Adam(loss_module.parameters(), lr)
    # optimizer = torch.optim.Adam(
    #     planner.advantage_module.parameters(),
    #     lr=lr,
    #     weight_decay=weight_decay
    #     # planner.advantage_module.parameters(), lr=lr, weight_decay=weight_decay
    # )

    early_stop = EarlyStopper(patience=patience, min_delta=min_delta)

    # Configure agent
    planner = hydra.utils.instantiate(
        cfg.planner, env=world_model, advantage_module=advantage_module
    )
    print("Planner: {}".format(planner))
    print(
        "planner.advantage_module.value_network  {}".format(
            planner.advantage_module.value_network
        )
    )

    planner = ProbabilisticActor(
        module=TensorDictModule(
            planner,
            in_keys=["state_vector"],
            out_keys=["loc", "scale"],
        ),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        # return_log_prob=True,
    )

    # from torchrl.objectives.value import TDLambdaEstimator

    # value = torchrl.objectives.value.TDLambdaEstimator(
    # value = TDLambdaEstimator(
    #     gamma=0.99,
    #     lmbda=0.95,
    #     value_network=torchrl.modules.value.ValueOperator(
    #         module=torch.nn.Linear(in_features=5, out_features=1),
    #         in_keys=["state_vector"],
    #     ),
    # )
    # print("value {}".format(value))
    # value=torchrl.objectives.value.TDLambdaEstimator(gamma=0.99,lmbda= 0.95,value_network=torchrl.modules.value.ValueOperator(module=torch.nn.Linear(in_features= 5,out_features= 1),in_keys=["state_vector"]))

    # print("Making GaussianModelBaseEnv")
    # model_env = GaussianModelBaseEnv(
    #     transition_model=transition_model,
    #     reward_model=reward_model,
    #     state_size=5,
    #     action_size=1,
    #     device=device,
    #     # dtype=None,
    #     # batch_size: int = None,
    # )
    # print("making planner")
    # planner = CEMPlanner(
    #     world_model,
    #     # model_env,
    #     planning_horizon=2,
    #     optim_steps=5,
    #     # optim_steps=11,
    #     num_candidates=5,
    #     top_k=3,
    #     reward_key="reward",
    #     # reward_key="expected_reward",
    #     action_key="action",
    # )
    # print("MADE planner")

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        buffer_size=cfg.replay_buffer_capacity,
        device=device,
        buffer_scratch_dir=cfg.buffer_scratch_dir,
        # batch_size=cfg.batch_size,
        # prefetch=3,
    )

    replay_buffer_model = make_replay_buffer(
        buffer_size=cfg.replay_buffer_capacity,
        device=device,
        buffer_scratch_dir=cfg.buffer_scratch_dir,
        # batch_size=cfg.batch_size,
        # prefetch=3,
    )

    transform_state_dict = get_env_stats(cfg, seed=cfg.random_seed)
    # TODO should this use different seed?
    # recorder = make_recorder(
    #     cfg,
    #     actor_model_explore=planner,
    #     transform_state_dict=transform_state_dict,
    #     logger=logger,
    #     record_interval=10,
    #     seed=42,
    # )

    # frames_per_batch = 50
    collector = SyncDataCollector(
        # create_env_fn=partial(make_transformed_env, train_env),
        create_env_fn=partial(make_transformed_env, cfg=cfg),
        policy=planner,
        # policy=policy_module,
        # policy=torchrl.collectors.collectors.RandomPolicy(
        #     make_transformed_env(cfg).action_spec
        # ),
        total_frames=total_frames,
        # max_frames_per_traj=50,
        frames_per_batch=frames_per_batch,
        # init_random_frames=-1,
        # init_random_frames=500,
        init_random_frames=500,
        reset_at_each_iter=False,
        # split_trajs=True,
        split_trajs=False,
        device=device,
        storing_device=device,
        # device="cpu",
        # storing_device="cpu",
    )

    # Make env for logging reward/video for policy
    eval_env = make_transformed_env(
        cfg=cfg,
        reward_scaling=1.0,
        init_env_steps=1000,
        from_pixels=True,
        pixels_only=False,
    )
    eval_env.transform.insert(0, CatTensors(["pixels"], "pixels_save", del_keys=False))

    norm_factor_training = 1  # TODO set this correctly
    rewards, rewards_eval = [], []
    collected_frames = 0

    j = 0
    for i, tensordict in enumerate(collector):
        print("Episode: {}".format(i))
        print("tensordict: {}".format(tensordict))

        # extend the replay buffer with the new data
        if ("collector", "mask") in tensordict.keys(True):
            # if multi-step, a mask is present to help filter padded values
            current_frames = tensordict["collector", "mask"].sum()
            tensordict = tensordict[tensordict.get(("collector", "mask"))]
        else:
            tensordict = tensordict.view(-1)
            current_frames = tensordict.numel()
        collected_frames += current_frames
        replay_buffer_model.extend(tensordict.cpu())
        # replay_buffer.extend(tensordict)
        print("replay_buffer {}".format(replay_buffer))

        print("Training advantage network")
        # we now have a batch of data to work with. Let's learn something from it.
        early_stop.reset()
        early_stop_flag = False
        optim = torch.optim.Adam(loss_module.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // frames_per_batch, 0.0
        )
        start = time.time()
        for epoch in range(num_epochs):
            advantage_module(tensordict)
            data_view = tensordict.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for batch_idx in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                wandb.log({"loss_objective": loss_vals["loss_objective"]}, step=j)
                wandb.log({"loss_critic": loss_vals["loss_critic"]}, step=j)
                wandb.log({"loss_entropy": loss_vals["loss_entropy"]}, step=j)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                wandb.log({"ppo_loss": loss_value}, step=j)

                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
                j += 1
                if early_stop(loss_value):
                    print("PPO loss early stop criteria met, exiting training loop")
                    early_stop_flag = True
                    break
            if epoch % 10 == 0:
                print("PPO: Epoch {} | Loss {}".format(epoch, loss_value))
            if early_stop_flag:
                break
        print("PPO: Epoch {}".format(epoch))
        end = time.time()
        print("Time: {}".format(end - start))

        # step the learning rate schedule
        scheduler.step()

        # num_new_inducing_points_per_episode = 0
        # logger.info("Training dynamic model")
        print("Training world model")
        print("num data: {}".format(len(replay_buffer_model)))
        print("Training transition model")
        # world_model.transition_model.train(replay_buffer_model)
        print("Training reward model")
        # world_model.reward_model.train(replay_buffer_model)
        print("DONE TRAINING WORLD MODEL")
        # logger.info("DONE TRAINING MODEL")

        # Log rewards/videos in eval env
        eval_policy(logger=logger, name="planner", policy=planner, env=eval_env, step=i)
        eval_policy(
            logger=logger, name="policy", policy=policy_module, env=eval_env, step=i
        )


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
