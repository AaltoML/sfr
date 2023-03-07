#!/usr/bin/env python3
import logging
import random
from functools import partial
from pathlib import Path
from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gpytorch
import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchrl
import tqdm
import wandb
from dm_env import specs
from models.gp.svgp import SVGPDynamicModel
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from setuptools.dist import Optional
from tensordict.nn import TensorDictModule
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
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.trainers import Recorder
from utils import set_seed_everywhere


def make_replay_buffer(
    buffer_size: int, device, buffer_scratch_dir: str = "/tmp/", prefetch: int = 3
):
    sampler = RandomSampler()
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(buffer_size, device=device),
        # storage=LazyMemmapStorage(
        #     buffer_size,
        #     scratch_dir=buffer_scratch_dir,
        #     device=device,
        # ),
        sampler=sampler,
        pin_memory=False,
        prefetch=prefetch,
    )
    return replay_buffer


def make_transformed_env(env, reward_scaling=1.0):
    """Apply transforms to the env (such as reward scaling and state normalization)."""

    env = TransformedEnv(env)

    # we append transforms one by one, although we might as well create the
    # transformed environment using the `env = TransformedEnv(base_env, transforms)`
    # syntax.
    env.append_transform(RewardScaling(loc=0.0, scale=reward_scaling))

    double_to_float_list = []
    double_to_float_inv_list = []
    if isinstance(env, torchrl.envs.libs.dm_control.DMControlEnv):
        logger.info("env is dm_env so making double precision")
        # if env_library is torchrl.envs.libs.dm_control.DMControlEnv:
        # DMControl requires double-precision
        double_to_float_list += [
            "reward",
            "action",
        ]
        double_to_float_inv_list += ["action"]

    # We concatenate all states into a single "observation_vector"
    # even if there is a single tensor, it'll be renamed in "observation_vector".
    # This facilitates the downstream operations as we know the name of the
    # output tensor.
    # In some environments (not half-cheetah), there may be more than one
    # observation vector: in this case this code snippet will concatenate them
    # all.
    out_key = "observation_vector"
    obs_keys = list(env.observation_spec.keys())
    env.append_transform(CatTensors(in_keys=obs_keys, out_key=out_key))

    # we normalize the states, but for now let's just instantiate a stateless
    # version of the transform
    # env.append_transform(ObservationNorm(in_keys=[out_key], standard_normal=True))

    double_to_float_list.append(out_key)
    env.append_transform(
        DoubleToFloat(
            in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
        )
    )

    return env


def make_recorder(
    cfg: DictConfig,
    actor_model_explore,
    transform_state_dict,
    record_interval=10,
    seed=42,
):
    base_env = hydra.utils.instantiate(cfg.env, random=seed)
    recorder = make_transformed_env(base_env)
    print("recorder")
    print(recorder)
    # TODO is it OK to comment this out
    # recorder.transform[2].init_stats(3)
    # recorder.transform[2].load_state_dict(transform_state_dict)
    recorder_obj = Recorder(
        record_frames=1000,
        frame_skip=cfg.env.frame_skip,
        policy_exploration=actor_model_explore,
        recorder=recorder,
        exploration_mode="mean",
        record_interval=record_interval,
    )
    return recorder_obj


def get_env_stats(cfg, init_env_steps=1000, seed=42):
    """Gets the stats of an environment."""
    proof_env = make_transformed_env(hydra.utils.instantiate(cfg.env, random=seed))
    print("proof_env")
    print(proof_env)
    # proof_env.set_seed(seed)
    # t = proof_env.transform[0]
    # t = proof_env.transform[-1]
    # t.init_stats(init_env_steps)
    # transform_state_dict = t.state_dict()
    transform_state_dict = proof_env.state_dict()
    proof_env.close()
    return transform_state_dict


def make_env(cfg: DictConfig):
    return hydra.utils.instantiate(cfg.env)


@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def train(cfg: DictConfig):
    # work_dir = Path.cwd()
    # timer = utils.Timer()
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Fetching the device that will be used throughout this notebook
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )

    if cfg.wandb.use_wandb:
        # Initialise WandB
        run = wandb.init(
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            name=cfg.wandb.run_name,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        )

    # Configure environment
    # train_env = make_env(cfg.env, device=device)
    train_env = hydra.utils.instantiate(cfg.env)
    train_env = make_transformed_env(train_env)
    # train_env.set_seed(cfg.random_seed)
    print("train_env")
    print(train_env)
    # cfg.action_dim = train_env.action_spec.shape
    # print("cfg.action_dim")
    # print(cfg.action_dim)
    # cfg.observation_dim = train_env.observation_spec
    # print(cfg.observation_dim)
    # train_env = dmc.make(
    #     cfg.env_name, cfg.task_name, cfg.frame_stack, cfg.action_repeat, cfg.random_seed
    # )
    # if torch.has_cuda and torch.cuda.device_count():
    #     train_env.to("cuda:0")
    #     train_env.reset()

    dynamic_model = hydra.utils.instantiate(cfg.model)
    # dynamic_model.gp_module = torch.compile(dynamic_model.gp_module)
    print("dynamic_model")
    print(dynamic_model)
    print(type(dynamic_model))
    planner = hydra.utils.instantiate(
        cfg.planner, model=dynamic_model, proof_env=make_transformed_env(make_env(cfg))
    )
    # proof_env = make_env(cfg)
    # print("proof_env")
    # print(proof_env)
    # planner = hydra.utils.instantiate(cfg.planner, proof_env=proof_env)
    print("planner")
    print(planner)
    # dynamic_model_module = TensorDictModule(
    #     dynamic_model,
    #     in_keys=["observation_vector", "action"],
    #     out_keys=[("next", "observation_vector")],
    # )

    # Configure agent
    # agent = hydra.utils.instantiate(cfg.agent, env=env)
    # agent = hydra.utils.instantiate(cfg.agent)
    # policy = hydra.utils.instantiate(cfg.policy)

    # Collect initial data set
    # replay_buffer = rollout_agent_and_populate_replay_buffer(
    #     env=train_env,
    #     policy=RandomPolicy(action_spec=train_env.action_spec),
    #     replay_buffer=ReplayBuffer(capacity=cfg.initial_dataset.replay_buffer_capacity),
    #     num_episodes=cfg.initial_dataset.num_episodes,
    # )
    # replay_buffer = ReplayBuffer(capacity=cfg.initial_dataset.replay_buffer_capacity)

    print("device={}".format(device))
    replay_buffer = make_replay_buffer(
        buffer_size=cfg.replay_buffer_capacity,
        device=device,
        buffer_scratch_dir=cfg.buffer_scratch_dir,
        prefetch=3,
    )
    logger = torchrl.record.loggers.WandbLogger(exp_name=cfg.wandb.run_name)

    # Create replay buffer

    # policy = RandomPolicy(action_spec=train_env.action_spec)
    print("train_env.action_spec")
    print(train_env.action_spec)
    policy = planner
    # policy = torchrl.collectors.collectors.RandomPolicy(train_env.action_spec)
    # policy_module = TensorDictModule(
    #     policy, in_keys=["observation"], out_keys=["action"]
    # )

    # recorder = torchrl.record.VideoRecorder(logger)
    transform_state_dict = get_env_stats(
        cfg, init_env_steps=cfg.init_env_steps, seed=cfg.random_seed + 1
    )
    # TODO should this use different seed?
    recorder = make_recorder(
        cfg,
        # actor_model_explore=policy_module,
        actor_model_explore=policy,
        transform_state_dict=transform_state_dict,
        record_interval=10,
        seed=42,
    )
    # recorder = torchrl.record.VideoRecorder(logger)
    # if cfg.save_video:
    #     video_recorder = VideoRecorder(work_dir)
    # if cfg.save_train_video:
    #     train_video_recorder = TrainVideoRecorder(work_dir)
    # torchrl.record.VideoRecorder(logger, tag, in_keys, skip: int = 2)

    frames_per_batch = 50
    total_frames = 50000 // cfg.env.frame_skip
    frames_per_batch = 1000 // cfg.env.frame_skip
    collector = SyncDataCollector(
        create_env_fn=partial(make_transformed_env, make_env(cfg=cfg)),
        policy=policy,
        total_frames=total_frames,
        # max_frames_per_traj=50,
        frames_per_batch=frames_per_batch,
        init_random_frames=-1,
        reset_at_each_iter=False,
        split_trajs=True,
        device="cpu",
        storing_device="cpu",
    )
    # collector = SyncDataCollector(
    #     create_env_fn=partial(make_env, cfg=cfg),
    #     # train_env,
    #     policy=policy_module,
    #     frames_per_batch=frames_per_batch,
    #     total_frames=total_frames,
    #     split_trajs=True,
    #     device=device,
    # )

    norm_factor_training = 1  # TODO set this correctly
    rewards, rewards_eval = [], []
    collected_frames = 0
    pbar = tqdm.tqdm(total=total_frames)
    r0 = None
    for i, tensordict in enumerate(collector):
        print("i={}".format(i))
        # print(tensordict)
        # print(tensordict["next"])

        # state_diff = tensordict.get("observation_vector") - tensordict.get(
        #     ("next", "observation_vector")
        # )
        # print("state_diff: {}".format(state_diff))
        # print(tensordict.get("observation_vector"))
        # print(tensordict.get(("next", "observation_vector")))
        # print(
        #     tensordict.get("observation_vector")[1:]
        #     == tensordict.get(("next", "observation_vector"))[:-1]
        # ).all()

        # print(tensordict["observation_vector"])
        if r0 is None:
            r0 = tensordict["reward"].mean().item()
        pbar.update(tensordict.numel())

        # extend the replay buffer with the new data
        if ("collector", "mask") in tensordict.keys(True):
            # if multi-step, a mask is present to help filter padded values
            # print("here")
            current_frames = tensordict["collector", "mask"].sum()
            tensordict = tensordict[tensordict.get(("collector", "mask"))]
        else:
            # print("Not here")
            tensordict = tensordict.view(-1)
            current_frames = tensordict.numel()
        collected_frames += current_frames
        replay_buffer.extend(tensordict.cpu())
        if i == 0:
            # Set inducing variables to data
            samples = replay_buffer.sample(len(replay_buffer))
            state = samples["observation_vector"]
            state_action_input = torch.concat([state, samples["action"]], -1)
            print("state_action_input.shape")
            print(state_action_input.shape)
            # Z = torch.nn.parameter.Parameter(state_action_input[:128, :])
            # print(Z.shape)
            # # data = (state_action_input, state_diff)
            # dynamic_model.gp_module.gp.variational_strategy.base_variational_strategy.inducing_points = (
            #     Z
            # )
            # print("Z")
            # print(Z)

        # logger.info("Training dynamic model")
        print("Training dynamic model")
        dynamic_model.train(replay_buffer)
        print("DONE TRAINING MODEL")
        # logger.info("DONE TRAINING MODEL")

        rewards.append(
            (
                i,
                tensordict["reward"].mean().item()
                / norm_factor_training
                / cfg.env.frame_skip,
            )
        )
        td_record = recorder(None)
        if td_record is not None:
            rewards_eval.append((i, td_record["r_evaluation"].item()))
        if len(rewards_eval):
            pbar.set_description(
                f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}) | reward eval: reward: {rewards_eval[-1][1]: 4.4f}"
            )
            wandb.log(
                {
                    # "reward": {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}),
                    "reward": rewards[-1][1],
                    "reward_eval": rewards_eval[-1][1],
                }
            )

    # global_step = 0
    # for episode in range(cfg.num_episodes):
    #     # Run the RL training loop
    #     episode_reward = 0
    #     # time_step = train_env.reset()
    #     tensordict = train_env.reset()

    #     # if cfg.save_train_video:
    #     #     train_video_recorder.init(time_step.observation)
    #     tensordict_rollout = train_env.rollout(
    #         max_steps=cfg.max_steps_per_episode, policy=policy_module
    #     )
    #     print("tensordict_rollout")
    #     print(tensordict_rollout)

    #     for episode_step in range(cfg.max_steps_per_episode):
    #         # while train_until_step(self.global_step):
    #         # while not time_step.last():
    #         # if time_step.last():
    #         if done:
    #             break

    #         # Sample action
    #         with torch.no_grad():
    #             action = policy(time_step.observation)
    #             # action = action.cpu().numpy()

    #         # Take env step
    #         next_obs, reward, done, info = train_env.step(action)
    #         episode_reward += reward
    #         # replay_storage.add(time_step)
    #         # print("time step")
    #         # print(time_step)
    #         replay_buffer.push(
    #             observation=obs,
    #             action=action,
    #             next_observation=next_obs,
    #             reward=reward,
    #             terminated=False,
    #             truncated=False,
    #             # discount=time_step.discount,
    #         )
    #         obs = next_obs

    #         # if cfg.save_train_video:
    #         #     train_video_recorder.record(time_step.observation)
    #         global_step += 1

    #         # if global_step % cfg.eval_every == 0:
    #         if cfg.use_wandb:
    #             # log stats
    #             elapsed_time, total_time = timer.reset()

    #             wandb.log({"episode_reward": episode_reward})
    #             # wandb.log({"episode_length": episode_step})
    #             wandb.log({"episode": episode})
    #             wandb.log({"step": global_step})

    #     # if cfg.save_train_video:
    #     #     train_video_recorder.save(f"{episode}.mp4")

    #     # Log stats
    #     elapsed_time, total_time = timer.reset()
    #     # wandb.log({"fps", episode_frame / elapsed_time})
    #     wandb.log({"total_time", total_time})
    #     wandb.log({"episode_reward", episode_reward})
    #     wandb.log({"episode_length", episode_step})
    #     wandb.log({"episode", episode})
    #     wandb.log({"buffer_size", len(replay_storage)})
    #     wandb.log({"step", global_step})

    #     # try to save snapshot
    #     # if self.cfg.save_snapshot:
    #     #     self.save_snapshot()

    #     # Train the agent ⚡
    #     logger.info("Training dynamic model...")
    #     # dynamic_model.train(replay_storage)
    #     logger.info("Done training dynamic model")
    #     # policy.train(replay_buffer, dynamic_model)


if __name__ == "__main__":
    train()  # pyright: ignore
