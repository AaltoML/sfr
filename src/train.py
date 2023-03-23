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
import imageio
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchrl
import tqdm
import wandb
from dm_env import specs

# from models.gp.svgp import SVGPTransitionModel
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
from torchrl.envs.transforms import ObservationNorm, RewardScaling, TransformedEnv
from torchrl.modules import CEMPlanner
from torchrl.record import VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.trainers import Recorder
from utils import set_seed_everywhere
from models import GaussianModelBaseEnv


def make_replay_buffer(
    buffer_size: int,
    device,
    buffer_scratch_dir: str = "/tmp/",
    batch_size: int = 64,
    prefetch: int = 3,
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
        prefetch=prefetch,
    )
    return replay_buffer


def make_transformed_env(env, reward_scaling=1.0, init_env_steps: int = 1000):
    """Apply transforms to the env (such as reward scaling and state normalization)."""

    double_to_float_list = []
    double_to_float_inv_list = []
    if isinstance(env, torchrl.envs.libs.dm_control.DMControlEnv):
        logger.info("env is dm_env so making double precision")
        # if env_library is torchrl.envs.libs.dm_control.DMControlEnv:
        # DMControl requires double-precision
        double_to_float_list += ["reward", "action"]
        double_to_float_inv_list += ["action"]

    # out_key = "state_vector"
    out_key = "state_vector"
    obs_keys = list(env.observation_spec.keys())
    print("obs_keys {}".format(obs_keys))
    # try:
    #     obs_keys.remove("pixels")
    # except:
    #     print("pixels not in obs")
    # print("obs_keys after {}".format(obs_keys))

    double_to_float_list.append(out_key)
    # obs_action_keys = obs_keys + ["action"]
    # print("obs_action_keys")
    # print(obs_action_keys)
    # print(obs_diff_keys)
    env = TransformedEnv(
        env,
        torchrl.envs.transforms.Compose(
            RewardScaling(loc=0.0, scale=reward_scaling),
            CatTensors(in_keys=obs_keys, out_key=out_key),
            ObservationNorm(in_keys=[out_key]),
            DoubleToFloat(
                in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
            ),
            # CatTensors(in_keys=obs_keys + ["action"], out_key="obs_action_vector"),
        ),
    )
    print("original env")
    print(env)

    env.transform[2].init_stats(init_env_steps)
    # try:
    #     env.transform.insert(0, CatTensors(["pixels"], "pixels_save", del_keys=False))
    #     print(env)
    #     print("removed pixels")
    # except:
    #     print("No pixels to remove")
    # print("env.observation_spec.keys()")
    # print(env.observation_spec.keys())

    # we append transforms one by one, although we might as well create the
    # transformed environment using the `env = TransformedEnv(base_env, transforms)`
    # syntax.
    # env.append_transform(RewardScaling(loc=0.0, scale=reward_scaling))

    # We concatenate all states into a single "state_vector"
    # even if there is a single tensor, it'll be renamed in "state_vector".
    # This facilitates the downstream operations as we know the name of the
    # output tensor.
    # In some environments (not half-cheetah), there may be more than one
    # observation vector: in this case this code snippet will concatenate them
    # all.
    # out_key = "state_vector"
    # obs_keys = list(env.observation_spec.keys())
    # print("obs_keys")
    # print(obs_keys)
    # try:
    #     obs_keys.remove("pixels")
    # except:
    #     print("pixels not in obs")
    # print(obs_keys)
    # env.append_transform(CatTensors(in_keys=obs_keys, out_key=out_key))

    # we normalize the states, but for now let's just instantiate a stateless
    # version of the transform
    # env.append_transform(ObservationNorm(in_keys=[out_key], standard_normal=True))
    # env.append_transform(ObservationNorm(in_keys=[out_key]))

    # double_to_float_list.append(out_key)
    # env.append_transform(
    #     DoubleToFloat(
    #         in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
    #     )
    # )
    # env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    return env


def make_recorder(
    cfg: DictConfig,
    actor_model_explore,
    transform_state_dict,
    logger,
    record_interval=10,
    seed=42,
):
    # base_env = hydra.utils.instantiate(
    #     cfg.env, random=seed, from_pixels=True, pixels_only=False
    # )
    # base_env = make_env(cfg)
    env = make_transformed_env(make_env(cfg))
    env.append_transform(VideoRecorder(logger=logger, tag="video"))
    # print("recorder")
    # print(recorder)
    # recorder.transform.insert(0, CatTensors(["pixels"], "pixels_save", del_keys=False))
    # recorder.append_transform(CatTensors(["pixels"], "pixels_save", del_keys=False))
    print("recorder env")
    print(env)
    # TODO is it OK to comment this out
    # recorder.transform[2].init_stats(3)
    # env.transform[2].load_state_dict(transform_state_dict)
    # env.transform[2].load_state_dict(transform_state_dict)
    env.transform.load_state_dict(transform_state_dict)
    recorder = Recorder(
        record_frames=1000,
        frame_skip=cfg.env.frame_skip,
        policy_exploration=actor_model_explore,
        recorder=env,
        # logger=logger,
        exploration_mode="mean",
        record_interval=record_interval,
    )
    return recorder


def get_env_stats(cfg, seed: Optional[int] = None):
    """Gets the stats of an environment."""
    if seed is not None:
        proof_env = make_transformed_env(make_env(cfg, seed=seed))
    else:
        proof_env = make_transformed_env(make_env(cfg))
    print("proof_env {}".format(proof_env))
    transform_state_dict = proof_env.state_dict()
    proof_env.close()
    return transform_state_dict


def make_env(cfg: DictConfig, seed: int = None):
    if seed is not None:
        return hydra.utils.instantiate(
            cfg.env,
            random=seed
            # cfg.env, from_pixels=True, pixels_only=False, random=seed
        )
    else:
        return hydra.utils.instantiate(cfg.env)
        # return hydra.utils.instantiate(cfg.env, from_pixels=True, pixels_only=False)


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

    dynamic_model = hydra.utils.instantiate(cfg.model)
    print("Dynamic model {}".format(dynamic_model))

    # Configure agent
    planner = hydra.utils.instantiate(cfg.planner, env=dynamic_model)
    print("Planner: {}".format(planner))
    random_policy = torchrl.collectors.collectors.RandomPolicy(
        make_transformed_env(make_env(cfg)).action_spec
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        buffer_size=cfg.replay_buffer_capacity,
        device=device,
        buffer_scratch_dir=cfg.buffer_scratch_dir,
        batch_size=cfg.batch_size,
        prefetch=3,
    )

    transform_state_dict = get_env_stats(cfg, seed=cfg.random_seed)
    # TODO should this use different seed?
    recorder = make_recorder(
        cfg,
        actor_model_explore=planner,
        transform_state_dict=transform_state_dict,
        logger=logger,
        record_interval=10,
        seed=42,
    )

    frames_per_batch = 50
    total_frames = 50000 // cfg.env.frame_skip
    frames_per_batch = 1000 // cfg.env.frame_skip
    collector = SyncDataCollector(
        # create_env_fn=partial(make_transformed_env, train_env),
        create_env_fn=partial(make_transformed_env, make_env(cfg=cfg)),
        # policy=planner,
        policy=random_policy,
        total_frames=total_frames,
        # max_frames_per_traj=50,
        frames_per_batch=frames_per_batch,
        # init_random_frames=-1,
        init_random_frames=500,
        reset_at_each_iter=False,
        split_trajs=True,
        device=device,
        storing_device=device,
        # device="cpu",
        # storing_device="cpu",
    )

    # # logger.info("Training dynamic model")
    # print("Training dynamic model")
    # dynamic_model.train(replay_buffer)
    # print("DONE TRAINING MODEL")
    # # logger.info("DONE TRAINING MODEL")

    norm_factor_training = 1  # TODO set this correctly
    rewards, rewards_eval = [], []
    collected_frames = 0
    # pbar = tqdm.tqdm(total=total_frames)
    r0 = None
    for i, tensordict in enumerate(collector):
        print("Episode: {}".format(i))
        print("tensordict: {}".format(tensordict))
        # print(tensordict["next"])

        # state_diff = tensordict.get("state_vector") - tensordict.get(
        #     ("next", "state_vector")
        # )
        # print("state_diff: {}".format(state_diff))
        # print(tensordict.get("state_vector"))
        # print(tensordict.get(("next", "state_vector")))
        # print(
        #     tensordict.get("state_vector")[1:]
        #     == tensordict.get(("next", "state_vector"))[:-1]
        # ).all()

        # print(tensordict["state_vector"])
        if r0 is None:
            r0 = tensordict["reward"].mean().item()
        # pbar.update(tensordict.numel())

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
        print("current_frames")
        print(current_frames)
        collected_frames += current_frames
        print(current_frames)
        replay_buffer.extend(tensordict.cpu())
        # replay_buffer.extend(tensordict)
        print("replay_buffer")
        print(replay_buffer)
        # if i == 0:
        #     # Set inducing variables to data
        #     samples = replay_buffer.sample(len(replay_buffer))
        #     state = samples["state_vector"]
        #     state_action_input = torch.concat([state, samples["action"]], -1)
        #     print("state_action_input.shape")
        #     print(state_action_input.shape)
        #     print("cfg.model.num_inducing")
        #     print(cfg.model.num_inducing)
        #     Z = torch.nn.parameter.Parameter(
        #         state_action_input[: cfg.model.num_inducing, :]
        #     )
        #     print("Z.shape")
        #     print(Z.shape)
        #     # # data = (state_action_input, state_diff)
        #     # dynamic_model.gp_module.gp.variational_strategy.base_variational_strategy.inducing_points = (
        #     #     Z
        #     # )

        # logger.info("Training dynamic model")
        print("Training dynamic model")
        # dynamic_model.train(replay_buffer)
        dynamic_model.transition_model.train(replay_buffer)
        dynamic_model.reward_model.train(replay_buffer)
        print("DONE TRAINING MODEL")

        # logger.info("DONE TRAINING MODEL")
        import torch.nn.functional as F

        class FakeReward(nn.Module):
            def __init__(self):
                super().__init__()
                # self.conv1 = nn.Conv2d(1, 20, 5)
                # self.conv2 = nn.Conv2d(20, 20, 5)
                self.conv1 = nn.Linear(5, 1)
                self.conv2 = nn.Linear(5, 1)
                self.conv3 = nn.Linear(5, 1)

            def forward(self, a, b, c):
                # print("a {}".format(a))
                # print("b {}".format(b))
                # print("c {}".format(c))
                # x = F.relu(self.conv1(a))
                # y = F.relu(self.conv1(b))
                y = dynamic_model.reward_model(a, b, c)
                # print("y {}".format(y.shape))
                return y

        # class FakeReward(nn.Module):
        #     def __init__(self):
        #         super().__init__()

        #     # def forward(self, a, b, c):
        #     def __call__(self, a):
        #         return torch.sum(a * b * c)

        #     def forward(self, x):
        #         """Returns change in state"""
        #         return 1

        print("type(nn.Linear(5, 1))")
        print(type(nn.Linear(5, 1)))
        print(type(FakeReward()))
        env = make_transformed_env(make_env(cfg=cfg))
        model_env = GaussianModelBaseEnv(
            transition_model=dynamic_model.transition_model,
            # reward_model=dynamic_model.transition_model,
            # reward_model=nn.Linear(5, 1),
            # reward_model=dynamic_model.reward_model,
            reward_model=FakeReward(),
            # reward_model=TensorDictModule(
            #     FakeReward(),
            #     # fake_reward,
            #     in_keys=["state_vector", "state_vector_var", "noise_var"],
            #     out_keys=["expected_reward"],
            # ),
            state_size=5,
            action_size=1,
            device=device,
            # dtype=None,
            # batch_size: int = None,
        )
        planner = CEMPlanner(
            model_env,
            planning_horizon=5,
            optim_steps=11,
            num_candidates=7,
            top_k=3,
            reward_key="reward",
            # reward_key="expected_reward",
            action_key="action",
        )
        tensordict = env.rollout(max_steps=350, policy=planner)
        print("CEM")
        print(tensordict)

        rewards.append(
            (
                i,
                tensordict["reward"].mean().item()
                / norm_factor_training
                / cfg.env.frame_skip,
            )
        )
        logger.log_scalar(name="reward", value=rewards[-1][1], step=i)

        # td_record = recorder(None)

        # if td_record is not None:
        #     rewards_eval.append((i, td_record["r_evaluation"].item()))
        # if len(rewards_eval):
        #     pbar.set_description(
        #         f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}) | reward eval: reward: {rewards_eval[-1][1]: 4.4f}"
        #     )
        #     logger.log_scalar(name="reward", value=rewards[-1][1], step=i)
        #     logger.log_scalar(name="reward_eval", value=rewards_eval[-1][1], step=i)
        #     # wandb.log(
        #     #     {
        #     #         "gameplays": wandb.Video(
        #     #             mp4,
        #     #             caption="episode: " + str(i - 10),
        #     #             fps=4,
        #     #             format="gif",
        #     #         ),
        #     #         "step": i,
        #     #     }
        #     # )

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
