#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def train_on_cluster(cfg: DictConfig):
    """import here so hydra's --multirun works with slurm"""
    import logging
    import random
    from copy import deepcopy
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
    from models import GaussianModelBaseEnv
    from models.gp import SVGPTransitionModel
    from models.reward.gp import GPRewardModel
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
    from torchrl.data.replay_buffers.storages import (
        LazyMemmapStorage,
        LazyTensorStorage,
    )
    from torchrl.envs import CatTensors, DoubleToFloat
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.envs.transforms import ObservationNorm, RewardScaling, TransformedEnv
    from torchrl.modules import CEMPlanner
    from torchrl.record import VideoRecorder
    from torchrl.record.loggers.wandb import WandbLogger
    from torchrl.trainers import Recorder
    from utils import set_seed_everywhere

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
                # CatTensors(in_keys=obs_keys + ["action"], out_key="obs_action_vector"),
            ),
        )
        print("original env")
        print(env)

        env.transform[2].init_stats(init_env_steps)
        return env

    def make_recorder(
        cfg: DictConfig,
        actor_model_explore,
        transform_state_dict,
        logger,
        record_interval=10,
        seed=42,
    ):
        env = make_transformed_env(make_env(cfg))
        print("recorder env")
        print(env)
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

    def make_env(
        cfg: DictConfig,
        seed: int = None,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ):
        if seed is not None:
            return hydra.utils.instantiate(
                cfg.env,
                random=seed
                # cfg.env, from_pixels=True, pixels_only=False, random=seed
            )
        else:
            return hydra.utils.instantiate(cfg.env)
            # return hydra.utils.instantiate(cfg.env, from_pixels=True, pixels_only=False)

    # @hydra.main(version_base="1.3", config_path="../configs", config_name="main")
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
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
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
        dynamic_model_new = hydra.utils.instantiate(cfg.model)
        print("Dynamic model {}".format(dynamic_model))

        # Configure agent
        planner = hydra.utils.instantiate(cfg.planner, env=dynamic_model)
        print("Planner: {}".format(planner))
        random_policy = torchrl.collectors.collectors.RandomPolicy(
            make_transformed_env(make_env(cfg)).action_spec
        )

        import torch.nn.functional as F

        class FakeReward(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Linear(5, 1)
                self.conv2 = nn.Linear(5, 1)
                self.conv3 = nn.Linear(5, 1)

            def forward(self, a, b, c):
                y = dynamic_model.reward_model(a, b, c)
                return y

        env = make_transformed_env(make_env(cfg=cfg))
        model_env = GaussianModelBaseEnv(
            transition_model=dynamic_model.transition_model,
            reward_model=FakeReward(),
            state_size=5,
            action_size=1,
            device=device,
        )
        print("making planner")
        planner = CEMPlanner(
            model_env,
            planning_horizon=2,
            optim_steps=5,
            # optim_steps=11,
            num_candidates=5,
            top_k=3,
            reward_key="reward",
            # reward_key="expected_reward",
            action_key="action",
        )
        print("MADE planner")

        # Create replay buffer
        replay_buffer = make_replay_buffer(
            buffer_size=cfg.replay_buffer_capacity,
            device=device,
            buffer_scratch_dir=cfg.buffer_scratch_dir,
            # batch_size=cfg.batch_size,
            # prefetch=3,
        )
        replay_buffer_new = make_replay_buffer(
            buffer_size=cfg.replay_buffer_capacity,
            device=device,
            buffer_scratch_dir="/tmp/new"
            # batch_size=cfg.batch_size,
            # prefetch=3,
        )
        replay_buffer_new_and_old = make_replay_buffer(
            buffer_size=cfg.replay_buffer_capacity,
            device=device,
            buffer_scratch_dir="/tmp/new_old"
            # batch_size=cfg.batch_size,
            # prefetch=3,
        )

        transform_state_dict = get_env_stats(cfg, seed=cfg.random_seed)

        # frames_per_batch = 50
        total_frames = 50000 // cfg.env.frame_skip
        frames_per_batch = 1000 // cfg.env.frame_skip
        collector = SyncDataCollector(
            # create_env_fn=partial(make_transformed_env, train_env),
            create_env_fn=partial(make_transformed_env, make_env(cfg=cfg)),
            policy=planner,
            # policy=random_policy,
            total_frames=total_frames,
            # max_frames_per_traj=50,
            frames_per_batch=frames_per_batch,
            init_random_frames=-1,
            # init_random_frames=500,
            reset_at_each_iter=False,
            # split_trajs=True,
            split_trajs=False,
            device=device,
            storing_device=device,
            # device="cpu",
            # storing_device="cpu",
        )

        norm_factor_training = 1  # TODO set this correctly
        rewards, rewards_eval = [], []
        collected_frames = 0
        r0 = None
        for i, tensordict in enumerate(collector):
            print("Episode: {}".format(i))
            print("tensordict: {}".format(tensordict))

            if ("collector", "mask") in tensordict.keys(True):
                current_frames = tensordict["collector", "mask"].sum()
                tensordict = tensordict[tensordict.get(("collector", "mask"))]
            else:
                tensordict = tensordict.view(-1)
                current_frames = tensordict.numel()
            collected_frames += current_frames
            if i == 0:
                print("making Dold")
                replay_buffer.extend(tensordict.cpu())
            if i == 1:
                print("making Dnew")
                replay_buffer_new.extend(tensordict.cpu())
            replay_buffer_new_and_old.extend(tensordict.cpu())
            print("making Dold+Dnew")
            if i == 2:
                break

        # logger.info("Training dynamic model")
        print("Training dynamic model on Dold")
        dynamic_model.transition_model.batch_size = len(replay_buffer)
        dynamic_model.transition_model.train(replay_buffer)

        print("Training dynamic model on Dold+Dnew")
        dynamic_model_new.transition_model.batch_size = len(replay_buffer_new_and_old)
        dynamic_model_new.transition_model.train(replay_buffer_new_and_old)

        print("Fast update on Dold with Dnew")
        # dynamic_model.transition_model.batch_size = len(replay_buffer_new)
        samples = replay_buffer_new.sample(batch_size=len(replay_buffer_new))
        X = torch.concat([samples["state_vector"], samples["action"]], -1)
        Y = samples["state_vector"] - samples["next"]["state_vector"]
        dynamic_model.transition_model.forward(Xtest, data_new=(X, Y))

        # dynamic_model.transition_model.train(replay_buffer_new)

    train(cfg)


if __name__ == "__main__":
    train_on_cluster()  # pyright: ignore
