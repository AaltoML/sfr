#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig, OmegaConf


def train(transition_model, data):
    X, Y = data
    num_data = X.shape[0]
    transition_model.gp.train()
    transition_model.likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": transition_model.gp.parameters()},
            {"params": transition_model.likelihood.parameters()},
        ],
        lr=transition_model.learning_rate,
    )

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(
        transition_model.likelihood, transition_model.gp, num_data=num_data
    )
    logger.info("Data set size: {}".format(num_data))

    for i in range(transition_model.num_iterations):
        optimizer.zero_grad()

        latent = transition_model.gp(x)
        loss = -mll(latent, y)
        logger.info("Transition model iteration {}, Loss: {}".format(i, loss))
        loss.backward()
        optimizer.step()
        wandb.log({"model loss": loss})

    transition_model.gp.eval()
    transition_model.likelihood.eval()


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

        input_dim = cfg.state_dim + cfg.action_dim
        output_dim = cfg.state_dim
        X = torch.linspace(0, 10, 1000)
        Y = torch.sin(X) + 3 * torch.cos(X)
        data_old = (X, Y)

        num_inducing = cfg.model.transition_model.num_inducing
        samples = replay_buffer.sample(batch_size=num_inducing)
        state = samples["state_vector"]
        Z = torch.concat([state, samples["action"]], -1)
        Zs = []
        for _ in range(cfg.state_dim):
            Zs.append(torch.clone(Z))
        Z = torch.nn.parameter.Parameter(torch.stack(Zs, 0))
        print("Z.shape {}".format(Z.shape))
        print(
            "Zold: {}".format(
                dynamic_model.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points.shape
            )
        )
        dynamic_model.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points = (
            Z
        )

        # logger.info("Training dynamic model")
        print("Training dynamic model on Dold")
        # dynamic_model.transition_model.batch_size = len(replay_buffer)
        # for param in dynamic_model.transition_model.gp.mean_module.parameters():
        #     param.requires_grad = False
        # for param in dynamic_model.transition_model.gp.covar_module.parameters():
        #     param.requires_grad = False
        # for param in dynamic_model.transition_model.likelihood.parameters():
        #     param.requires_grad = False
        print(
            "dynamic_model.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points"
        )
        print(
            dynamic_model.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points.requires_grad
        )
        # dynamic_model.transition_model.train(replay_buffer)

        # dynamic_model_new.transition_model = dynamic_model.transition_model.make_copy()
        # # likelihood = deepcopy(dynamic_model.transition_model.likelihood)
        # # mean_module = deepcopy(dynamic_model.transition_model.gp.mean_module)
        # # covar_module = deepcopy(dynamic_model.transition_model.gp.covar_module)
        # for param in dynamic_model.transition_model.gp.mean_module.parameters():
        #     param.requires_grad = False
        # for param in dynamic_model.transition_model.gp.covar_module.parameters():
        #     param.requires_grad = False
        # for param in dynamic_model.transition_model.likelihood.parameters():
        #     param.requires_grad = False
        print("Training dynamic model on Dold+Dnew")
        # dynamic_model_new.transition_model = SVGPTransitionModel(
        #     likelihood=likelihood,
        #     mean_module=mean_module,
        #     covar_module=covar_module,
        #     num_inducing=cfg.model.transition_model.num_inducing,
        #     learning_rate=dynamic_model.transition_model.learning_rate,
        #     num_iterations=dynamic_model.transition_model.num_iterations,
        #     delta_state=dynamic_model.transition_model.delta_state,
        #     num_workers=dynamic_model.transition_model.num_workers,
        #     # learn_inducing_locations=dynamic_model.transition_model.learn_inducing_locations,
        #     learn_inducing_locations=cfg.model.transition_model.learn_inducing_locations,
        # )
        # dynamic_model_new.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points = deepcopy(
        #     dynamic_model.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points
        # )
        # dynamic_model_new.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points.requires_grad = (
        #     False
        # )
        # print(
        #     "dynamic_model_new.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points"
        # )
        # print(
        #     dynamic_model_new.transition_model.gp.variational_strategy.base_variational_strategy.inducing_points.requires_grad
        # )
        dynamic_model_new.transition_model.batch_size = len(replay_buffer_new)
        dynamic_model_new.transition_model.train(replay_buffer_new)
        # dynamic_model_new.transition_model.batch_size = len(replay_buffer_new_and_old)
        # dynamic_model_new.transition_model.train(replay_buffer_new_and_old)

        print("Fast update on Dold with Dnew")
        # dynamic_model.transition_model.batch_size = len(replay_buffer_new)
        samples = replay_buffer_new.sample(batch_size=len(replay_buffer_new))
        X = torch.concat([samples["state_vector"], samples["action"]], -1)
        Y = samples["next"]["state_vector"] - samples["state_vector"]
        dynamic_model.transition_model.forward(Xtest, data_new=(X, Y))

        # dynamic_model.transition_model.train(replay_buffer_new)

    train(cfg)


if __name__ == "__main__":
    train_on_cluster()  # pyright: ignore
