#!/usr/bin/env python3
import logging
import random
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tensordict import TensorDict
import hydra
import numpy as np
import omegaconf
import torch
import torchrl
import utils
import wandb
from dm_env import specs
from omegaconf import DictConfig, OmegaConf

# from src.utils.buffer import ReplayBuffer
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
    cfg.state_dim = cfg.state_dim[0]
    cfg.action_dim = tuple(int(x) for x in env.action_spec().shape)
    cfg.action_dim = cfg.action_dim[0]
    cfg.input_dim = cfg.state_dim + cfg.action_dim
    cfg.output_dim = cfg.state_dim

    ###### Set up workspace ######
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

    video_recorder = utils.VideoRecorder(work_dir) if cfg.save_video else None

    # Create replay buffer
    num_workers = 4
    replay_buffer = torchrl.data.TensorDictReplayBuffer(
        storage=torchrl.data.replay_buffers.LazyTensorStorage(
            int(num_train_steps) // max(1, num_workers), device=cfg.device
        ),
        # storage=LazyMemmapStorage(
        #     buffer_size,
        #     scratch_dir=buffer_scratch_dir,
        #     device=device,
        # ),
        batch_size=cfg.batch_size,
        sampler=torchrl.data.replay_buffers.RandomSampler(),
        pin_memory=False,
        # prefetch=prefetch,
    )

    transition_model = hydra.utils.instantiate(cfg.transition_model)
    reward_model = hydra.utils.instantiate(cfg.reward_model)
    agent = hydra.utils.instantiate(cfg.agent)

    global_step = 0
    for episode_idx in range(cfg.num_train_episodes):
        # Collect trajectory
        time_step = env.reset()
        episode_reward = 0
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

            # Create TensorDict for state transition to store inreplay buffer
            time_step_td = TensorDict(
                {"state": time_step["observation"]}, batch_size=[], device=cfg.device
            )

            time_step = env.step(action)

            time_step_td.update(
                {
                    "action": time_step["action"],
                    "reward": time_step["reward"],
                    "next_state": time_step["observation"],
                }
            )
            for key in time_step_td.keys():
                time_step_td[key] = torch.as_tensor(
                    time_step_td[key], device=cfg.device, dtype=torch.float32
                )
            replay_buffer.add(time_step_td)

            global_step += 1

        # Train agent
        # for _ in range(cfg.episode_length // cfg.update_every_steps):
        if episode_idx >= cfg.init_random_episodes:
            logger.info("Training reward_model")
            reward_model.train(replay_buffer)
            logger.info("Training transition_model")
            transition_model.train(replay_buffer)

            logger.info("Training agent")
            agent.train(replay_buffer)

        # Log rewards/videos in eval env
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
            logger.info("Episode: {} | Reward: {}".format(episode_idx, episode_reward))
            if cfg.wandb.use_wandb:
                wandb.log({"eval/": eval_metrics}, step=env_step)


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
