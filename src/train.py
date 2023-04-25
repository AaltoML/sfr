#!/usr/bin/env python3
import logging
import random
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import numpy as np
import omegaconf
import torch


torch.set_default_dtype(torch.float64)
import torchrl
import utils
import wandb
from dm_env import specs, StepType
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

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
    print("Using device: {}".format(cfg.device))
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
        / cfg.alg_name
        # / cfg.name
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
            # monitor_gym=True,
        )

    print("Making recorder")
    video_recorder = utils.VideoRecorder(work_dir) if cfg.save_video else None
    print("Made recorder")

    # Create replay buffer
    num_workers = 4
    print("Making replay buffer")
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
    print("Made replay buffer")

    agent = hydra.utils.instantiate(cfg.agent)
    print("Made agent")

    # elapsed_time, total_time = timer.reset()
    start_time = time.time()
    last_time = start_time
    global_step = 0
    for episode_idx in range(cfg.num_train_episodes):
        logger.info("Episode {} | Collecting data".format(episode_idx))
        # Collect trajectory
        time_step = env.reset()
        episode_reward = 0
        t = 0
        reset_updates = False
        while not time_step.last():
            if episode_idx < cfg.init_random_episodes:
                action = np.random.uniform(-1, 1, env.action_spec().shape).astype(
                    dtype=np.float64
                    # dtype=env.action_spec().dtype
                )
            else:
                if cfg.online_updates and t > 0:
                    # transition_data_new = (state_action_input, state_diff_output)
                    # reward_data_new = (state_action_input, reward_output)
                    # data_new = (state_action_input, state_diff_output, reward_output)
                    # agent.update(data_new)
                    # agent.transition_model.update(transition_data_new)
                    # agent.reward_model.update(reward_data_new)
                    if (
                        t % cfg.online_update_freq == 0
                    ):  # TODO uncomment this when updates are caching
                        data_new = (
                            state_action_inputs,
                            state_diff_outputs,
                            reward_outputs,
                        )
                        agent.update(data_new)
                        reset_updates = True
                    else:
                        reset_updates = False
                    #     # if cfg.online_updates and t > 1:
                    #     # transition_data_new = (state_action_inputs, state_diff_outputs)
                    #     # reward_data_new = (state_action_inputs, reward_outputs)
                    #     transition_data_new = (state_action_input, state_diff_output)
                    #     reward_data_new = (state_action_input, reward_output)
                    #     # data_new = {
                    #     #     "transition": transition_data_new,
                    #     #     "reward": reward_data_new,
                    #     # }
                    #     # print("USING new data")
                    #     agent.transition_model.update(transition_data_new)
                    #     agent.reward_model.update(reward_data_new)
                    # # else:
                    # #     data_new = {"transition": None, "reward": None}
                # else:
                # transition_data_new = None
                # reward_data_new = None
                # data_new = {"transition": None, "reward": None}
                # TODO data_new should only be one input
                # data_new = None
                action = agent.select_action(
                    time_step.observation,
                    eval_mode=False,
                    t0=time_step.step_type == StepType.FIRST,
                )
                action = action.cpu().numpy()
            # action = np.random.uniform(-1, 1, env.action_spec().shape).astype(
            #     dtype=env.action_spec().dtype
            # )

            # Create TensorDict for state transition to store in replay buffer
            time_step_td = TensorDict(
                {"state": time_step["observation"]}, batch_size=[], device=cfg.device
            )
            state = torch.Tensor(time_step["observation"]).to(cfg.device)

            time_step = env.step(action)

            reward_output = torch.Tensor([time_step["reward"]]).to(cfg.device)
            # print("reward_output {}".format(reward_output.shape))
            state_action_input = torch.concatenate(
                [state, torch.Tensor(time_step["action"]).to(cfg.device)], -1
            )[None, ...]
            state_diff_output = (
                torch.Tensor(time_step["observation"]).to(cfg.device) - state
            )[None, ...]
            if t == 0 or reset_updates:
                state_action_inputs = state_action_input
                state_diff_outputs = state_diff_output
                reward_outputs = reward_output
                # state_diff_reward_outputs = torch.concat([sts])
            else:
                reward_outputs = torch.concat([reward_outputs, reward_output], 0)
                state_action_inputs = torch.concat(
                    [state_action_inputs, state_action_input], 0
                )
                state_diff_outputs = torch.concat(
                    [state_diff_outputs, state_diff_output], 0
                )
            time_step_td.update(
                {
                    "action": time_step["action"],
                    "reward": time_step["reward"],
                    "next_state": time_step["observation"],
                }
            )
            for key in time_step_td.keys():
                time_step_td[key] = torch.as_tensor(
                    # time_step_td[key], device=cfg.device, dtype=torch.float32
                    time_step_td[key],
                    device=cfg.device,
                    dtype=torch.float64,
                )
            replay_buffer.add(time_step_td)

            global_step += 1
            episode_reward += time_step["reward"]
            t += 1

        logger.info("Finished collecting {} time steps".format(t))

        # Log training metrics
        env_step = global_step * cfg.env.action_repeat

        elapsed_time = time.time() - last_time
        total_time = time.time() - start_time
        last_time = time.time()
        train_metrics = {
            "episode": episode_idx,
            "step": global_step,
            "env_step": env_step,
            "episode_time": elapsed_time,
            "total_time": total_time,
            "episode_reward": np.mean(episode_reward),
        }
        logger.info(
            "TRAINING | Episode: {} | Reward: {}".format(episode_idx, episode_reward)
        )
        if cfg.wandb.use_wandb:
            wandb.log({"train/": train_metrics}, step=env_step)

        # Train agent
        # for _ in range(cfg.episode_length // cfg.update_every_steps):
        if episode_idx >= cfg.init_random_episodes:
            # logger.info("Training reward_model")
            # reward_model.train(replay_buffer)
            # logger.info("Training transition_model")
            # transition_model.train(replay_buffer)

            logger.info("Training agent")
            agent.train(replay_buffer)

            # Log rewards/videos in eval env
            # if episode_idx % cfg.eval_episode_freq == 0:
            #     # print("Evaluating {}".format(episode_idx))
            #     print("before G")
            #     Gs = utils.evaluate(
            #         eval_env,
            #         agent,
            #         episode_idx=episode_idx,
            #         # num_episode=cfg.eval_episode_freq,
            #         num_episodes=1,
            #         # num_episodes=10,
            #         # video=video_recorder,
            #     )
            #     print("after G")
            #     # print("DONE EVALUATING")
            #     eval_episode_reward = np.mean(Gs)
            #     env_step = global_step * cfg.env.action_repeat
            #     eval_metrics = {
            #         "episode": episode_idx,
            #         "step": global_step,
            #         "env_step": env_step,
            #         "episode_time": elapsed_time,
            #         "total_time": total_time,
            #         "episode_reward": eval_episode_reward,
            #     }
            #     logger.info(
            #         "EVAL | Episode: {} | Reward: {}".format(
            #             episode_idx, eval_episode_reward
            #         )
            #     )
            #     if cfg.wandb.use_wandb:
            #         wandb.log({"eval/": eval_metrics}, step=env_step)


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
