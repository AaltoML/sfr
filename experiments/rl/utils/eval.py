#!/usr/bin/env python3
import logging
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import torch
from dm_env import StepType
from src.rl.agents import Agent
from tensordict import TensorDict

# from torchrl.data import ReplayBuffer
from src.rl.utils.buffer import ReplayBuffer

from .video import VideoRecorder


def rollout(
    env,
    agent: Agent,
    online_updates: bool = False,
    online_update_freq: int = 1,
    eval_mode: bool = True,
    replay_buffer: Optional[ReplayBuffer] = None,
    video: Optional[VideoRecorder] = None,
    device: str = "cuda",
):
    t = 0
    reset_updates = False
    time_step = env.reset()
    episode_reward = 0
    while not time_step.last():
        if online_updates and t > 0:
            if (
                t % online_update_freq == 0
            ):  # TODO uncomment this when updates are caching
                data_new = (
                    state_action_inputs,
                    state_diff_outputs,
                    reward_outputs,
                )
                logger.info("Updating mode at t={}".format(t))
                agent.update(data_new)
                logger.info("Finished updating models")
                reset_updates = True
            else:
                reset_updates = False
        # TODO data_new should only be one input
        # data_new = None
        action = agent.select_action(
            time_step.observation,
            eval_mode=eval_mode,
            t0=time_step.step_type == StepType.FIRST,
        )
        action = action.cpu().numpy()

        # Create TensorDict for state transition to store in replay buffer
        if replay_buffer:
            time_step_td = TensorDict(
                {"state": time_step["observation"]}, batch_size=[], device=cfg.device
            )
        if online_updates:
            state = torch.Tensor(time_step["observation"]).to(device)

        time_step = env.step(action)

        if online_updates:
            reward_output = torch.Tensor([time_step["reward"]]).to(device)
            # print("reward_output {}".format(reward_output.shape))
            action_input = torch.Tensor(time_step["action"]).to(device)
            state_action_input = torch.concatenate(
                [state, torch.Tensor(time_step["action"]).to(device)], -1
            )[None, ...]
            state_diff_output = (
                torch.Tensor(time_step["observation"]).to(device) - state
            )[None, ...]
            if t == 0 or reset_updates:
                state_action_inputs = state_action_input
                state_diff_outputs = state_diff_output
                reward_outputs = reward_output
                state_action_inputs_all = state_action_input
                state_diff_outputs_all = state_diff_output
                reward_outputs_all = reward_output
                # state_diff_reward_outputs = torch.concat([sts])
            else:
                reward_outputs = torch.concat([reward_outputs, reward_output], 0)
                state_action_inputs = torch.concat(
                    [state_action_inputs, state_action_input], 0
                )
                state_diff_outputs = torch.concat(
                    [state_diff_outputs, state_diff_output], 0
                )
            reward_outputs_all = torch.concat([reward_outputs_all, reward_output], 0)
            state_action_inputs_all = torch.concat(
                [state_action_inputs_all, state_action_input], 0
            )
            state_diff_outputs_all = torch.concat(
                [state_diff_outputs_all, state_diff_output], 0
            )
        if replay_buffer:
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

        episode_reward += time_step.reward

        if video:
            video.record(env)
    return episode_reward


def evaluate(
    env,
    agent,
    episode_idx: int = 0,
    num_episodes: int = 10,
    online_updates: bool = False,
    online_update_freq: int = 1,
    video: Optional[VideoRecorder] = None,
    device: str = "cuda",
):
    """Evaluate a trained agent and optionally save a video."""
    episode_returns = []
    for i in range(num_episodes):
        # time_step = env.reset()
        if video:
            video.init(env, enabled=(i == 0))
        episode_returns.append(
            rollout(
                env=env,
                agent=agent,
                online_updates=online_updates,
                online_update_freq=online_update_freq,
                eval_mode=True,
                video=video,
                device=device,
            )
        )
        logger.info(
            "Eval episode {}/{}, G={}".format(i + 1, num_episodes, episode_returns[-1])
        )
        # while not time_step.last():
        #     # TODO add data_new here
        #     action = agent.select_action(
        #         time_step.observation, eval_mode=True, t0=time_step.first()
        #     )

        #     time_step = env.step(action.cpu().numpy())

        #     episode_reward += time_step.reward

        #     if video:
        #         video.record(env)
        # episode_rewards.append(episode_reward)
        if i == 0:
            if video:
                # print("saving video {}, {}".format(i, episode_idx))
                video.save(episode_idx)
    return np.nanmean(episode_returns)
