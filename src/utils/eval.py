#!/usr/bin/env python3

from typing import Optional

from .video import VideoRecorder
import numpy as np


def evaluate(
    env,
    agent,
    episode_idx: int = 0,
    num_episodes: int = 10,
    video: Optional[VideoRecorder] = None,
):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []

    for i in range(num_episodes):
        episode_reward = 0
        time_step = env.reset()
        if video:
            video.init(env, enabled=(i == 0))
        while not time_step.last():
            action = agent.select_action(
                time_step.observation, eval_mode=True, t0=time_step.first()
            )

            time_step = env.step(action.cpu().numpy())

            episode_reward += time_step.reward
            if video:
                video.record(env)
        episode_rewards.append(episode_reward)
        if i == 0:
            if video:
                # print("saving video {}, {}".format(i, episode_idx))
                video.save(episode_idx)
    return np.nanmean(episode_rewards)
