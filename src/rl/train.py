#!/usr/bin/env python3
import logging
import random
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch


torch.set_default_dtype(torch.float64)

import src
import torchrl
import wandb
from dm_env import specs, StepType
from omegaconf import DictConfig, OmegaConf
from src.rl.utils import EarlyStopper, set_seed_everywhere
from tensordict import TensorDict


@hydra.main(version_base="1.3", config_path="./configs", config_name="main")
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
    video_recorder = src.rl.utils.VideoRecorder(work_dir) if cfg.save_video else None
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

    transition_model = hydra.utils.instantiate(cfg.agent.transition_model)
    # svgp = hydra.utils.instantiate(cfg.agent.reward_model.svgp)
    # reward_model = hydra.utils.instantiate(cfg.agent.reward_model, svgp=svgp)
    reward_model = hydra.utils.instantiate(cfg.agent.reward_model)
    agent = hydra.utils.instantiate(
        cfg.agent, reward_model=reward_model, transition_model=transition_model
    )
    # agent = hydra.utils.instantiate(cfg.agent)
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
            # logger.info("Timestep: {}".format(t))
            if episode_idx <= cfg.init_random_episodes:
                action = np.random.uniform(-1, 1, env.action_spec().shape).astype(
                    dtype=np.float64
                    # dtype=env.action_spec().dtype
                )
            else:
                # if t > 10:
                #     break
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
                        # print(
                        #     "state_action_inputs {}".format(state_action_inputs.shape)
                        # )
                        # print("state_diff_outputs {}".format(state_diff_outputs.shape))
                        # print("reward_outputs {}".format(reward_outputs.shape))
                        logger.info("Updating model at t={}".format(t))
                        agent.update(data_new)
                        logger.info("Finished updating models")
                        reset_updates = True
                    else:
                        reset_updates = False
                        logger.info("Not updating models")
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
            action_input = torch.Tensor(time_step["action"]).to(cfg.device)
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

            # # Z=reward_model.predict(state=s)
            # X_new = state_action_inputs_all
            # print("X_new {}".format(X_new.shape))
            # Y_new = reward_outputs_all
            # print("Y_new {}".format(Y_new.shape))
            # Y = reward_model.predict(
            #     state=state_action_inputs_all[:, 0:5],
            #     action=state_action_inputs_all[:, 5:],
            # )
            # mean_new = Y.reward_mean
            # print("mean_new {}".format(mean_new.shape))
            # var_new = Y.reward_var
            # print("var_new {}".format(var_new.shape))
            # # X_test = torch.linspace(-10, 10, 1000)
            # # X_test = torch.concat([state[]])
            # X_test = state_action_inputs_all
            # print("X_test {}".format(X_test.shape))
            # Z = svgp.variational_strategy.inducing_points.detach()
            # print("Z {}".format(Z.shape))

            # def plot(i):
            #     # plt.scatter(
            #     #     Z[:, 0],
            #     #     np.zeros_like(Z[:, 0]),
            #     #     color="k",
            #     #     marker="|",
            #     #     alpha=0.6,
            #     #     label="Z",
            #     # )
            #     plt.scatter(
            #         X_new[:, 0],
            #         Y_new,
            #         color="c",
            #         marker="o",
            #         alpha=0.6,
            #         label="New data",
            #     )
            #     # plt.scatter(
            #     #     X_new_2,
            #     #     Y_new_2[:, i],
            #     #     color="r",
            #     #     marker="o",
            #     #     alpha=0.6,
            #     #     label="New data",
            #     # )

            #     # plt.plot(
            #     #     X_test[:, 0],
            #     #     mean.detach()[:, i],
            #     #     color="m",
            #     #     label=r"$\mu_{old}(\cdot)$",
            #     # )
            #     # plt.fill_between(
            #     #     X_test[:, 0],
            #     #     mean[:, i] - 1.98 * torch.sqrt(var[:, i]),
            #     #     # pred.mean[:, 0],
            #     #     mean[:, i] + 1.98 * torch.sqrt(var[:, i]),
            #     #     color="m",
            #     #     alpha=0.2,
            #     #     label=r"$\mu_{old}(\cdot) \pm 1.98\sigma_{old}(\cdot)$",
            #     # )

            #     plt.plot(
            #         X_test[:, 0],
            #         mean_new.detach(),
            #         color="c",
            #         label=r"$\mu_{new}(\cdot)$",
            #     )
            #     plt.fill_between(
            #         X_test[:, 0],
            #         mean_new - 1.98 * torch.sqrt(var_new),
            #         # pred.mean[:, 0],
            #         mean_new + 1.98 * torch.sqrt(var_new),
            #         color="c",
            #         alpha=0.2,
            #         label=r"$\mu_{new}(\cdot) \pm 1.98\sigma_{new}(\cdot)$",
            #     )

            #     # mean, var, noise_var = predict(X_test, data_new=data_new)
            #     plt.legend()
            #     plt.savefig("mo_gp" + str(i) + ".pdf", transparent=True)

            # plt.figure()
            # plot(t)

            global_step += 1
            episode_reward += time_step["reward"]
            t += 1

        logger.info("Finished collecting {} time steps".format(t))

        # Log training metrics
        env_step = global_step * cfg.env.action_repeat

        elapsed_time = time.time() - last_time
        total_time = time.time() - start_time
        last_time = time.time()
        # logger.info("reward shape {}".format(episode_reward.shape))
        logger.info("reward type {}".format(type(episode_reward)))
        train_metrics = {
            "episode": episode_idx,
            "step": global_step,
            "env_step": env_step,
            "episode_time": elapsed_time,
            "total_time": total_time,
            "episode_return": episode_reward,
            # "episode_reward": np.mean(episode_reward),
        }
        logger.info(
            "TRAINING | Episode: {} | Reward: {}".format(episode_idx, episode_reward)
        )
        if cfg.wandb.use_wandb:
            wandb.log({"train/": train_metrics})
            # wandb.log({"train/": train_metrics}, step=env_step)

        # Train agent
        # for _ in range(cfg.episode_length // cfg.update_every_steps):
        if episode_idx >= cfg.init_random_episodes:
            logger.info("Training agent")
            agent.train(replay_buffer)

            # Log rewards/videos in eval env
            if episode_idx % cfg.eval_episode_freq == 0:
                # print("Evaluating {}".format(episode_idx))
                logger.info("Starting eval episodes")
                G_no_online_updates = src.rl.utils.evaluate(
                    eval_env,
                    agent,
                    episode_idx=episode_idx,
                    num_episodes=1,
                    online_updates=False,
                    online_update_freq=cfg.online_update_freq,
                    video=video_recorder,
                    device=cfg.device,
                )

                # Gs = utils.evaluate(
                #     eval_env,
                #     agent,
                #     episode_idx=episode_idx,
                #     # num_episode=cfg.eval_episode_freq,
                #     num_episodes=1,
                #     # num_episodes=10,
                #     # video=video_recorder,
                # )
                # print("DONE EVALUATING")
                # eval_episode_reward = np.mean(Gs)
                env_step = global_step * cfg.env.action_repeat
                eval_metrics = {
                    "episode": episode_idx,
                    "step": global_step,
                    "env_step": env_step,
                    "episode_time": elapsed_time,
                    "total_time": total_time,
                    # "episode_reward": eval_episode_reward,
                    "episode_return/no_online_updates": G_no_online_updates,
                }
                logger.info(
                    "EVAL (no updates) | Episode: {} | Retrun: {}".format(
                        episode_idx, G_no_online_updates
                    )
                )

                if cfg.online_updates:
                    G_online_updates = src.rl.utils.evaluate(
                        eval_env,
                        agent,
                        episode_idx=episode_idx,
                        num_episodes=1,
                        online_updates=cfg.online_updates,
                        online_update_freq=cfg.online_update_freq,
                        video=video_recorder,
                        device=cfg.device,
                    )
                    eval_metrics.update(
                        {"episode_return/online_updates": G_online_updates}
                    )
                    logger.info(
                        "EVAL (with updates) | Episode: {} | Return: {}".format(
                            episode_idx, G_online_updates
                        )
                    )

                if cfg.wandb.use_wandb:
                    wandb.log({"eval/": eval_metrics})

            dataset = replay_buffer.sample(batch_size=len(replay_buffer))
            state_action_inputs = torch.concat(
                [dataset["state"], dataset["action"]], -1
            ).to(cfg.device)
            state_diff_output = (dataset["next_state"] - dataset["state"]).to(
                cfg.device
            )
            reward_output = dataset["reward"].to(cfg.device)
            # print("state_action_input {}".format(state_action_inputs.shape))
            state_diff_mean, state_diff_var = transition_model.ntksvgp.predict(
                state_action_inputs
            )
            # print("state_diff_mean {}".format(state_diff_mean.shape))
            # print("state_diff_var {}".format(state_diff_var.shape))
            # print("state_diff_pred {}".format(state_diff_pred))
            # print("state_diff_output {}".format(state_diff_output.shape))
            mse_transition_model = torch.mean(
                (state_diff_mean - state_diff_output) ** 2
            )
            nlpd_transition_model = -torch.sum(
                torch.distributions.Normal(
                    state_diff_mean, torch.sqrt(state_diff_var)
                ).log_prob(state_diff_output)
            )

            reward_mean, reward_var = reward_model.ntksvgp.predict(
                state_action_inputs
                # state=state[None, ...], action=action_input[None, ...]
            )
            mse_reward_model = torch.mean((reward_mean - reward_output) ** 2)
            # print("mse_trans {}".format(mse_transition_model))
            # print("mse_reward {}".format(mse_reward_model))
            wandb.log({"mse_transition_model": mse_transition_model})
            wandb.log({"mse_reward_model": mse_reward_model})

            nlpd_reward_model = -torch.sum(
                torch.distributions.Normal(
                    reward_mean, torch.sqrt(reward_var)
                ).log_prob(reward_output)
            )
            # print("nlpd_transition_model {}".format(nlpd_transition_model))
            # print("nlpd_reward_model {}".format(nlpd_reward_model))
            wandb.log({"nlpd_transition_model": torch.prod(nlpd_transition_model)})
            wandb.log({"nlpd_reward_model": torch.prod(nlpd_reward_model)})


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
