#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import wandb
from src.rl.custom_types import Action, InputData, OutputData, State, RewardPrediction
from src.rl.utils import EarlyStopper
from torchrl.data import ReplayBuffer
import src
from .base import RewardModel


def init(
    network: torch.nn.Module,
    learning_rate: float = 1e-2,
    num_iterations: int = 1000,
    batch_size: int = 64,
    # num_workers: int = 1,
    num_inducing: int = 100,
    delta: float = 0.0001,  # weight decay
    sigma_noise: float = 1.0,
    jitter: float = 1e-4,
    wandb_loss_name: str = "Transition model loss",
    early_stopper: EarlyStopper = None,
    device: str = "cuda",
) -> RewardModel:
    likelihood = src.nn2svgp.likelihoods.Gaussian(sigma_noise=sigma_noise)
    prior = src.nn2svgp.priors.Gaussian(params=network.parameters, delta=delta)
    ntksvgp = src.nn2svgp.NTKSVGP(
        network=network,
        prior=prior,
        likelihood=likelihood,
        output_dim=1,
        num_inducing=num_inducing,
        jitter=jitter,
    )

    # loss_fn = torch.nn.MSELoss()
    print("trans device {}".format(device))
    print("after svgp cuda")
    if "cuda" in device:
        network.cuda()
        ntksvgp.cuda()

    def predict_fn(state: State, action: Action) -> RewardPrediction:
        state_action_input = torch.concat([state, action], -1)
        reward_mean = ntksvgp.predict_mean(state_action_input)
        # print("reward_mean {}".format(reward_mean.shape))
        # TODO use reward_var??
        # delta_state_mean, delta_state_var, noise_var = svgp_predict_fn(
        return RewardPrediction(
            reward_mean=reward_mean[:, 0], reward_var=0, noise_var=0
        )

    def train_fn(replay_buffer: ReplayBuffer):
        early_stopper.reset()
        network.train()
        optimizer = torch.optim.Adam(
            [{"params": network.parameters()}], lr=learning_rate
        )
        for i in range(num_iterations):
            samples = replay_buffer.sample(batch_size=batch_size)
            state_action_inputs = torch.concat(
                [samples["state"], samples["action"]], -1
            )

            loss = ntksvgp.loss(x=state_action_inputs, y=samples["reward"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if wandb_loss_name is not None:
                wandb.log({wandb_loss_name: loss})

            logger.info("Iteration : {} | Loss: {}".format(i, loss))
            stop_flag = early_stopper(loss)
            if stop_flag:
                logger.info("Early stopping criteria met, stopping training")
                logger.info("Breaking out loop")
                break

        data = replay_buffer.sample(batch_size=len(replay_buffer))
        state_action_inputs = torch.concat([data["state"], data["action"]], -1)
        reward = data["reward"]
        # print("reward {}".format(reward.shape))
        ntksvgp.set_data((state_action_inputs, reward))

    def update_fn(data_new):
        # ntksvgp.set_data((x, y))
        return ntksvgp.update(x=data_new[0], y=data_new[1])

    return RewardModel(predict=predict_fn, train=train_fn, update=update_fn)
