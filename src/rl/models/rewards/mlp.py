#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import wandb
from src.custom_types import Action, RewardPrediction, State
from src.utils import EarlyStopper
from torchrl.data import ReplayBuffer

from .base import RewardModel


def init(
    network: torch.nn.Module,
    learning_rate: float = 1e-2,
    num_iterations: int = 1000,
    batch_size: int = 64,
    # num_workers: int = 1,
    wandb_loss_name: str = "Reward model loss",
    early_stopper: EarlyStopper = None,
    device: str = "cuda",
) -> RewardModel:
    loss_fn = torch.nn.MSELoss()
    print("trans device {}".format(device))
    print("after svgp cuda")
    if "cuda" in device:
        network.cuda()

    def predict_fn(state: State, action: Action) -> RewardPrediction:
        state_action_input = torch.concat([state, action], -1)
        reward = network.forward(state_action_input)[:, 0]
        # delta_state_mean, delta_state_var, noise_var = svgp_predict_fn(
        return RewardPrediction(reward_mean=reward, reward_var=0, noise_var=0)

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

            pred = network(state_action_inputs)
            loss = loss_fn(pred, samples["reward"])

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

    return RewardModel(predict=predict_fn, train=train_fn)
