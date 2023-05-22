#!/usr/bin/env python3
import logging
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import wandb
from src.rl.custom_types import Action, InputData, OutputData, RewardPrediction, State
from src.rl.models.util import weights_init_normal
from src.rl.utils import EarlyStopper

# from torchrl.data import ReplayBuffer

from src.rl.utils.buffer import ReplayBuffer
from .base import RewardModel


class NTKSVGPRewardModel(RewardModel):
    def __init__(
        self,
        network: torch.nn.Module,
        learning_rate: float = 1e-2,
        num_iterations: int = 1000,
        batch_size: int = 64,
        num_inducing: int = 100,
        dual_batch_size: Optional[int] = None,
        delta: float = 0.0001,  # weight decay
        sigma_noise: float = 1.0,
        jitter: float = 1e-4,
        wandb_loss_name: str = "Reward model loss",
        early_stopper: EarlyStopper = None,
        device: str = "cuda",
        prediction_type: str = "SVGPMeanOnly",  # "SVGPMeanOnly" or "SVGP" or "NN"
        logging_freq: int = 500,
    ):
        if "cuda" in device:
            network.cuda()
        self.network = network
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.delta = delta
        self.sigma_noise = sigma_noise
        self.wandb_loss_name = wandb_loss_name
        self.early_stopper = early_stopper
        self.device = device
        self.prediction_type = prediction_type
        self.logging_freq = logging_freq

        likelihood = src.nn2svgp.likelihoods.Gaussian(sigma_noise=sigma_noise)
        prior = src.nn2svgp.priors.Gaussian(params=network.parameters, delta=delta)
        self.ntksvgp = src.nn2svgp.NTKSVGP(
            network=network,
            prior=prior,
            likelihood=likelihood,
            output_dim=1,
            num_inducing=num_inducing,
            dual_batch_size=dual_batch_size,
            jitter=jitter,
            device=device,
        )

        # loss_fn = torch.nn.MSELoss()
        # print("trans device {}".format(device))
        if "cuda" in device:
            self.ntksvgp.cuda()
            print("put reward ntksvgp on cuda")

    @torch.no_grad()
    def predict(self, state: State, action: Action) -> RewardPrediction:
        state_action_input = torch.concat([state, action], -1)
        if "NN" in self.prediction_type:
            reward_mean = self.network.forward(state_action_input)
            reward_var = torch.zeros_like(reward_mean)
        elif "SVGP" in self.prediction_type:
            reward_mean, reward_var = self.ntksvgp.predict_f(state_action_input)
        elif "SVGPMeanOnly" in self.prediction_type:
            reward_mean, reward_var = self.ntksvgp.predict_mean(state_action_input)
            reward_var = torch.zeros_like(reward_mean)
        else:
            raise NotImplementedError(
                "prediction_type should be one of SVGP, SVGPMeanOnly or NN"
            )
        return RewardPrediction(
            reward_mean=reward_mean[:, 0],
            reward_var=reward_var[:, 0],
            noise_var=self.sigma_noise**2,
        )

    def train(self, replay_buffer: ReplayBuffer):
        if self.early_stopper is not None:
            self.early_stopper.reset()

        # self.network.apply(weights_init_normal)
        self.network.train()
        optimizer = torch.optim.Adam(
            [{"params": self.network.parameters()}], lr=self.learning_rate
        )
        for i in range(self.num_iterations):
            samples = replay_buffer.sample(batch_size=self.batch_size)
            state_action_inputs = torch.concat(
                [samples["state"], samples["action"]], -1
            )

            loss = self.ntksvgp.loss(x=state_action_inputs, y=samples["reward"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.wandb_loss_name is not None:
                wandb.log({self.wandb_loss_name: loss})

            if i % self.logging_freq == 0:
                logger.info("Iteration : {} | Loss: {}".format(i, loss))
            if self.early_stopper is not None:
                stop_flag = self.early_stopper(loss)
                if stop_flag:
                    logger.info("Early stopping criteria met, stopping training")
                    logger.info("Breaking out loop")
                    break

        data = replay_buffer.sample(batch_size=len(replay_buffer))
        state_action_inputs = torch.concat([data["state"], data["action"]], -1)
        reward = data["reward"]
        # print("reward {}".format(reward.shape))
        self.ntksvgp.set_data((state_action_inputs, reward))

    @torch.no_grad()
    def update(self, data_new):
        # ntksvgp.set_data((x, y))
        return self.ntksvgp.update(x=data_new[0], y=data_new[1])
