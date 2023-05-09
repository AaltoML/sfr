#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import wandb
from src.rl.custom_types import Action, InputData, OutputData, State, StatePrediction
from src.rl.models.util import weights_init_normal
from src.rl.utils import EarlyStopper
from torchrl.data import ReplayBuffer

from .base import TransitionModel


class NTKSVGPTransitionModel(TransitionModel):
    def __init__(
        self,
        network: torch.nn.Module,
        state_dim: int,
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
    ):
        if "cuda" in device:
            network.cuda()

        self.network = network
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.num_inducing = num_inducing
        self.delta = delta
        self.sigma_noise = sigma_noise
        self.wandb_loss_name = wandb_loss_name
        self.early_stopper = early_stopper
        self.device = device

        likelihood = src.nn2svgp.likelihoods.Gaussian(sigma_noise=sigma_noise)
        prior = src.nn2svgp.priors.Gaussian(params=network.parameters, delta=delta)
        self.ntksvgp = src.nn2svgp.NTKSVGP(
            network=network,
            prior=prior,
            likelihood=likelihood,
            output_dim=state_dim,
            num_inducing=num_inducing,
            jitter=jitter,
            device=device,
        )

        # loss_fn = torch.nn.MSELoss()
        # print("trans device {}".format(device))
        # print("after svgp cuda")
        if "cuda" in device:
            self.ntksvgp.cuda()
            print("put transition ntksvgp on cuda")

    def predict(self, state: State, action: Action) -> StatePrediction:
        state_action_input = torch.concat([state, action], -1)
        # delta_state = network.forward(state_action_input)
        delta_state_mean, delta_state_var = self.ntksvgp.predict_f(state_action_input)
        # delta_state_mean = ntksvgp.predict_mean(state_action_input)
        # delta_state_mean, delta_state_var, noise_var = svgp_predict_fn(
        return StatePrediction(
            state_mean=state + delta_state_mean,
            # state_var=0,
            state_var=delta_state_var,
            noise_var=self.sigma_noise**2,
            # state_var=delta_state_var,
            # noise_var=noise_var,
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
            state_diff = samples["next_state"] - samples["state"]

            # pred = network(state_action_inputs)
            # loss = loss_fn(pred, state_diff)
            loss = self.ntksvgp.loss(x=state_action_inputs, y=state_diff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.wandb_loss_name is not None:
                wandb.log({self.wandb_loss_name: loss})

            logger.info("Iteration : {} | Loss: {}".format(i, loss))
            if self.early_stopper is not None:
                stop_flag = self.early_stopper(loss)
                if stop_flag:
                    logger.info("Early stopping criteria met, stopping training")
                    logger.info("Breaking out loop")
                    break

        data = replay_buffer.sample(batch_size=len(replay_buffer))
        state_action_inputs = torch.concat([data["state"], data["action"]], -1)
        state_diff = data["next_state"] - data["state"]
        self.ntksvgp.set_data((state_action_inputs, state_diff))

    def update(self, data_new):
        return self.ntksvgp.update(x=data_new[0], y=data_new[1])