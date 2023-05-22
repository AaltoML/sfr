#!/usr/bin/env python3
import logging
from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import laplace
import src
import torch
import wandb
from src.rl.custom_types import Action, InputData, OutputData, State, StatePrediction
from src.rl.models.util import weights_init_normal
from src.rl.utils import EarlyStopper
from src.rl.utils.buffer import ReplayBuffer
from torch.utils.data import DataLoader, TensorDataset

from .base import TransitionModel


class EnsembleTransitionModel(TransitionModel):
    def __init__(
        self,
        networks: List[torch.nn.Module],
        state_dim: int,
        learning_rate: float = 1e-2,
        num_iterations: int = 1000,
        batch_size: int = 64,
        # num_workers: int = 1,
        delta: float = 0.0001,  # weight decay
        sigma_noise: float = 1.0,
        wandb_loss_name: str = "Transition model loss",
        early_stopper: EarlyStopper = None,
        device: str = "cuda",
        # prediction_type: str = "NN",  # "NN" or TODO
        logging_freq: int = 500,
    ):
        for network in networks:
            network.apply(weights_init_normal)
        if "cuda" in device:
            for network in networks:
                network.cuda()

        self.networks = torch.nn.ModuleList(networks)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        # self.num_inducing = num_inducing
        self.delta = delta
        self.sigma_noise = sigma_noise
        self.wandb_loss_name = wandb_loss_name
        self.early_stopper = early_stopper
        self.device = device
        # self.prediction_type = prediction_type
        self.logging_freq = logging_freq

        class Ensemble(torch.nn.Module):
            def __init__(self, networks: List[torch.nn.Module]):
                super().__init__()
                self.networks = networks

            def predict(self, x: InputData) -> OutputData:
                # TODO use vmap here
                ys = []
                for network in self.networks:
                    y = network.forward(x)
                    ys.append(y)
                ys = torch.stack(ys, 0)
                y_mean = torch.mean(ys, 0)
                y_var = torch.var(ys, 0)
                return y_mean, y_var

        self.ensemble = Ensemble(networks=self.networks)

        # build ntksvgp to get loss fn
        self.ntksvgps = []
        for network in self.networks:
            likelihood = src.nn2svgp.likelihoods.Gaussian(sigma_noise=sigma_noise)
            prior = src.nn2svgp.priors.Gaussian(params=network.parameters, delta=delta)
            ntksvgp = src.nn2svgp.NTKSVGP(
                network=network,
                prior=prior,
                likelihood=likelihood,
                output_dim=state_dim,
                num_inducing=None,
                dual_batch_size=None,
                device=device,
            )

            if "cuda" in device:
                ntksvgp.cuda()
                print("put transition LA on cuda")
            self.ntksvgps.append(ntksvgp)

    @torch.no_grad()
    def predict(self, state: State, action: Action) -> StatePrediction:
        state_action_input = torch.concat([state, action], -1)
        # delta_states = []
        # for network in self.networks:
        #     delta_states = network.forward(state_action_input)
        #     delta_states.append(delta_states)
        # delta_states = torch.stack(delta_states, 0)
        # delta_state_mean = torch.mean(delta_states, 0)
        # delta_state_var = torch.var(delta_states, 0)
        delta_state_mean, delta_state_var = self.ensemble.predict(state_action_input)
        # TODO implement way to return members individual predictions
        return StatePrediction(
            state_mean=state + delta_state_mean,
            state_var=delta_state_var,
            noise_var=self.sigma_noise**2,
        )

    def train(self, replay_buffer: ReplayBuffer):
        if self.early_stopper is not None:
            self.early_stopper.reset()
        # self.network.apply(weights_init_normal)
        # params = []
        for idx, ntksvgp in enumerate(self.ntksvgps):
            ntksvgp.network.train()
            # params = {"{}".format(idx): ntksvgp.parameters()}
            # params.append(ntksvgp.parameters())

        optimizer = torch.optim.Adam(
            # [self.ensemble.parameters()],
            # # [params],
            # lr=self.learning_rate
            [{"params": self.ensemble.parameters()}],
            lr=self.learning_rate,
        )
        for i in range(self.num_iterations):
            samples = replay_buffer.sample(batch_size=self.batch_size)
            state_action_inputs = torch.concat(
                [samples["state"], samples["action"]], -1
            )
            state_diff = samples["next_state"] - samples["state"]

            loss = self._loss_fn(x=state_action_inputs, y=state_diff)

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

    def update(self, data_new):
        pass

    def _loss_fn(self, x, y):
        loss = 0
        for ntksvgp in self.ntksvgps:
            loss += ntksvgp.loss(x=x, y=y)
        return loss / self.ensemble_size

    @property
    def ensemble_size(self):
        return len(self.networks)
