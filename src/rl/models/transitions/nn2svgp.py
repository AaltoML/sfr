#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src
import torch
import wandb
from src.custom_types import Action, State, StatePrediction, InputData, OutputData
from src.utils import EarlyStopper
from torchrl.data import ReplayBuffer

from .base import TransitionModel


class NTKSVGPTransitionModel:
    def __init__(
        self,
        network: torch.nn.Module,
        learning_rate: float = 1e-2,
        num_iterations: int = 1000,
        batch_size: int = 64,
        # num_workers: int = 1,
        num_inducing: int = 30,
        jitter: float = 1e-6,
        delta: float = 0.001,
        nll=src.models.nn2svgp.nll,
        wandb_loss_name: str = "Transition model loss",
        early_stopper: EarlyStopper = None,
        device: str = "cuda",
    ):
        self.network = network
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.num_inducing = num_inducing
        self.jitter = jitter
        self.delta = delta
        self.nll = nll
        self.wandb_loss_name = wandb_loss_name
        self.early_stopper = early_stopper
        self.device = device

        loss_fn = torch.nn.MSELoss()

        def regularised_loss_fn(x, y):
            squared_params = torch.cat(
                [torch.square(param.view(-1)) for param in network.parameters()]
            )
            l2r = 0.5 * torch.sum(squared_params)
            f_pred = network(x)
            return 0.5 * loss_fn(f_pred, y) + self.delta * l2r

        self.loss_fn = regularised_loss_fn

    def train(self, replay_buffer: ReplayBuffer):
        # MLP training
        self.early_stopper.reset()
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

            loss = self.loss_fn(x=state_action_inputs, y=state_diff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.wandb_loss_name is not None:
                wandb.log({self.wandb_loss_name: loss})

            logger.info("Iteration : {} | Loss: {}".format(i, loss))
            stop_flag = self.early_stopper(loss)
            if stop_flag:
                logger.info("Early stopping criteria met, stopping training")
                logger.info("Breaking out loop")
                break

        # Build SVGP
        samples = replay_buffer.sample(batch_size=len(replay_buffer))
        print("num_data replay {}".format(samples))
        state_action_inputs = torch.concat([samples["state"], samples["action"]], -1)
        state_diff = samples["next_state"] - samples["state"]
        self.build_svgp(state_action_inputs, state_diff)

    def build_svgp(self, X, Y):
        logger.info("Building SVGP...")
        self.svgp = src.models.nn2svgp.NTKSVGP(
            network=self.network,
            train_data=(X, Y),
            num_inducing=self.num_inducing,
            jitter=self.jitter,
            delta=self.delta,
            nll=self.nll,
        )
        logger.info("Finished building SVGP")

    def predict(self, state: State, action: Action) -> StatePrediction:
        state_action_input = torch.concat([state, action], -1)
        prediction = self.svgp.predict(state_action_input)
        return StatePrediction(
            state_mean=state + prediction.mean,
            state_var=prediction.var,
            noise_var=prediction.noise_var,
        )
        # TODO how to handle when svgp not build?
        try:
            prediction = self.svgp.predict(state_action_input)
            return StatePrediction(
                state_mean=state + prediction.mean,
                state_var=prediction.var,
                noise_var=prediction.noise_var,
            )
        except:
            logger.info("SVGP prediction failed so using neural network")
            delta_state = self.network.forward(state_action_input)
            return StatePrediction(
                state_mean=state + delta_state, state_var=0.0, noise_var=0.0
            )

    def update(self, x: InputData, y: OutputData):
        self.svgp.update(x, y)

    # def nn_predict(self, x):
    #     return self.network(x)

    # def gp_predict(self, x):
    #     return self.network(x)


def init(
    network: torch.nn.Module,
    learning_rate: float = 1e-2,
    # learning_rate: float = 1e-3,
    num_iterations: int = 1000,
    batch_size: int = 64,
    # num_workers: int = 1,
    num_inducing: int = 500,
    jitter: float = 1e-4,
    delta: float = 0.0001,
    wandb_loss_name: str = "Transition model loss",
    early_stopper: EarlyStopper = None,
    device: str = "cuda",
) -> TransitionModel:
    # loss_fn = torch.nn.MSELoss()
    print("trans device {}".format(device))
    print("after svgp cuda")
    if "cuda" in device:
        network.cuda()

    svgp = NTKSVGPTransitionModel(
        network=network,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        batch_size=batch_size,
        num_inducing=num_inducing,
        jitter=jitter,
        delta=delta,
        nll=src.models.nn2svgp.nll,
        wandb_loss_name=wandb_loss_name,
        early_stopper=early_stopper,
        device=device,
    )
    # TODO put svgp on cuda
    # if "cuda" in device:
    #     svgp.cuda()

    return TransitionModel(predict=svgp.predict, train=svgp.train, update=svgp.update)
