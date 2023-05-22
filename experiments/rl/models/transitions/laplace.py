#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import laplace
import src
import torch
import wandb
from src.rl.custom_types import Action, State, StatePrediction
from src.rl.utils import EarlyStopper
from torch.utils.data import DataLoader, TensorDataset
from src.rl.utils.buffer import ReplayBuffer


from .base import TransitionModel


class LaplaceTransitionModel(TransitionModel):
    def __init__(
        self,
        network: torch.nn.Module,
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
        prediction_type: str = "LA",  # "LA" or "NN" or TODO
        logging_freq: int = 500,
        hessian_structure: str = "full",
        subset_of_weights: str = "all",
        backend=laplace.curvature.BackPackGGN,
    ):
        if "cuda" in device:
            network.cuda()

        self.network = network
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        # self.num_inducing = num_inducing
        self.delta = delta
        self.sigma_noise = sigma_noise
        self.wandb_loss_name = wandb_loss_name
        self.early_stopper = early_stopper
        self.device = device
        self.prediction_type = prediction_type
        self.logging_freq = logging_freq

        self.subset_of_weights = subset_of_weights
        self.hessian_structure = hessian_structure

        # build ntksvgp to get loss fn
        likelihood = src.likelihoods.Gaussian(sigma_noise=sigma_noise)
        prior = src.priors.Gaussian(params=network.parameters, delta=delta)
        self.ntksvgp = src.SFR(
            network=network,
            prior=prior,
            likelihood=likelihood,
            output_dim=state_dim,
            num_inducing=None,
            # dual_batch_size=dual_batch_size,
            # dual_batch_size=None,
            # jitter=jitter,
            device=device,
        )

        self.la = laplace.Laplace(
            self.network,
            "regression",
            sigma_noise=sigma_noise,
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
            prior_precision=delta,
            backend=backend,
        )

        if "cuda" in device:
            self.la.cuda()
            print("put transition LA on cuda")

    @torch.no_grad()
    def predict(self, state: State, action: Action) -> StatePrediction:
        state_action_input = torch.concat([state, action], -1)
        if "NN" in self.prediction_type:
            delta_state_mean = self.network.forward(state_action_input)
            delta_state_var = torch.zeros_like(delta_state_mean)
        elif "LA" in self.prediction_type:
            delta_state_mean, delta_state_var = self.la(state_action_input)
            delta_state_var = torch.diagonal(delta_state_var, dim1=-1, dim2=-2)
        else:
            raise NotImplementedError("prediction_type should be one of LA or NN")
        # delta_state_mean = ntksvgp.predict_mean(state_action_input)
        # delta_state_mean, delta_state_var, noise_var = svgp_predict_fn(
        return StatePrediction(
            state_mean=state + delta_state_mean,
            state_var=delta_state_var,
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
            state_diff = samples["next_state"] - samples["state"]

            # pred = network(state_action_inputs)
            # loss = loss_fn(pred, state_diff)
            loss = self.ntksvgp.loss(x=state_action_inputs, y=state_diff)

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
        state_action_inputs_all = torch.concat([data["state"], data["action"]], -1)
        state_diff_all = data["next_state"] - data["state"]

        train_loader = DataLoader(
            TensorDataset(*(state_action_inputs_all, state_diff_all)),
            batch_size=self.batch_size,
            # shuffle=False,
        )
        print("made train_loader {}".format(train_loader))
        # self.ntksvgp.set_data((state_action_inputs_all, state_diff_all))
        # la = laplace.Laplace(
        #     self.network,
        #     "regression",
        #     subset_of_weights=self.subset_of_weights,
        #     hessian_structure=self.hessian_structure,
        # )
        self.la.fit(train_loader)

    def update(self, data_new):
        pass
        # return self.ntksvgp.update(x=data_new[0], y=data_new[1])
