#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gpytorch
import torch
from models.svgp import SVGP
from src.custom_types import Action, State, StatePrediction
from src.utils import EarlyStopper
from torch.utils.data import DataLoader, TensorDataset
from torchrl.data import ReplayBuffer

from .base import TransitionModel


def init(
    svgp: SVGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    learning_rate: float = 1e-2,
    batch_size: int = 64,
    num_epochs: int = 1000,
    # num_workers: int = 1,
    wandb_loss_name: str = "Transition model loss",
    early_stopper: EarlyStopper = None,
    device: str = "cuda",
) -> TransitionModel:
    from models.svgp import predict, train

    assert (
        len(svgp.variational_strategy.base_variational_strategy.inducing_points.shape)
        == 3
    )
    (
        output_dim,
        num_inducing,
        input_dim,
    ) = svgp.variational_strategy.base_variational_strategy.inducing_points.shape
    svgp_predict_fn = predict(svgp=svgp, likelihood=likelihood)

    def predict_fn(state: State, action: Action) -> StatePrediction:
        state_action_input = torch.concat([state, action], -1)
        delta_state_mean, delta_state_var, noise_var = svgp_predict_fn(
            state_action_input
        )
        return StatePrediction(
            state_mean=state + delta_state_mean,
            state_var=delta_state_var,
            noise_var=noise_var,
        )

    def train_fn(replay_buffer: ReplayBuffer) -> dict:
        samples = replay_buffer.sample(batch_size=len(replay_buffer))
        state = samples["state"]
        action = samples["action"]
        next_state = samples["next_state"]
        state_action_inputs = torch.concat([state, action], -1)
        state_diff = next_state - state

        num_data = len(replay_buffer)
        print("num_data: {}".format(num_data))
        train_loader = DataLoader(
            TensorDataset(state_action_inputs, state_diff),
            # batch_size=batch_size,
            batch_size=num_data,
            shuffle=True,
            # num_workers=num_workers
        )

        # TODO increase num_inducing as num_data grows??
        # (
        #     output_dim,
        #     num_inducing,
        #     input_dim,
        # ) = svgp.variational_strategy.base_variational_strategy.inducing_points.shape
        # num_inducing = 500
        # num_inducing = (
        #     cfg.model.transition_model.num_inducing
        #     + i * num_new_inducing_points_per_episode
        # )

        indices = torch.randperm(num_data)[:num_inducing]
        Z = state_action_inputs[indices]
        # Zs = torch.stack([torch.clone(Z) for _ in range(output_dim)], 0)
        # TODO check it's ok to use [1, ...] instead of [5, ...]
        Zs = torch.clone(Z)[None, ...]
        Z = torch.nn.parameter.Parameter(Zs, requires_grad=True)
        print("Z.shape {}".format(Z.shape))
        print(
            "Zold: {}".format(
                svgp.variational_strategy.base_variational_strategy.inducing_points.shape
            )
        )
        svgp.variational_strategy.base_variational_strategy.inducing_points = Z

        # TODO reset m and V

        return train(
            svgp=svgp,
            likelihood=likelihood,
            learning_rate=learning_rate,
            num_data=num_data,
            wandb_loss_name=wandb_loss_name,
            early_stopper=early_stopper,
        )(data_loader=train_loader, num_epochs=num_epochs)

    return TransitionModel(predict=predict_fn, train=train_fn)