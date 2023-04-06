#!/usr/bin/env python3
import logging
from copy import deepcopy
from typing import Any, Callable, NamedTuple, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import EarlyStopper
import gpytorch
import torch
import torch.distributions as td
import torch.optim as optim
from custom_types import (
    Action,
    DeltaStateMean,
    DeltaStateVar,
    NoiseVar,
    Prediction,
    State,
    StateMean,
    StateVar,
)
from models.svgp import SVGP
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.utils.data import DataLoader, TensorDataset
from torchrl.data import CompositeSpec, ReplayBuffer, UnboundedContinuousTensorSpec
from torchrl.envs import ModelBasedEnvBase
from torchrl.modules import SafeModule, WorldModelWrapper
from torchtyping import TensorType


class TransitionModel(NamedTuple):
    # [InputData, Optional[Data]], Tuple[DeltaStateMean, DeltaStateVar, NoiseVar]
    predict: Callable[[TensorDict], TensorDict]
    train: Callable[[ReplayBuffer], TensorType[float, ""]]


def build_SVGPTransitionModel(
    svgp: SVGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    learning_rate: float = 1e-2,
    batch_size: int = 64,
    num_epochs: int = 1000,
    num_workers: int = 1,
    wandb_loss_name: str = "Transition model loss",
    num_samples: int = 10,
    early_stopper: EarlyStopper = None,
):
    from models.svgp import predict, train

    # output_dim = 5
    # num_inducing = 500
    # Z = torch.rand(output_dim, num_inducing, 6)
    # svgp = SVGP(
    #     inducing_points=Z,
    #     mean_module=deepcopy(mean_module),
    #     covar_module=deepcopy(covar_module),
    #     learn_inducing_locations=learn_inducing_locations,
    # )

    assert (
        len(svgp.variational_strategy.base_variational_strategy.inducing_points.shape)
        == 3
    )
    (
        output_dim,
        num_inducing,
        input_dim,
    ) = svgp.variational_strategy.base_variational_strategy.inducing_points.shape
    predict_fn = predict(svgp=svgp, likelihood=likelihood)

    def expected_transition(state, state_var, noise_var, action):
        state_action_inputs = torch.concat([state, action], -1)
        state_diff_mean, state_diff_var, next_noise_var = predict_fn(
            state_action_inputs
        )
        next_state_mean = state + state_diff_mean
        return next_state_mean, state_diff_var, next_noise_var
        # print("inside expected transition")
        if state_var is None:
            # print("state_var is none: {}".format(state_var))
            state_action_inputs = torch.concat([state, action], -1)
            state_diff_mean, state_diff_var, next_noise_var = predict_fn(
                state_action_inputs
            )
            # print("state_diff_mean.shape")
            # print(state_diff_mean.shape)
            # print(state_diff_var.shape)
            # print(next_noise_var.shape)
            next_state = state + state_diff_mean
            # print("next_state: {}".format(next_state.shape))
            return next_state, state_diff_var, next_noise_var
        else:
            # print("state_var: {}".format(state_var.shape))
            # print("noise_var: {}".format(noise_var.shape))
            # TODO use noise_var
            state_dist = td.Normal(loc=state, scale=torch.sqrt(state_var + noise_var))
            state_samples = state_dist.sample([num_samples])
            # def expected_transition(td: TensorDict) -> TensorDict:
            # state_dist = td.Normal(
            #     loc=td["state_vector"], scale=torch.sqrt(td["state_vector_var"])
            # )
            # print("state_samples {}".format(state_samples.shape))
            # print("action {}".format(action.shape))
            action_broadcast = action[None, :, :].repeat(num_samples, 1, 1)
            # action_broadcast = torch.broadcast_to(
            #     torch.unsqueeze(action, dim=0), state_samples.shape
            # )
            # print("action_broadcast {}".format(action_broadcast.shape))
            state_action_inputs = torch.concat([state_samples, action_broadcast], -1)
            # print("state_action_inputs {}".format(state_action_inputs.shape))
            dim_1, dim_2, input_dim = state_action_inputs.shape
            state_diff_mean, state_diff_var, next_noise_var = predict_fn(
                state_action_inputs.reshape(-1, input_dim)
            )
            output_dim = state_diff_mean.shape[-1]
            # TODO should we be taking mean here?
            state_diff_mean = torch.mean(
                state_diff_mean.reshape(dim_1, dim_2, output_dim), 0
            )
            state_diff_var = torch.mean(
                state_diff_var.reshape(dim_1, dim_2, output_dim), 0
            )
            next_noise_var = torch.mean(
                next_noise_var.reshape(dim_1, dim_2, output_dim), 0
            )
            # print("state_diff_mean.shape")
            # print(state_diff_mean.shape)
            # print(state_diff_var.shape)
            # print(next_noise_var.shape)
            next_state = state + state_diff_mean
            next_state_var = state_diff_var
            # TODO make sure unc is propagated correctly
            # print("next_state: {}".format(next_state.shape))
            return next_state, next_state_var, next_noise_var

    def train_from_replay_buffer(replay_buffer: ReplayBuffer):
        num_data = len(replay_buffer)
        samples = replay_buffer.sample(num_data)
        state = samples["state_vector"]
        action = samples["action"]
        state_action_input = torch.concat([state, action], -1)
        next_state = samples["next"]["state_vector"]
        state_diff = next_state - state
        train_loader = DataLoader(
            TensorDataset(state_action_input, state_diff),
            batch_size=batch_size,
            # batch_size=num_data,
            shuffle=True,
            # num_workers=num_workers
        )

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

        samples = replay_buffer.sample(batch_size=num_inducing)
        Z = torch.concat([samples["state_vector"], samples["action"]], -1)
        Zs = torch.stack([torch.clone(Z) for _ in range(output_dim)], 0)
        Z = torch.nn.parameter.Parameter(Zs, requires_grad=True)
        print("Z.shape {}".format(Z.shape))
        print(
            "Zold: {}".format(
                svgp.variational_strategy.base_variational_strategy.inducing_points.shape
            )
        )
        svgp.variational_strategy.base_variational_strategy.inducing_points = Z

        # # mean_module = svgp.mean_module
        # # covar_module = svgp.covar_module
        # svgp = SVGP(
        #     inducing_points=Z,
        #     mean_module=mean_module,
        #     covar_module=covar_module,
        #     learn_inducing_locations=learn_inducing_locations,
        # )

        return train(
            svgp=svgp,
            likelihood=likelihood,
            learning_rate=learning_rate,
            num_data=num_data,
            wandb_loss_name=wandb_loss_name,
            early_stopper=early_stopper,
        )(data_loader=train_loader, num_epochs=num_epochs)

    return TransitionModel(predict=expected_transition, train=train_from_replay_buffer)
