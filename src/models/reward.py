#!/usr/bin/env python3
#!/usr/bin/env python3
import logging
from typing import Any, Callable, NamedTuple, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from torchrl.data import ReplayBuffer
from torchrl.envs import ModelBasedEnvBase
from torchrl.modules import SafeModule, WorldModelWrapper
from torchtyping import TensorType
from utils import EarlyStopper


class RewardModel(NamedTuple):
    # predict: Callable[[InputData, Optional[Data]], Tuple[Mean, Var, NoiseVar]]
    predict: Callable[[TensorDict], TensorDict]
    train: Callable[[ReplayBuffer], TensorType[float, ""]]


def build_SVGPRewardModel(
    svgp: SVGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    learning_rate: float = 1e-2,
    batch_size: int = 64,
    num_epochs: int = 1000,
    num_workers: int = 1,
    wandb_loss_name: str = "Reward model loss",
    num_samples: int = 20,
    early_stopper: EarlyStopper = None,
):
    assert len(svgp.variational_strategy.inducing_points.shape) == 2
    num_inducing, input_dim = svgp.variational_strategy.inducing_points.shape
    from models.svgp import predict, train

    predict_fn = predict(svgp=svgp, likelihood=likelihood)

    def expected_reward_fn(state, state_var, noise_var, action):
        # state_action_inputs = torch.concat([state, action], -1)
        # # print("state_action_inputs {}".format(state_action_inputs.shape))
        # reward_mean, reward_var, noise_var = predict_fn(state_action_inputs)
        # return reward_mean

        # print("inside expected reward")
        # # def expected_reward_fn(td: TensorDict) -> TensorDict:
        # state_dist = td.Normal(loc=state, scale=torch.sqrt(state_var + noise_var))
        # print("state_dist: {}".format(state_dist))
        # state_samples = state_dist.sample([num_samples])
        # print("state_samples {}".format(state_samples.shape))
        # reward_samples, reward_var_samples, noise_var_samples = predict_fn(
        #     state_samples
        # )
        # print("reward_samples {}".format(reward_samples.shape))
        # print("reward_var_samples {}".format(reward_var_samples.shape))
        # expected_value = torch.mean(reward_samples, 0)
        # print("expected_value: {}".format(expected_value.shape))
        # return expected_value

        state_dist = td.Normal(loc=state, scale=torch.sqrt(state_var + noise_var))
        state_samples = state_dist.sample([num_samples])
        # print("state_samples {}".format(state_samples.shape))
        # print("action {}".format(action.shape))
        action_broadcast = action[None, :, :].repeat(num_samples, 1, 1)
        # print("action_broadcast {}".format(action_broadcast.shape))
        state_action_inputs = torch.concat([state_samples, action_broadcast], -1)
        # print("state_action_inputs {}".format(state_action_inputs.shape))
        dim_1, dim_2, input_dim = state_action_inputs.shape
        reward_mean, reward_var, noise_var = predict_fn(
            state_action_inputs.reshape(-1, input_dim)
        )
        # print("reward_mean {}".format(reward_mean.shape))
        # TODO should we be taking mean here?
        reward_mean = torch.mean(reward_mean.reshape(dim_1, dim_2), 0)
        # print("reward_mean: {}".format(reward_mean.shape))
        return reward_mean

    def train_from_replay_buffer(replay_buffer: ReplayBuffer):
        num_data = len(replay_buffer)
        samples = replay_buffer.sample(num_data)
        state = samples["state_vector"]
        action = samples["action"]
        state_action_input = torch.concat([state, action], -1)
        reward = samples["next"]["reward"][:, 0]
        # TODO should this predict from next state to reward or state-action to reward
        train_loader = DataLoader(
            TensorDataset(state_action_input, reward),
            # batch_size=num_data,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=num_workers,
        )

        # num_inducing = 500
        # num_inducing = (
        #     cfg.model.transition_model.num_inducing
        #     + i * num_new_inducing_points_per_episode
        # )
        samples = replay_buffer.sample(batch_size=num_inducing)
        Z = torch.concat([samples["state_vector"], samples["action"]], -1)
        Zs = torch.clone(Z)
        Z = torch.nn.parameter.Parameter(Zs, requires_grad=True)
        print("Z.shape {}".format(Z.shape))
        print("Zold: {}".format(svgp.variational_strategy.inducing_points.shape))
        svgp.variational_strategy.inducing_points = Z

        return train(
            svgp=svgp,
            likelihood=likelihood,
            learning_rate=learning_rate,
            num_data=num_data,
            wandb_loss_name=wandb_loss_name,
            early_stopper=early_stopper,
        )(data_loader=train_loader, num_epochs=num_epochs)

    return RewardModel(predict=expected_reward_fn, train=train_from_replay_buffer)
