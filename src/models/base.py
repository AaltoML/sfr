#!/usr/bin/env python3
from typing import Any, Callable, NamedTuple, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
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
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec, ReplayBuffer, UnboundedContinuousTensorSpec
from torchrl.envs import ModelBasedEnvBase
from torchrl.modules import SafeModule, WorldModelWrapper
from torchtyping import TensorType


class TransitionModel:
    def __call__(
        # self, state: State, action: Action
        self,
        state_mean: StateMean,
        state_var: StateVar,
        noise_var: NoiseVar,
        action: Action,
    ) -> Tuple[StateMean, StateVar, NoiseVar]:
        """Returns next state mean/var and noise var"""
        print("inside transition model call")
        print("state_mean: {}".format(state_mean.shape))
        print("state_var: {}".format(state_var))
        print("noise_var: {}".format(noise_var))
        print("action : {}".format(action.shape))
        assert state_mean.shape[0] == action.shape[0]
        # assert state_var.shape[0] == action.shape[0]
        assert state_mean.ndim == action.ndim
        if state_var is None:
            print("state_var is none")
            state = state_mean
        else:
            state_dist = td.Normal(loc=state_mean, scale=torch.sqrt(state_var))
            state = state_dist.sample([1])[0, ...]  # TODO use more than one sample?
            print("state sample you: {}".format(state.shape))

        # assert state_mean.shape[0] == state_var.shape[0]
        state_action_input = torch.concat([state, action], -1)
        print("state_action_input {}".format(state_action_input.shape))
        delta_state_mean, delta_state_var, noise_var = self.forward(state_action_input)
        print("delta_state_mean {}".format(delta_state_mean.shape))
        print("delta_state_var {}".format(delta_state_var.shape))
        print("noise_var {}".format(noise_var.shape))
        next_state_mean = state + delta_state_mean
        print("next_state_mean {}".format(next_state_mean.shape))
        next_state_var = delta_state_var
        print("next_state_var {}".format(next_state_var.shape))
        return next_state_mean, next_state_var, noise_var

    def forward(
        self, x, data_new: Optional = None
    ) -> Tuple[DeltaStateMean, DeltaStateVar, NoiseVar]:
        """Returns change in state"""
        raise NotImplementedError

    def train(self, replay_buffer: ReplayBuffer):
        raise NotImplementedError


class RewardModel:
    def __call__(
        self,
        state_mean: StateMean,
        state_var: StateVar,
        noise_var: NoiseVar,
        # num_samples: int = 5,
    ) -> TensorType["one"]:
        """Returns expected reward"""
        num_samples = 5  # TODO move this somewehre esle
        # TODO use noise_var
        state_dist = td.Normal(loc=state_mean, scale=torch.sqrt(state_var))
        print("state_dist {}".format(state_dist))
        state_samples = state_dist.sample(num_samples)
        print("state_samples {}".format(state_samples))
        reward_samples = self.model(state_samples)
        print("reward_samples {}".format(reward_samples))
        expected_reward = torch.mean(reward_samples, -1)
        return expected_reward

    def forward(self, state: State):
        """Returns reward"""
        raise NotImplementedError

    def train(self, replay_buffer: ReplayBuffer):
        raise NotImplementedError


class GaussianModelBaseEnv(ModelBasedEnvBase):
    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        state_size: int,
        action_size: int,
        device: str = "cpu",
        dtype=None,
        batch_size: int = None,
    ):
        world_model = WorldModelWrapper(
            transition_model=TensorDictModule(
                transition_model,
                in_keys=["state_vector", "state_vector_var", "noise_var", "action"],
                out_keys=["state_vector", "state_vector_var", "noise_var"],
            ),
            reward_model=TensorDictModule(
                reward_model,
                in_keys=["state_vector", "state_vector_var", "noise_var"],
                # in_keys=["state_vector"],
                # in_keys=["state_vector"],
                out_keys=["reward"],
                # out_keys=["expected_reward"],
            ),
        )
        super(GaussianModelBaseEnv, self).__init__(
            world_model=world_model, device=device, dtype=dtype, batch_size=batch_size
        )
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.observation_spec = CompositeSpec(
            state_vector=UnboundedContinuousTensorSpec((state_size)),
            state_vector_var=UnboundedContinuousTensorSpec((state_size)),
            noise_var=UnboundedContinuousTensorSpec((state_size)),
        )
        self.input_spec = CompositeSpec(
            state_vector=UnboundedContinuousTensorSpec((state_size)),
            state_vector_var=UnboundedContinuousTensorSpec((state_size)),
            noise_var=UnboundedContinuousTensorSpec((state_size)),
            action=UnboundedContinuousTensorSpec((action_size)),
        )
        self.reward_spec = UnboundedContinuousTensorSpec((1,))

    def _reset(self, tensordict: TensorDict) -> TensorDict:
        print("inside reset yo")
        print(tensordict)
        tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)
        tensordict = tensordict.update(self.input_spec.rand())
        tensordict = tensordict.update(self.observation_spec.rand())
        tensordict = tensordict.update({"state_vector_mean": torch.zeros(1)})
        print(tensordict)
        return tensordict


# class transitionmodel(safemodule):
#     def __init__(self, model):
#         self.model = model

#     def __call__(self, td: tensordict) -> tensordict:
#         state = td["state_vector"]
#         action = td["action"]
#         state_action_input = torch.concat([state, action], -1)
#         f_mean, f_var, noise_var = self.model.forward(state_action_input)
#         # next_state = self.predict(state=state, action=action)
#         td = tensordict({"f_mean": f_mean, "f_var": f_var, "noise_var": noise_var})
#         return td

#     # def predict(
#     #     # self, stateervation: state, action: action, data_new: optional = none
#     #     self,
#     #     stateervation: state,
#     #     action: action,
#     # ):
#     #     x = torch.concat([state, action], -1)
#     #     return self.forward(x)

#     def forward(self, x, data_new: optional = none) -> prediction:
#         raise notimplementederror

#     def train(self, replay_buffer: ReplayBuffer, delta_state: bool = True):
#         raise NotImplementedError


def expected_reward(
    reward_model: Callable[[State], float],
    state_mean,
    state_stddev,
    num_samples: int = 5,
):
    state = td.Normal(loc=state_mean, scale=state_stddev)
    state_samples = state.sample(num_samples)
    print("state_samples {}".format(state_samples))
    reward_samples = reward_model(state_samples)
    print("reward_samples {}".format(reward_samples))
    expected_reward = torch.mean(reward_samples, -1)
    return expected_reward


def expected_reward_model(reward_model, td: TensorDict) -> TensorDict:
    """Expectation over state dist (transition noise of MDP)"""
    state_dist = td.Normal(
        loc=td["observation_vector_mean"], scale=td["observation_vector_stddev"]
    )
    state_samples = state_dist.sample(num_samples)
    print("state_samples {}".format(state_samples))
    reward_samples = reward_model(state_samples)
    print("reward_samples {}".format(reward_samples))
    expected_reward = torch.mean(reward_samples, -1)
    td = TensorDict({"expected_reward": expected_reward})
    return td


def transition_model(td: TensorDict) -> TensorDict:
    """Expectation over state dist (transition noise of MDP)"""
    state_dist = td.Normal(
        loc=td["observation_vector_mean"], scale=td["observation_vector_stddev"]
    )
    state_samples = state_dist.sample(num_samples)
    print("state_samples {}".format(state_samples.shape))
    action_broadcast = torch.broadcast_to(
        torch.unsqueeze(td["action"], dim=0), state_samples.shape
    )
    print("action_broadcast {}".format(action_broadcast.shape))
    next_state = transition_model(state_samples, action_broadcast)
    td = TensorDict({"next_observation_vector": next_state})
    return td


# class HUCRLModelEnv(ModelBasedEnv):
#     def __init__(
#         self,
#         state_size: int,
#         action_size: int,
#         transition_model: TransitionModel,
#         reward_model: RewardModel,
#         num_samples: int = 5,
#         # world_model: WorldModelWrapper,
#         device: str = "cpu",
#         dtype=None,
#         batch_size: int = None,
#     ):
#         world_model = WorldModelWrapper(
#             transition_model=TensorDictModule(
#                 transition_model,
#                 in_keys=[
#                     "observation_vector_mean",
#                     "observation_vector_stddev",
#                     "action",
#                 ],
#                 out_keys=["observation_vector_mean", "observation_vector_stddev"],
#             ),
#             reward_model=TensorDictModule(
#                 expected_reward_model,
#                 in_keys=[
#                     "observation_vector_mean",
#                     "observation_vector_stddev",
#                     "action",
#                 ],
#                 out_keys=["expected_reward"],
#             ),
#         )
#         super(HUCRLModelBaseEnv, self).__init__(
#             world_model=world_model, device=device, dtype=dtype, batch_size=batch_size
#         )
#         self.observation_spec = CompositeSpec(
#             observation_vector_mean=UnboundedContinuousTensorSpec((state_size)),
#             observation_vector_std=UnboundedContinuousTensorSpec((state_size)),
#         )
#         self.input_spec = CompositeSpec(
#             observation_mean=UnboundedContinuousTensorSpec((state_size)),
#             observation_std=UnboundedContinuousTensorSpec((state_size)),
#             action=UnboundedContinuousTensorSpec((action_size)),
#         )
#         self.reward_spec = CompositeSpec(
#             observation_vector_mean=UnboundedContinuousTensorSpec((state_size)),
#             observation_vector_std=UnboundedContinuousTensorSpec((state_size)),
#         )

#     def predict_hucrl(
#         self,
#         state: State,
#         action: Action,
#         beta: float = 0.01,
#         data_new: Optional = None,
#     ) -> Prediction:
#         real_action = action[:, 0:-2]
#         print("action: {}".format(action.shape))
#         hallucinated_action = action[:, -1]
#         print("hallucinated_action: {}".format(hallucinated_action.shape))
#         x = torch.concat([state, real_action], -1)
#         prediction = transition_model.forward(x)
#         latent = prediction.latent_dist
#         mu = latent.mean
#         sigma = latent.stddev
#         next_state_mean = mu + sigma * beta
#         next_state_dist = td.Normal(
#             loc=next_state_mean, scale=torch.sqrt(prediction.noise_var)
#         )
#         return next_state_dist

#     def hucrl_reward(
#         self,
#         state: State,
#         action: Action,
#         beta: float = 0.01,
#         data_new: Optional = None,
#     ) -> Prediction:
#         real_action = action[:, 0:-2]
#         print("action: {}".format(action.shape))
#         hallucinated_action = action[:, -1]
#         print("hallucinated_action: {}".format(hallucinated_action.shape))
#         x = torch.concat([state, real_action], -1)
#         prediction = transition_model.forward(x)
#         latent = prediction.latent_dist
#         mu = latent.mean
#         sigma = latent.stddev
#         next_state_mean = mu + sigma * beta
#         next_state_dist = td.Normal(
#             loc=next_state_mean, scale=torch.sqrt(prediction.noise_var)
#         )


# class TransitionsModel(pl.LightningModule):
#     def __init__(
#         self,
#         model,
#         trainer,
#         batch_size: int = 64,
#         num_workers: int = 1,
#         shuffle: bool = True,
#     ):
#         self.model = model
#         self.trainer = trainer
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.shuffle = shuffle

#     def predict(self, state: State, action: Action) -> Prediction:
#         x = torch.concat([state, action], -1)
#         return self.forward(x)

#     def forward(self) -> Prediction:
#         raise NotImplementedError

#     def train(self, replay_buffer: ReplayBuffer, delta_state: bool = True):
#         train_loader = ReplayBuffer_to_dynamics_DataLoader(
#             replay_buffer=replay_buffer,
#             batch_size=self.batch_size,
#             shuffle=self.shuffle,
#             delta_state=delta_state,
#             num_workers=self.num_workers,
#         )
#         self.trainer.fit(self.model, train_loader)

#     def training_step(self, batch, batch_idx):
#         raise NotImplementedError

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer


# class HUCRLMBEnv(ModelBasedEnvBase):
#     def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
#         super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
#         self.observation_spec = CompositeSpec(
#             hidden_observation=NdUnboundedContinuousTensorSpec((4,))
#         )
#         self.input_spec = CompositeSpec(
#             hidden_observation=NdUnboundedContinuousTensorSpec((4,)),
#             action=NdUnboundedContinuousTensorSpec((1,)),
#         )
#         self.reward_spec = NdUnboundedContinuousTensorSpec((1,))

#     def _reset(self, tensordict: TensorDict) -> TensorDict:
#         tensordict = TensorDict(
#             {},
#             batch_size=self.batch_size,
#             device=self.device,
#         )
#         tensordict = tensordict.update(self.input_spec.rand())
#         tensordict = tensordict.update(self.observation_spec.rand())
#         return tensordict


# from torchrl.modules import MLP, WorldModelWrapper
# import torch.nn as nn

# world_model = WorldModelWrapper(
#     TensorDictModule(
#         MLP(
#             out_features=4, activation_class=nn.ReLU, activate_last_layer=True, depth=0
#         ),
#         in_keys=["hidden_observation", "action"],
#         out_keys=["hidden_observation"],
#     ),
#     TensorDictModule(
#         nn.Linear(4, 1),
#         in_keys=["hidden_observation"],
#         out_keys=["reward"],
#     ),
# )
