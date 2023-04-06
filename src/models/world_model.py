#!/usr/bin/env python3
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import ModelBasedEnvBase
from torchrl.modules import (
    CEMPlanner,
    MLP,
    MPPIPlanner,
    ValueOperator,
    WorldModelWrapper,
)
from torchrl.objectives.value import TDLambdaEstimate
from models import TransitionModel, RewardModel


# def cartpole_pets(state, state_var, noise_var, action) -> TensorDict:
#     assert len(next_obs.shape) == len(act.shape) == 2
#     goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
#     x0 = next_obs[:, :1]
#     theta = next_obs[:, 1:2]
#     ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
#     obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6**2))
#     act_cost = -0.01 * torch.sum(act**2, dim=1)
#     return (obs_cost + act_cost).view(-1, 1)


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
                transition_model.predict,
                in_keys=["state_vector", "state_vector_var", "noise_var", "action"],
                out_keys=["state_vector", "state_vector_var", "noise_var"],
            ),
            reward_model=TensorDictModule(
                reward_model.predict,
                in_keys=["state_vector", "state_vector_var", "noise_var", "action"],
                out_keys=["reward"],
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


# class SVGPModelBaseEnv(ModelBasedEnvBase):
#     def __init__(
#         self,
#         transition_model: TransitionModel,
#         reward_model: RewardModel,
#         state_size: int,
#         action_size: int,
#         learning_rate: float = 0.01,
#         num_iterations: int = 5000,
#         device: str = "cpu",
#         dtype=None,
#         batch_size: int = None,
#     ):
#         world_model = WorldModelWrapper(
#             transition_model=TensorDictModule(
#                 transition_model,
#                 in_keys=["state_vector", "state_vector_var", "noise_var", "action"],
#                 out_keys=["state_vector", "state_vector_var", "noise_var"],
#             ),
#             reward_model=TensorDictModule(
#                 reward_model,
#                 in_keys=["state_vector", "state_vector_var", "noise_var"],
#                 # in_keys=["state_vector"],
#                 # in_keys=["state_vector"],
#                 out_keys=["reward"],
#                 # out_keys=["expected_reward"],
#             ),
#         )
#         super(GaussianModelBaseEnv, self).__init__(
#             world_model=world_model, device=device, dtype=dtype, batch_size=batch_size
#         )
#         self.transition_model = transition_model
#         self.reward_model = reward_model
#         self.observation_spec = CompositeSpec(
#             state_vector=UnboundedContinuousTensorSpec((state_size)),
#             state_vector_var=UnboundedContinuousTensorSpec((state_size)),
#             noise_var=UnboundedContinuousTensorSpec((state_size)),
#         )
#         self.input_spec = CompositeSpec(
#             state_vector=UnboundedContinuousTensorSpec((state_size)),
#             state_vector_var=UnboundedContinuousTensorSpec((state_size)),
#             noise_var=UnboundedContinuousTensorSpec((state_size)),
#             action=UnboundedContinuousTensorSpec((action_size)),
#         )
#         self.reward_spec = UnboundedContinuousTensorSpec((1,))
#         self.learning_rate = learning_rate
#         self.num_iterations = num_iterations

#     def _reset(self, tensordict: TensorDict) -> TensorDict:
#         print("inside reset yo")
#         print(tensordict)
#         tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)
#         tensordict = tensordict.update(self.input_spec.rand())
#         tensordict = tensordict.update(self.observation_spec.rand())
#         tensordict = tensordict.update({"state_vector_mean": torch.zeros(1)})
#         print(tensordict)
#         return tensordict

#     def train(self, replay_buffer: ReplayBuffer):
#         transition_data = (
#             torch.concat([sample["state_vector"], sample["action"]], -1),
#             sample["next"]["state_vector"] - sample["state_vector"],
#         )

#         optimizer = torch.optim.Adam(
#             [
#                 {"params": self.transition_model.parameters()},
#                 {"params": self.reward_model.parameters()},
#             ],
#             lr=self.learning_rate,
#         )

#         num_data = len(replay_buffer)
#         self.transition_model.build_loss(num_data=num_data)
#         self.reward_model.build_loss(num_data=num_data)
#         logger.info("Data set size: {}".format(num_data))

#         for i in range(self.num_iterations):
#             print("batch_size: {}".format(self.batch_size))
#             sample = replay_buffer.sample(batch_size=[self.batch_size])
#             batch = (
#                 torch.concat([sample["state_vector"], sample["action"]], -1),
#                 sample["next"]["state_vector"] - sample["state_vector"],
#             )
#             model_loss = self.transition_model(data=batch)
#             wandb.log({"Model loss": model_loss})

#             batch = (sample["next"]["state_vector"], sample["reward"][..., 0])
#             reward_loss = self.reward_model(data=batch)
#             wandb.log({"Reward loss": loss})
#             logger.info(
#                 "Model training iteration {} | Transition loss: {} | Reward loss: {}".format(
#                     i, model_loss, reward_loss
#                 )
#             )
