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


class MyMBEnv(ModelBasedEnvBase):
    def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
        super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
        self.observation_spec = CompositeSpec(
            observation_vector_mean=UnboundedContinuousTensorSpec((obs_size)),
            observation_vector_var=UnboundedContinuousTensorSpec((obs_size)),
        )
        self.input_spec = CompositeSpec(
            observation_vector_mean=UnboundedContinuousTensorSpec((obs_size)),
            ebservation_vector_var=UnboundedContinuousTensorSpec((obs_size)),
            action=UnboundedContinuousTensorSpec((action_size)),
        )
        self.reward_spec = UnboundedContinuousTensorSpec((1,))

    def _reset(self, tensordict: TensorDict) -> TensorDict:
        tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)
        tensordict = tensordict.update(self.input_spec.rand())
        tensordict = tensordict.update(self.observation_spec.rand())
        return tensordict


model = MLP(
    out_features=obs_size,
    activation_class=nn.ReLU,
    activate_last_layer=True,
    depth=0,
)


def transition_model(td: TensorDict) -> TensorDict:
    """Expectation over state dist (transition noise of MDP)"""
    obs_dist = td.Normal(
        loc=td["observation_vector_mean"], scale=td["observation_vector_stddev"]
    )
    obs_samples = obs_dist.sample(num_samples)
    print("obs_samples {}".format(obs_samples.shape))
    action_broadcast = torch.broadcast_to(
        torch.unsqueeze(td["action"], dim=0), obs_samples.shape
    )
    print("action_broadcast {}".format(action_broadcast.shape))
    next_obs = model(obs_samples, action_broadcast)
    td = TensorDict({"next_observation_vector_mean": next_obs})
    td = TensorDict({"next_observation_vector_var": next_obs})
    return td


obs_size = 5
action_size = 1

world_model = WorldModelWrapper(
    transition_model=TensorDictModule(
        # MLP(
        #     out_features=obs_size * 2,
        #     activation_class=nn.ReLU,
        #     activate_last_layer=True,
        #     depth=0,
        # ),
        transition_model,
        in_keys=["observation_vector_mean", "observation_vector_var", "action"],
        out_keys=["observation_vector_mean", "observation_vector_var"],
    ),
    reward_model=TensorDictModule(
        nn.Linear(obs_size, 1),
        in_keys=["observation_vector_mean", "observation_vector_var"],
        out_keys=["expected_reward"],
    ),
)
print(world_model)

env = MyMBEnv(world_model)
tensordict = env.rollout(max_steps=10)
print(tensordict)

# value_net = nn.Linear(obs_size, 1)
# value_net = ValueOperator(value_net, in_keys=["observation_vector"])
# adv = TDLambdaEstimate(0.99, 0.95, value_net)
# Build a planner and use it as actor
# planner = MPPIPlanner(
#     env,
#     adv,
#     temperature=1.0,
#     planning_horizon=10,
#     optim_steps=11,
#     num_candidates=7,
#     top_k=3,
# )
# print("MPPI")
planner = CEMPlanner(
    env,
    planning_horizon=5,
    optim_steps=11,
    num_candidates=7,
    top_k=3,
    reward_key="expected_reward",
    action_key="action",
)
tensordict = env.rollout(max_steps=500, policy=planner)
print("CEM")
print(tensordict)
