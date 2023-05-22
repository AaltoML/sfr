#!/usr/bin/env python3
from typing import NamedTuple, Tuple

from torchtyping import TensorType

InputData = TensorType["num_data", "input_dim"]
OutputData = TensorType["num_data", "output_dim"]
Data = Tuple[InputData, OutputData]

State = TensorType["batch_size", "state_dim"]
Action = TensorType["batch_size", "action_dim"]

ActionTrajectory = TensorType["horizon", "batch_size", "action_dim"]
StateTrajectory = TensorType["horizon", "batch_size", "state_dim"]

EvalMode = bool
T0 = bool

StateMean = TensorType["batch_size", "state_dim"]
StateVar = TensorType["batch_size", "state_dim"]

DeltaStateMean = TensorType["batch_size", "state_dim"]
DeltaStateVar = TensorType["batch_size", "state_dim"]
NoiseVar = TensorType["state_dim"]

RewardMean = TensorType["batch_size"]
RewardVar = TensorType["batch_size"]

Input = TensorType["batch_size, input_dim"]
OutputMean = TensorType["batch_size, output_dim"]
OutputVar = TensorType["batch_size, output_dim"]


class RewardPrediction(NamedTuple):
    reward_mean: RewardMean
    reward_var: RewardVar  # No noise
    noise_var: NoiseVar


class StatePrediction(NamedTuple):
    state_mean: StateMean
    state_var: StateVar  # No noise
    noise_var: NoiseVar


class DeltaStatePrediction(NamedTuple):
    delta_state_mean: DeltaStateMean
    delta_state_var: DeltaStateVar  # No noise
    noise_var: NoiseVar


class Prediction(NamedTuple):
    mean: OutputMean
    var: OutputVar  # No noise
    noise_var: NoiseVar
