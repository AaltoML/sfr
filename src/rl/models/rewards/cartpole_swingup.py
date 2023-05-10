#!/usr/bin/env python3
import torch
from dm_control.utils import rewards
from src.rl.custom_types import Action, RewardPrediction, State

from .base import RewardModel

_DEFAULT_VALUE_AT_MARGIN = torch.Tensor([0.1])


class CartpoleRewardModel(RewardModel):
    def __init__(self):
        self._reward_fn = torch.vmap(tensor_reward)
        # self._reward_fn = torch.vmap(cartpole_swingup_reward)

    def predict(self, state: State, action: Action) -> RewardPrediction:
        # print("state {}".format(state.shape))
        # print("aciton {}".format(action.shape))
        reward = self._reward_fn(state, action)
        # print("reward {}".format(reward.shape))
        if reward.ndim == 2:
            reward = reward[:, 0]
        return RewardPrediction(reward_mean=reward, reward_var=None, noise_var=None)

    def update(self, data_new):
        pass

    def train(self, replay_buffer):
        pass


def tensor_reward(state, action):
    """DIFFERENT FROM ORIGINAL GYM"""
    cos_theta = state[1]
    v = state[3]
    # theta = next_state[1]
    arm_length = 0.6
    y = arm_length * cos_theta
    x = arm_length * cos_theta
    dist_penalty = 0.01 * x**2 + (y - 1) ** 2
    vel_penalty = 1e-3 * v**2
    reward = -dist_penalty - vel_penalty
    return reward.view([1])


def cartpole_swingup_reward(single_state, single_action):
    # TODO map over data
    assert single_state.ndim == 1
    assert single_action.ndim == 1
    # return torch.sum(single_state + single_action)
    # list_indices = [[0], [2], [3]]
    # convert list_indices to Tensor
    # indices = torch.tensor(list_indices)
    # get elements from tensor_a using indices.
    # tensor_a=torch.index_select(tensor_a, 0, indices.view(-1))
    # print(tensor_a)

    # cart_position = torch.stack([single_state[0], single_state[1], single_state[2]], 0)
    cart_position = single_state[0]
    # print("cart_position {}".format(cart_position.shape))
    # cart_velocity = single_state[1]
    pole_angle_sine = single_state[1]
    # pole_angle_cosine = single_state[1]
    # pole_angle_sine = single_state[3]
    pole_angle_cosine = single_state[2]
    angular_vel = single_state[3]
    angular_vel = single_state[4]

    control = single_action
    # control = single_state[4]
    # angular_vel = single_state[4]
    # if sparse:
    #   cart_in_bounds = rewards.tolerance(physics.cart_position(),
    #                                      self._CART_RANGE)
    #   angle_in_bounds = rewards.tolerance(physics.pole_angle_cosine(),
    #                                       self._ANGLE_COSINE_RANGE).prod()
    #   return cart_in_bounds * angle_in_bounds
    # else:
    # obs['position'] = physics.bounded_position()
    # obs['velocity'] = physics.velocity()
    upright = (pole_angle_cosine + 1) / 2
    # print("upright {}".format(upright.shape))
    centered = tolerance(cart_position, margin=2)
    # print("centered {}".format(centered.shape))
    centered = (1 + centered) / 2
    # print("centered {}".format(centered.shape))
    small_control = tolerance(
        control,
        margin=1,
        value_at_margin=torch.Tensor([0.0]),
        sigmoid="quadratic",
    )[0]
    # print("smallcontrol {}".format(small_control.shape))
    small_control = (4 + small_control) / 5
    # print("smallcontrol {}".format(small_control.shape))
    small_velocity = tolerance(angular_vel, margin=5).min()
    # print("small_velocity {}".format(small_velocity.shape))
    small_velocity = (1 + small_velocity) / 2
    # print("small_velocity {}".format(small_velocity.shape))
    reward = upright.mean() * small_control * small_velocity * centered
    # print("reward {}".format(reward.shape))
    # return upright.mean() * small_control * small_velocity * centered
    return reward


# def cartpole_swingup_reward(single_state, single_action):
#     # TODO map over data
#     assert single_state.ndim == 1
#     assert single_action.ndim == 1
#     # return torch.sum(single_state + single_action)
#     # list_indices = [[0], [2], [3]]
#     # convert list_indices to Tensor
#     # indices = torch.tensor(list_indices)
#     # get elements from tensor_a using indices.
#     # tensor_a=torch.index_select(tensor_a, 0, indices.view(-1))
#     # print(tensor_a)

#     # cart_position = torch.stack([single_state[0], single_state[1], single_state[2]], 0)
#     cart_position = single_state[0]
#     # print("cart_position {}".format(cart_position.shape))
#     # cart_velocity = single_state[1]
#     pole_angle_sine = single_state[1]
#     pole_angle_cosine = single_state[1]
#     # pole_angle_sine = single_state[3]
#     # pole_angle_cosine = single_state[2]
#     angular_vel = single_state[3]

#     control = single_action
#     # control = single_state[4]
#     # angular_vel = single_state[4]
#     # if sparse:
#     #   cart_in_bounds = rewards.tolerance(physics.cart_position(),
#     #                                      self._CART_RANGE)
#     #   angle_in_bounds = rewards.tolerance(physics.pole_angle_cosine(),
#     #                                       self._ANGLE_COSINE_RANGE).prod()
#     #   return cart_in_bounds * angle_in_bounds
#     # else:
#     # obs['position'] = physics.bounded_position()
#     # obs['velocity'] = physics.velocity()
#     upright = (pole_angle_cosine + 1) / 2
#     # print("upright {}".format(upright.shape))
#     centered = tolerance(cart_position, margin=2)
#     # print("centered {}".format(centered.shape))
#     centered = (1 + centered) / 2
#     # print("centered {}".format(centered.shape))
#     small_control = tolerance(
#         control,
#         margin=1,
#         value_at_margin=torch.Tensor([0.0]),
#         sigmoid="quadratic",
#     )[0]
#     # print("smallcontrol {}".format(small_control.shape))
#     small_control = (4 + small_control) / 5
#     # print("smallcontrol {}".format(small_control.shape))
#     small_velocity = tolerance(angular_vel, margin=5).min()
#     # print("small_velocity {}".format(small_velocity.shape))
#     small_velocity = (1 + small_velocity) / 2
#     # print("small_velocity {}".format(small_velocity.shape))
#     reward = upright.mean() * small_control * small_velocity * centered
#     # print("reward {}".format(reward.shape))
#     # return upright.mean() * small_control * small_velocity * centered
#     return reward


def tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    sigmoid="gaussian",
    value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
      x: A scalar or numpy array.
      bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
      margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
          outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
          increasing distance from the nearest bound.
      sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
         'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
      value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
      A float or numpy array with values between 0.0 and 1.0.

    Raises:
      ValueError: If `bounds[0] > bounds[1]`.
      ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    # return float(value) if torch.isscalar(x) else value
    return value


def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
      x: A scalar or numpy array.
      value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
      sigmoid: String, choice of sigmoid type.

    Returns:
      A numpy array with values between 0.0 and 1.0.

    Raises:
      ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
      ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be nonnegative and smaller than 1, "
                "got {}.".format(value_at_1)
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be strictly between 0 and 1, "
                "got {}.".format(value_at_1)
            )

    if sigmoid == "gaussian":
        scale = torch.sqrt(-2 * torch.log(value_at_1))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = torch.arccosh(1 / value_at_1)
        return 1 / torch.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = torch.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = torch.arccos(2 * value_at_1 - 1) / torch.pi
        scaled_x = x * scale
        # with warnings.catch_warnings():
        #     warnings.filterwarnings(
        #         action="ignore", message="invalid value encountered in cos"
        #     )
        cos_pi_scaled_x = torch.cos(torch.pi * scaled_x)
        return torch.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        # print("value_at_1 {}".format(type(value_at_1)))
        scale = torch.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == "tanh_squared":
        scale = torch.arctanh(torch.sqrt(1 - value_at_1))
        return 1 - torch.tanh(x * scale) ** 2

    else:
        raise ValueError("Unknown sigmoid type {!r}.".format(sigmoid))
