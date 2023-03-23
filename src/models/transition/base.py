#!/usr/bin/env python3
from typing import Callable, NamedTuple, Optional

import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim
from custom_types import Action, Prediction, State
from torchrl.data import ReplayBuffer
from torchrl.modules import WorldModelWrapper


def predict_hucrl(
    transition_model: TransitionModel,
    state: State,
    action: Action,
    beta: float = 0.01,
    data_new: Optional = None,
) -> Prediction:
    real_action = action[:, 0:-2]
    print("action: {}".format(action.shape))
    hallucinated_action = action[:, -1]
    print("hallucinated_action: {}".format(hallucinated_action.shape))
    x = torch.concat([state, real_action], -1)
    prediction = transition_model.forward(x)
    latent = prediction.latent_dist
    mu = latent.mean
    sigma = latent.stddev
    next_state_mean = mu + sigma * beta
    next_state_dist = td.Normal(
        loc=next_state_mean, scale=torch.sqrt(prediction.noise_var)
    )
    return next_state_dist


class TransitionModel:
    def __call__(self, state: State, action: Action) -> Prediction:
        return self.predict(state=state, action=action)

    def predict(
        self, state: State, action: Action, data_new: Optional = None
    ) -> Prediction:
        x = torch.concat([state, action], -1)
        state_diff_pred = self.forward(x)
        return state_diff_pred

    # def predict_posterior_sampling(
    #     self, state: State, action: Action, data_new: Optional = None
    # ) -> Prediction:
    #     x = torch.concat([state, action], -1)
    #     return self.forward(x)
    def predict_hucrl(
        self,
        state: State,
        action: Action,
        beta: float = 0.01,
        data_new: Optional = None,
    ) -> Prediction:
        real_action = action[:, 0:-2]
        print("action: {}".format(action.shape))
        hallucinated_action = action[:, -1]
        print("hallucinated_action: {}".format(hallucinated_action.shape))
        x = torch.concat([state, real_action], -1)
        prediction = self.forward(x)
        latent = prediction.latent_dist
        mu = latent.mean
        sigma = latent.stddev
        next_state_mean = mu + sigma * beta
        next_state_dist = td.Normal(
            loc=next_state_mean, scale=torch.sqrt(prediction.noise_var)
        )
        return next_state_dist
        # prediction

    def forward(self, x, data_new: Optional = None) -> Prediction:
        raise NotImplementedError

    def train(self, replay_buffer: ReplayBuffer, delta_state: bool = True):
        raise NotImplementedError
