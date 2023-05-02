#!/usr/bin/env python3
from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from ..models.ensemble import Ensemble, ProbabilisticEnsemble
from ..models.networks import MLP, GaussianMLP


def build_ensemble_of_mlps(
    in_size: int,
    out_size: int,
    ensemble_size: int,
    features: List[int],
    probabilistic: bool = True,  # MLP with Gaussian output or deterministic MLP
    trainers: List[Trainer] = None,
    activations: Optional[List] = None,
    dropout_probs: List[int] = None,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 1,
) -> Union[Ensemble, ProbabilisticEnsemble]:
    if trainers is None:
        build_trainers = True
    else:
        build_trainers = False
    networks, trainers = [], []
    for _ in range(ensemble_size):
        if probabilistic:
            networks.append(
                GaussianMLP(
                    in_size=in_size,
                    out_size=out_size,
                    features=features,
                    activations=activations,
                    dropout_probs=dropout_probs,
                    learning_rate=learning_rate,
                )
            )
        else:
            networks.append(
                MLP(
                    in_size=in_size,
                    out_size=out_size,
                    features=features,
                    activations=activations,
                    dropout_probs=dropout_probs,
                    learning_rate=learning_rate,
                )
            )
        if build_trainers:
            trainers.append(
                Trainer(
                    logger=pl.loggers.WandbLogger(
                        project="mbrl-under-uncertainty", log_model="all"
                    ),
                    callbacks=[
                        pl.callbacks.EarlyStopping(
                            monitor="train_loss",
                            mode="min",
                            min_delta=0.00,
                            patience=50,
                        )
                    ],
                    # verbose=True,
                    max_epochs=1000,
                )
            )

    if probabilistic:
        return ProbabilisticEnsemble(
            networks=networks,
            trainers=trainers,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    else:
        return Ensemble(
            networks=networks,
            trainers=trainers,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
