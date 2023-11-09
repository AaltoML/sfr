#!/usr/bin/env python3
from typing import Callable, Optional, Tuple, Union

import torch
from jaxtyping import Float

# InputData can be images
InputData = Float[torch.Tensor, "num_data ..."]

# OutputData can be classification or regression
ClassificationData = Float[torch.Tensor, "num_data"]
RegressionData = Float[torch.Tensor, "num_data output_dim"]
OutputData = Union[RegressionData, ClassificationData]
Data = Tuple[InputData, OutputData]

# Input = Float[torch.Tensor, "batch_size input_dim"]
OutputMean = Float[torch.Tensor, "batch_size output_dim"]
OutputVar = Float[torch.Tensor, "batch_size output_dim"]

FuncData = Float[torch.Tensor, "num_data output_dim"]
FuncMean = Float[torch.Tensor, "num_data output_dim"]
FuncVar = Float[torch.Tensor, "num_data output_dim"]

Alpha = Float[torch.Tensor, "num_data output_dim"]
Beta = Float[torch.Tensor, "num_data num_data output_dim"]
BetaDiag = Float[torch.Tensor, "num_data output_dim"]
Lambda = Float[torch.Tensor, "num_data output_dim"]
AlphaInducing = Float[torch.Tensor, "output_dim num_inducing"]
BetaInducing = Float[torch.Tensor, "output_dim num_inducing num_inducing"]

FuncData = Float[torch.Tensor, "num_data output_dim"]
InducingPoints = Float[torch.Tensor, "num_inducing ..."]


FullCov = Optional[bool]
Index = Optional[int]
NTK = Callable[[InputData, InputData, FullCov, Index], torch.Tensor]
NTK_single = Callable[[InputData, InputData, int, FullCov], torch.Tensor]

TestInput = Float[torch.Tensor, "num_test ..."]
