#!/usr/bin/env python3
from typing import Any, Callable, Iterator, NamedTuple, Optional, Tuple, Union

import torch
from jaxtyping import Float

InputData = Float[torch.Tensor, "num_data input_dim"]
OutputData = Float[torch.Tensor, "num_data output_dim"]
Data = Tuple[InputData, OutputData]

Input = Float[torch.Tensor, "batch_size input_dim"]
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
InducingPoints = Float[torch.Tensor, "num_inducing input_dim"]


FullCov = Optional[bool]
Index = Optional[int]
NTK = Callable[[InputData, InputData, FullCov, Index], Float[torch.Tensor, ""]]
NTK_single = Callable[[InputData, InputData, int, FullCov], Float[torch.Tensor, ""]]

TestInput = Float[torch.Tensor, "num_test input_dim"]

# from torchtyping import TensorType
# InputData = TensorType["num_data", "input_dim"]
# OutputData = TensorType["num_data", "output_dim"]
# Data = Tuple[InputData, OutputData]

# Input = TensorType["batch_size, input_dim"]
# OutputMean = TensorType["batch_size, output_dim"]
# OutputVar = TensorType["batch_size, output_dim"]

# FuncData = TensorType["num_data", "output_dim"]
# FuncMean = TensorType["num_data", "output_dim"]
# FuncVar = TensorType["num_data", "output_dim"]

# Alpha = TensorType["num_data", "output_dim"]
# Beta = TensorType["num_data", "num_data", "output_dim"]
# BetaDiag = TensorType["num_data", "output_dim"]
# Lambda = TensorType["num_data", "output_dim"]
# AlphaInducing = TensorType["output_dim", "num_inducing"]
# BetaInducing = TensorType["output_dim", "num_inducing", "num_inducing"]

# FuncData = TensorType["num_data", "output_dim"]
# InducingPoints = TensorType["num_inducing", "input_dim"]


# FullCov = Optional[bool]
# Index = Optional[int]
# NTK = Callable[[InputData, InputData, FullCov, Index], TensorType[""]]
# NTK_single = Callable[[InputData, InputData, int, FullCov], TensorType[""]]

# TestInput = TensorType["num_test", "input_dim"]
