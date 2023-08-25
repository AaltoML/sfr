#!/usr/bin/env python3
from typing import Any, Callable, Iterator, NamedTuple, Optional, Tuple, Union

from torchtyping import TensorType


InputData = TensorType["num_data", "input_dim"]
OutputData = TensorType["num_data", "output_dim"]
Data = Tuple[InputData, OutputData]

Input = TensorType["batch_size, input_dim"]
OutputMean = TensorType["batch_size, output_dim"]
OutputVar = TensorType["batch_size, output_dim"]

FuncData = TensorType["num_data", "output_dim"]
FuncMean = TensorType["num_data", "output_dim"]
FuncVar = TensorType["num_data", "output_dim"]

Alpha = TensorType["num_data", "output_dim"]
Beta = TensorType["num_data", "num_data", "output_dim"]
BetaDiag = TensorType["num_data", "output_dim"]
Lambda = TensorType["num_data", "output_dim"]
AlphaInducing = TensorType["output_dim", "num_inducing"]
BetaInducing = TensorType["output_dim", "num_inducing", "num_inducing"]

FuncData = TensorType["num_data", "output_dim"]
InducingPoints = TensorType["num_inducing", "input_dim"]


FullCov = Optional[bool]
NTK = Callable[[InputData, InputData, FullCov], TensorType[""]]
NTK_single = Callable[[InputData, InputData, int, FullCov], TensorType[""]]

TestInput = TensorType["num_test", "input_dim"]
