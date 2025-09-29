import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union

ActivationFunc = Callable[[Union[float, NDArray[np.float64]]], NDArray[np.float64]]


# Sigmoid(x) = 1/(1+e^(-x))
def Sigmoid(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))


# ReLU(x) = max(0, x)
def ReLU(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
    return np.maximum(0, x)
