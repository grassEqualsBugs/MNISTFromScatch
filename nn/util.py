import numpy as np
from numpy.typing import NDArray
from typing import Callable

ActivationFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


# Sigmoid(x) = 1/(1+e^(-x))
def Sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))


# Sigmoid'(x) = d/dx Sigmoid(x) = e^(-x)/(1+e^(-x))^2
def SigmoidPrime(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


# ReLU(x) = max(0, x)
def ReLU(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0, x)


# ReLUPrime(x) = d/dx ReLU(x) = piecewise shit
def ReLUPrime(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(x > 0, 1.0, 0.0)


# Softmax_i(z) = e^(z_i)/sum(e^(z_j))
def Softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # shift for numerical stability
    e_x: NDArray[np.float64] = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
