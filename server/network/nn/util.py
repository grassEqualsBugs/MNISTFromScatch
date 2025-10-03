import numpy as np
from numpy.typing import NDArray
from typing import Callable

ActivationFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def activation_func_deriv(func: ActivationFunc):
    if func == Sigmoid:
        return SigmoidPrime
    elif func == ReLU:
        return ReLUPrime
    else:
        raise ValueError(
            f"Activation function {func} does not have a defined derivative in code"
        )


# Sigmoid(x) = 1/(1+e^(-x))
def Sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty_like(x)

    pos_mask = x >= 0
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    neg_mask = ~pos_mask  # invert the mask
    result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))

    return result


# Sigmoid'(x) = d/dx Sigmoid(x) = e^(-x)/(1+e^(-x))^2
def SigmoidPrime(x: NDArray[np.float64]) -> NDArray[np.float64]:
    s = Sigmoid(x)
    return s * (1 - s)


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


ACTIVATION_MAP = {
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
    "Softmax": Softmax,
}


class CostFunc:
    def cost(self, a: NDArray[np.float64], y: NDArray[np.float64]) -> np.float64:
        raise NotImplementedError

    def derivative(
        self, a: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raise NotImplementedError


class QuadraticCost(CostFunc):
    def cost(self, a: NDArray[np.float64], y: NDArray[np.float64]) -> np.float64:
        return np.linalg.norm(a - y) ** 2

    def derivative(
        self, a: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return 2 * (a - y)


class CrossEntropyCost(CostFunc):
    def cost(self, a: NDArray[np.float64], y: NDArray[np.float64]) -> np.float64:
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def derivative(
        self, a: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return a - y


COST_MAP = {"QuadraticCost": QuadraticCost, "CrossEntropyCost": CrossEntropyCost}


def Kaiming(n_neurons, n_inputs):
    return np.random.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)


def Standard(n_neurons, n_inputs):
    return np.random.randn(n_neurons, n_inputs)


WEIGHT_INIT_MAP = {
    "Kaiming": Kaiming,
    "Standard": Standard,
}
