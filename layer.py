import numpy as np
from numpy.typing import NDArray
from util import ActivationFunc


class InputLayer:
    def __init__(self, n_neurons: int) -> None:
        self.activations: NDArray[np.float64] = np.zeros(n_neurons)

    def compute(self, in_activations: NDArray[np.float64]) -> NDArray[np.float64]:
        self.activations = in_activations
        return self.activations


class Layer:
    def __init__(
        self, n_inputs: int, n_neurons: int, activation_func: ActivationFunc
    ) -> None:
        self.activation_func: ActivationFunc = activation_func
        self.n_inputs: int = n_inputs
        """
        Weights is a matrix:
        Let n = n_inputs, m = n_neurons.
        Then we have
            [ w_00      w_10      ...       w_(m-1)0      ]
            [ w_01      w_11      ...       w_(m-1)1      ]
        W = [ w_02      w_12      ...       w_(m-1)2      ]
            [ ...       ...       ...       ...           ]
            [ w_0(n-1)  w_1(n-1)  ...       w_(m-1)(n-1)  ]
        where for any neuron 0 <= k < m, its weights are the vector [w_k0 ... w_k(n-1)]

        np.random.rand(a,b) constructs matrix with a rows, b columns
        """
        self.weights: NDArray[np.float64] = np.random.rand(n_inputs, n_neurons)
        self.bias: NDArray[np.float64] = np.random.rand(n_neurons)
        self.activations: NDArray[np.float64] = np.zeros(n_neurons)

    # z_k = σ(Wz_(k-1)+b) where z_k is the kth layer
    def compute(self, in_activations: NDArray[np.float64]) -> NDArray[np.float64]:
        assert in_activations.size == self.n_inputs
        self.activations = self.activation_func(
            self.weights @ in_activations + self.bias
        )
        return self.activations
