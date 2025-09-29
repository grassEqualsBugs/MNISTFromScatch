from typing_extensions import override
import numpy as np
from numpy.typing import NDArray
from nn.util import ActivationFunc


class InputLayer:
    def __init__(self, n_neurons: int) -> None:
        self.activations: NDArray[np.float64] = np.zeros(n_neurons)

    @override
    def __repr__(self) -> str:
        return f"InputLayer(activations={self.activations})"

    def set_activations(self, in_activations: NDArray[np.float64]) -> None:
        self.activations = in_activations


class Layer:
    def __init__(
        self, n_inputs: int, n_neurons: int, activation_func: ActivationFunc
    ) -> None:
        self.activation_func: ActivationFunc = activation_func
        self.n_inputs: int = n_inputs
        """
        Weights is a matrix:
        Let n = n_inputs-1, m = n_neurons-1.
        Then we have
            [ w_00      w_01      ...       w_0n ]
            [ w_10      w_11      ...       w_1n ]
        W = [ w_20      w_21      ...       w_2n ]
            [ ...       ...       ...       ...  ]
            [ w_m0      w_m1      ...       w_mn ]
        where for any neuron 0 <= k <= m, its weights are the row vector [w_k0, w_k1, ..., w_kn]
        """
        # create weights matrix with...              n_inputs rows, n_neurons cols
        self.weights: NDArray[np.float64] = np.random.rand(n_neurons, n_inputs)
        self.bias: NDArray[np.float64] = np.random.rand(n_neurons)
        self.activations: NDArray[np.float64] = np.zeros(n_neurons)

    @override
    def __repr__(self) -> str:
        return f"Layer(\nweights=\n{self.weights}, \nbias={self.bias}, \nactivations={self.activations}\n)"

    # z_k = σ(Wz_(k-1)+b) where z_k is the kth layer
    def compute(self, in_activations: NDArray[np.float64]) -> None:
        assert in_activations.size == self.n_inputs
        self.activations = self.activation_func(
            self.weights @ in_activations + self.bias
        )
