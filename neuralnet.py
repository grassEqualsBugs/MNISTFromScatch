from typing import Union
from typing_extensions import override
from layer import InputLayer, Layer
from util import ActivationFunc, ReLU, Sigmoid
from numpy.typing import NDArray
import numpy as np


class NeuralNetwork:
    # layerSpec array is just an array of tuples. the first tuple is
    # just one element long because its the input. the tuples are structured
    # in that (layerSize, activationFunction)
    def __init__(
        self, layer_spec: list[tuple[int, Union[ActivationFunc, None]]]
    ) -> None:
        # init layers with Inputlayer
        self.layers: list[Union[InputLayer, Layer]] = [InputLayer(layer_spec[0][0])]
        for i in range(1, len(layer_spec)):
            layer_size, activation_func = layer_spec[i]
            assert activation_func
            self.layers.append(
                Layer(
                    n_inputs=layer_spec[i - 1][0],
                    n_neurons=layer_size,
                    activation_func=activation_func,
                )
            )

    @override
    def __repr__(self) -> str:
        return "[\n" + "\n".join(str(layer) for layer in self.layers) + "\n]\n"

    # (TODO): finish this
    def feed_forward(self, in_activations: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.layers[-1].activations


# quick test
nn = NeuralNetwork([(5, None), (3, ReLU), (3, ReLU), (2, Sigmoid)])
print(nn, "-" * 20)
assert isinstance(nn.layers[1], Layer)
print(nn.layers[1].compute(np.array([1.0, 0.5, 0.75, 0.75, 1.0])))
