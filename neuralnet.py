from typing import Union
from typing_extensions import override
from layer import InputLayer, Layer
from util import ActivationFunc, ReLU, Sigmoid
from numpy.typing import NDArray
import numpy as np


class NeuralNetwork:
    # layer_spec is a list of tuples. The first tuple is just one element long
    # because it's the input. Tuples are structured as (layerSize, activationFunction)
    def __init__(
        self, layer_spec: list[tuple[int, Union[ActivationFunc, None]]]
    ) -> None:
        # first tuple defines the input layer
        self.input_layer: InputLayer = InputLayer(layer_spec[0][0])

        # all other tuples define regular layers
        self.layers: list[Layer] = []
        for (prev_size, _), (layer_size, activation_func) in zip(
            layer_spec[:-1], layer_spec[1:]
        ):
            assert activation_func
            self.layers.append(
                Layer(
                    n_inputs=prev_size,
                    n_neurons=layer_size,
                    activation_func=activation_func,
                )
            )

    @override
    def __repr__(self) -> str:
        return (
            "[\n"
            + str(self.input_layer)
            + "\n"
            + "\n".join(str(layer) for layer in self.layers)
            + "\n]\n"
        )

    def feed_forward(self, in_activations: NDArray[np.float64]) -> NDArray[np.float64]:
        # set activations for input layer
        self.input_layer.set_activations(in_activations)

        # pass activations forward through each layer
        prev_activations = self.input_layer.activations
        for layer in self.layers:
            layer.compute(prev_activations)
            prev_activations = layer.activations

        return prev_activations


# quick test
nn = NeuralNetwork([(5, None), (3, ReLU), (3, ReLU), (2, Sigmoid)])
print(nn, "-" * 20)
print(nn.feed_forward(np.array([1.0, 0.5, 0.75, 0.75, 1.0])))
