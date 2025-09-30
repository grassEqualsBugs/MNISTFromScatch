from typing import Union
from typing_extensions import override
from numpy.typing import NDArray
import numpy as np

from nn.layer import InputLayer, Layer
from nn.util import ActivationFunc


class NeuralNetwork:
    # layer_spec is a list of tuples. The first tuple is just one element long
    # because it's the input. Tuples are structured as (layerSize, activationFunction)
    def __init__(
        self, layer_spec: list[tuple[int, Union[ActivationFunc, None]]]
    ) -> None:
        # training data to be set later
        self.training_data: NDArray[np.float64]
        self.training_labels: NDArray[np.float64]

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
            "NeuralNetwork(\n"
            + str(self.input_layer)
            + ",\n"
            + ",\n".join(str(layer) for layer in self.layers)
            + "\n)\n"
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

    def load_training(
        self, training_data: NDArray[np.float64], training_labels: NDArray[np.float64]
    ):
        self.training_data = training_data
        self.training_labels = training_labels

    def train(self, n_epochs: int, minibatch_size: int, learning_rate: np.float64):
        for i in range(1, n_epochs + 1):
            tdata: NDArray[np.float64] = self.training_data.copy()
            tlabels: NDArray[np.float64] = self.training_labels.copy()

            print("Epoch number:", i)
            while len(tdata) > 0:
                idx = np.random.choice(
                    len(tdata), size=min(minibatch_size, len(tdata)), replace=False
                )
                minibatch_data = tdata[idx]
                minibatch_labels = tlabels[idx]

                tdata = np.delete(tdata, idx, axis=0)
                tlabels = np.delete(tlabels, idx, axis=0)
