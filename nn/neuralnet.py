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

    def average_cost(self, data: NDArray[np.float64], labels: NDArray[np.float64]):
        cost: np.float64 = np.float64(0.0)
        # compute average cost
        for i, x in enumerate(data):
            """
            x is the training data for one training example
            y is the correct vector for the label of x
            a is the prediction vector output by the network with input x
            C is the quadratic cost function, calculated by ||(a-y)||^2
            """
            a: NDArray[np.float64] = self.feed_forward(x)
            y: NDArray[np.float64] = np.eye(1, 10, labels[i], dtype=np.float64).ravel()
            C: np.float64 = np.sum((a - y) ** 2)
            cost += C
        return cost / len(data)

    def handle_minibatch(
        self,
        data: NDArray[np.float64],
        labels: NDArray[np.float64],
        learning_rate: np.float64,
    ):
        pass

    def train(self, n_epochs: int, minibatch_size: int, learning_rate: np.float64):
        for i in range(1, n_epochs + 1):
            print("Epoch number:", i)

            # shuffle training data and labels together
            p = np.random.permutation(len(self.training_data))
            tdata = self.training_data[p]
            tlabels = self.training_labels[p]

            for j in range(0, len(tdata), minibatch_size):
                self.handle_minibatch(
                    tdata[j : j + minibatch_size],
                    tlabels[j : j + minibatch_size],
                    learning_rate,
                )
