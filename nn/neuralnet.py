from typing import Union
from typing_extensions import override
from numpy.typing import NDArray
import numpy as np
import json

from nn.layer import InputLayer, Layer
from nn.util import ActivationFunc, activation_func_deriv, ACTIVATION_MAP

LayerSpec = list[tuple[int, Union[ActivationFunc, None]]]


class NeuralNetwork:
    # layer_spec is a list of tuples. The first tuple is just one element long
    # because it's the input. Tuples are structured as (layerSize, activationFunction)
    def __init__(
        self,
        layer_spec: LayerSpec,
    ) -> None:
        self.layer_spec = layer_spec
        # data to be set later
        self.data: NDArray[np.float64]
        self.labels: NDArray[np.float64]

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
            "NeuralNetwork("
            + str(self.input_layer)
            + ",\n"
            + ",\n".join(str(layer) for layer in self.layers)
            + "\n)"
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

    def load_data(self, data: NDArray[np.float64], labels: NDArray[np.float64]) -> None:
        self.data = data
        self.labels = labels

    def layer_activation_func_deriv(self, l: int) -> ActivationFunc:
        return activation_func_deriv(self.layers[l].activation_func)

    # computes partial derivatives for weights and biases for one training example given by a and y
    def backprop(
        self,
        a: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        pderiv_C_b = [np.zeros(l.bias.shape) for l in self.layers]
        pderiv_C_w = [np.zeros(l.weights.shape) for l in self.layers]

        # error and gradients for the OUTPUT layer
        grad_C_a: NDArray[np.float64] = 2 * (a - y)
        error = grad_C_a * self.layer_activation_func_deriv(-1)(
            self.layers[-1].weighted_activations
        )
        pderiv_C_b[-1] = error
        pderiv_C_w[-1] = np.outer(error, self.layers[-2].activations)

        # loop backwards through the hidden layers
        for l in range(len(self.layers) - 2, -1, -1):
            # error for the current layer l
            error = (
                self.layers[l + 1].weights.T @ error
            ) * self.layer_activation_func_deriv(l)(self.layers[l].weighted_activations)

            # partial derivatives for layer l using the new error
            a_lm1 = (
                self.layers[l - 1].activations
                if l > 0
                else self.input_layer.activations
            )
            pderiv_C_b[l] = error
            pderiv_C_w[l] = np.outer(error, a_lm1)

        return (pderiv_C_w, pderiv_C_b)

    def handle_minibatch(
        self,
        data: NDArray[np.float64],
        labels: NDArray[np.float64],
        eta: np.float64,
    ) -> None:
        accumulated_pderiv_C_b = [np.zeros(l.bias.shape) for l in self.layers]
        accumulated_pderiv_C_w = [np.zeros(l.weights.shape) for l in self.layers]
        for i, x in enumerate(data):
            a: NDArray[np.float64] = self.feed_forward(x)
            y: NDArray[np.float64] = np.eye(1, 10, labels[i], dtype=np.float64).ravel()
            pderiv_C_w, pderiv_C_b = self.backprop(a, y)
            accumulated_pderiv_C_b = [
                prev + addition
                for prev, addition in zip(accumulated_pderiv_C_b, pderiv_C_b)
            ]
            accumulated_pderiv_C_w = [
                prev + addition
                for prev, addition in zip(accumulated_pderiv_C_w, pderiv_C_w)
            ]
        for l in range(len(self.layers)):
            self.layers[l].weights -= accumulated_pderiv_C_w[l] * eta / len(data)
            self.layers[l].bias -= accumulated_pderiv_C_b[l] * eta / len(data)

    def train(self, n_epochs: int, minibatch_size: int, eta: np.float64) -> None:
        for i in range(1, n_epochs + 1):
            print("Epoch number:", i)

            # shuffle training data and labels together
            p = np.random.permutation(len(self.data))
            tdata = self.data[p]
            tlabels = self.labels[p]

            for j in range(0, len(tdata), minibatch_size):
                self.handle_minibatch(
                    tdata[j : j + minibatch_size],
                    tlabels[j : j + minibatch_size],
                    eta,
                )

    def test(self):
        correct = 0
        for i, x in enumerate(self.data):
            a = self.feed_forward(x)
            y = self.labels[i]
            correct += max(a) == a[y]
        print(f"Network testing stats: {correct}/{len(self.data)}")

    def save_weights_and_biases(self, file_path: str) -> None:
        # we use a dictionary to store the parameters with descriptive keys
        save_dict = {}
        for i, layer in enumerate(self.layers):
            save_dict[f"weights_{i}"] = layer.weights
            save_dict[f"bias_{i}"] = layer.bias

        # np.savez_compressed is efficient for saving multiple numpy arrays
        np.savez_compressed(file_path, **save_dict)
        print(f"network weights saved to {file_path}")

    def load_weights_and_biases(self, file_path: str) -> None:
        # load the .npz file, which acts like a dictionary
        loaded_data = np.load(file_path)

        # check if the loaded data matches the network architecture
        if len(loaded_data.files) != len(self.layers) * 2:
            raise ValueError(
                "saved model architecture does not match current network architecture."
            )

        # assign the loaded weights and biases to the layers
        for i, layer in enumerate(self.layers):
            layer.weights = loaded_data[f"weights_{i}"]
            layer.bias = loaded_data[f"bias_{i}"]
        print(f"network weights loaded from {file_path}")

    def save_architecture(self, file_path: str):
        serializable_spec = []
        for size, func in self.layer_spec:
            func_name = None
            if func is not None:
                # find the function name by iterating through the map
                for name, map_func in ACTIVATION_MAP.items():
                    if func == map_func:
                        func_name = name
                        break
            serializable_spec.append((size, func_name))

        with open(file_path, "w") as f:
            json.dump(serializable_spec, f, indent=4)
        print(f"model architecture saved to {file_path}")


def load_architecture(file_path: str) -> list[tuple[int, Union[None, ActivationFunc]]]:
    with open(file_path, "r") as f:
        loaded_spec = json.load(f)

    # convert string names back to function objects
    layer_spec = [
        (size, ACTIVATION_MAP.get(func_name)) for size, func_name in loaded_spec
    ]

    print(f"model architecture loaded from {file_path}")
    return layer_spec
