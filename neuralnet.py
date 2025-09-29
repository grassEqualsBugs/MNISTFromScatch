from typing import Union
from layer import InputLayer, Layer
from util import ActivationFunc


class NeuralNetwork:
    # layerSpec array is just an array of tuples. the first tuple is
    # just one element long because its the input. the tuples are structured
    # in that (layerSize, activationFunction)
    def __init__(self, layer_spec: list[tuple[int, ActivationFunc]]) -> None:
        # init layers with Inputlayer
        self.layers: list[Union[InputLayer, Layer]] = [InputLayer(layer_spec[0][0])]
        for i in range(1, len(layer_spec)):
            layer_size, activation_func = layer_spec[i]
            self.layers.append(
                Layer(
                    n_inputs=layer_spec[i - 1][0],
                    n_neurons=layer_size,
                    activation_func=activation_func,
                )
            )
