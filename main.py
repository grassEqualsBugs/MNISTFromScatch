import numpy as np

from nn.neuralnet import NeuralNetwork
from nn.util import Sigmoid, ReLU

nn = NeuralNetwork([(28 * 28, None), (15, ReLU), (15, ReLU), (10, Sigmoid)])
nn.feed_forward(np.random.rand(28 * 28))
print(nn)
