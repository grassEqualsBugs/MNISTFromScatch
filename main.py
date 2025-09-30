import numpy as np

from nn.neuralnet import NeuralNetwork
from nn.util import Softmax, ReLU
from mnist.loader import train_images, train_labels

nn = NeuralNetwork([(28 * 28, None), (30, ReLU), (10, Softmax)])
nn.load_training(train_images, train_labels)
nn.train(30, 10, np.float64(3.0))
