import numpy as np

from nn.neuralnet import NeuralNetwork
from nn.util import Softmax, ReLU
from mnist.loader import load_mnistdata

nn = NeuralNetwork([(28 * 28, None), (30, ReLU), (10, Softmax)])

mnistdata = load_mnistdata()

nn.load_training(mnistdata["train_images"], mnistdata["train_labels"])
nn.train(30, 10, np.float64(3.0))
