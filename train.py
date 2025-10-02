import numpy as np

from nn.neuralnet import NeuralNetwork
from nn.util import Sigmoid
from mnist.loader import load_mnistdata

nn = NeuralNetwork([(28 * 28, None), (30, Sigmoid), (10, Sigmoid)])

mnistdata = load_mnistdata()

nn.load_data(mnistdata["train_images"], mnistdata["train_labels"])
nn.train(30, 10, np.float64(3.0))
nn.save_network("models/30e10m3l.npz")
