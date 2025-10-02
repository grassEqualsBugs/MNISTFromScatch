from nn.neuralnet import NeuralNetwork
from nn.util import Sigmoid
from mnist.loader import load_mnistdata

nn = NeuralNetwork([(28 * 28, None), (30, Sigmoid), (10, Sigmoid)])

mnistdata = load_mnistdata()

nn.load_data(mnistdata["test_images"], mnistdata["test_labels"])
nn.load_network("models/30e10m3l.npz")
nn.test()
