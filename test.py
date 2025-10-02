from nn.neuralnet import NeuralNetwork, load_architecture
from mnist.loader import load_mnistdata

model_name = "four_layers_accurate"
model_path = f"models/{model_name}"
weights_path = f"{model_path}/weights_and_biases/30e10m3l.npz"
architecture_path = f"{model_path}/architecture.json"

layer_spec = load_architecture(architecture_path)
nn = NeuralNetwork(layer_spec)
nn.load_weights_and_biases(weights_path)

mnistdata = load_mnistdata()
nn.load_data(mnistdata["test_images"], mnistdata["test_labels"])

nn.test()
