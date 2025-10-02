import numpy as np
import os

from nn.neuralnet import LayerSpec, NeuralNetwork
from nn.util import Sigmoid
from mnist.loader import load_mnistdata

model_name = "four_layers_accurate"
model_path = f"models/{model_name}"
weights_path = f"{model_path}/weights_and_biases"
architecture_path = f"{model_path}/architecture.json"

# make sure model directories exist
os.makedirs(weights_path, exist_ok=True)

# define network architecture
layer_spec: LayerSpec = [
    (28 * 28, None),
    (100, Sigmoid),
    (30, Sigmoid),
    (10, Sigmoid),
]
nn = NeuralNetwork(layer_spec)

# load data
mnistdata = load_mnistdata()
nn.load_data(mnistdata["train_images"], mnistdata["train_labels"])

# train the network
nn.train(30, 10, np.float64(3.0))

print("Training complete.")

# save the trained model
nn.save_architecture(architecture_path)
nn.save_weights_and_biases(f"{weights_path}/30e10m3l.npz")
