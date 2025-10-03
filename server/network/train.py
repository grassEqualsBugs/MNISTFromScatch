import numpy as np
import os
import time

from nn.neuralnet import LayerSpec, NeuralNetwork
from nn.util import ReLU, Softmax
from mnist.loader import load_mnistdata

model_name = "four_layers_relu"
model_path = f"models/{model_name}"
weights_path = f"{model_path}/weights_and_biases"
architecture_path = f"{model_path}/architecture.json"

# make sure model directories exist
os.makedirs(weights_path, exist_ok=True)

# define network architecture
layer_spec: LayerSpec = [
    (28 * 28, None),
    (100, ReLU),
    (30, ReLU),
    (10, Softmax),
]
nn = NeuralNetwork(
    layer_spec, cost_func_name="CrossEntropyCost", weight_init_name="Kaiming"
)

# load data
mnistdata = load_mnistdata()
nn.load_data(mnistdata["train_images"], mnistdata["train_labels"])

# train the network
before_training = time.time()
nn.train(60, 30, np.float64(0.1))
print(
    f"Training complete, total training time was {time.time() - before_training} seconds."
)

# save the trained model
nn.save_architecture(architecture_path)
nn.save_weights_and_biases(f"{weights_path}/60e30m0.1l.npz")
