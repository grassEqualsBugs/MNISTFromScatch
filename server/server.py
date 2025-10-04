from flask import Flask, jsonify, request
from network.nn.neuralnet import NeuralNetwork, load_architecture

from network.mnist.loader import load_mnistdata
import numpy as np

app = Flask(__name__)

model_name = "four_layers_relu"
model_path = f"network/models/{model_name}"
weights_path = f"{model_path}/weights_and_biases/60e30m0.1l.npz"
architecture_path = f"{model_path}/architecture.json"

layer_spec, cost_func_name, weight_init_name = load_architecture(architecture_path)
nn = NeuralNetwork(
    layer_spec, cost_func_name=cost_func_name, weight_init_name=weight_init_name
)
nn.load_weights_and_biases(weights_path)
mnistdata = load_mnistdata()
nn.load_data(mnistdata["test_images"], mnistdata["test_labels"])

nn.test()


@app.route("/predict", methods=["POST"])
def handle_prediction():
    data = request.get_json()
    return jsonify(
        {"prediction": nn.feed_forward(np.array(data["activations"]).ravel()).tolist()}
    )


if __name__ == "__main__":
    app.run(debug=True)
