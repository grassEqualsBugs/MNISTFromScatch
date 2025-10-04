from flask import Flask, jsonify, request
from network.nn.neuralnet import NeuralNetwork, load_architecture
import numpy as np
from scipy.ndimage import center_of_mass, shift

app = Flask(__name__)

# load in the model
model_name = "four_layers_relu"
model_path = f"network/models/{model_name}"
weights_path = f"{model_path}/weights_and_biases/60e30m0.1l.npz"
architecture_path = f"{model_path}/architecture.json"

layer_spec, cost_func_name, weight_init_name = load_architecture(architecture_path)
nn = NeuralNetwork(
    layer_spec, cost_func_name=cost_func_name, weight_init_name=weight_init_name
)
nn.load_weights_and_biases(weights_path)


def preprocess_image(image_1d):
    image_2d = np.reshape(image_1d, (28, 28))

    # calculate the center of mass
    com = center_of_mass(image_2d)

    # shift of a 28x28 grid is at approximately 13.5, 13.5
    shift_arr = (13.5 - com[0], 13.5 - com[1])
    shifted_image = shift(image_2d, shift_arr, cval=0)

    return shifted_image.ravel()


@app.route("/predict", methods=["POST"])
def handle_prediction():
    data = request.get_json()
    activations = np.array(data["activations"])

    # pre-process the image to center the digit
    processed_activations = preprocess_image(activations)

    return jsonify({"prediction": nn.feed_forward(processed_activations).tolist()})


if __name__ == "__main__":
    app.run(debug=True)
