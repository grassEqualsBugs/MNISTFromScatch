from flask import Flask, jsonify, request
from network.nn.neuralnet import NeuralNetwork, load_architecture
import numpy as np
from scipy.ndimage import center_of_mass, shift

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

def preprocess_image(image_1d):
    """Centers the digit within the 28x28 grid."""
    # Reshape the flat array into a 28x28 grid
    image_2d = np.reshape(image_1d, (28, 28))

    # Calculate the center of mass
    com = center_of_mass(image_2d)

    # Calculate the shift required to move the center of mass to the center of the grid
    # The center of a 28x28 grid is at approximately 13.5, 13.5
    shift_arr = (13.5 - com[0], 13.5 - com[1])

    # Apply the shift to the image
    shifted_image = shift(image_2d, shift_arr, cval=0)

    # Flatten the processed image back to a 1D array and return
    return shifted_image.ravel()


@app.route("/predict", methods=["POST"])
def handle_prediction():
    data = request.get_json()
    activations = np.array(data["activations"])

    # Pre-process the image to center the digit
    processed_activations = preprocess_image(activations)

    return jsonify(
        {"prediction": nn.feed_forward(processed_activations).tolist()}
    )


if __name__ == "__main__":
    app.run(debug=True)