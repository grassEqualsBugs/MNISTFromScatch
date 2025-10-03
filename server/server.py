from flask import Flask

app = Flask(__name__)


# NeuralNetwork API result route
@app.route("/neuralNetwork")
def neuralNetwork():
    return {"Hello": ["World!", "Flask!", "React!"]}


if __name__ == "__main__":
    app.run(debug=True)
