from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/data", methods=["POST"])
def handle_data():
    data = request.get_json()
    print(f"Received data: {data}")
    return jsonify({"status": "success", "data_received": data})


if __name__ == "__main__":
    app.run(debug=True)
