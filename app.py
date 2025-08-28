from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")


# Load pickle models
with open("C:\\Users\\nikhi\\Downloads\\SLR&MLR\\SLR_model.pkl", "rb") as f:
    simple_model = pickle.load(f)

with open("C:\\Users\\nikhi\\Downloads\\SLR&MLR\\MLR_model.pkl", "rb") as f:
    multiple_model = pickle.load(f)


@app.route("/predict_simple", methods=["POST"])
def predict_simple():
    try:
        data = request.get_json()
        exp = float(data["experience"])
        prediction = simple_model.predict(np.array([[exp]]))
        return jsonify({"predicted_salary": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_multiple", methods=["POST"])
def predict_multiple():
    try:
        data = request.get_json()
        rnd = float(data["rnd"])
        admin = float(data["admin"])
        marketing = float(data["marketing"])
        state = int(data["state"])

        features = np.array([[rnd, admin, marketing, state]])
        prediction = multiple_model.predict(features)
        return jsonify({"predicted_profit": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
