from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# Load saved model
model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

model_info = joblib.load(
    os.path.join(MODEL_DIR, "model_info.pkl")
)

threshold = model_info["threshold"]
features = model_info["features"]


@app.route("/")
def home():

    return jsonify({
        "message": "Credit Default Risk Prediction API is running!",
        "model": model_info["model_name"],
        "threshold": threshold
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_data = request.get_json()

        values = []

        for feature in features:

            if feature not in input_data:
                return jsonify({
                    "error": f"Missing feature: {feature}"
                }), 400

            values.append(float(input_data[feature]))

        input_array = np.array(values).reshape(1, -1)

        probability = model.predict_proba(input_array)[0][1]

        prediction = int(probability >= threshold)

        result = (
            "Potential Default"
            if prediction == 1
            else "No Default"
        )

        return jsonify({

            "default_probability": round(
                probability * 100, 2
            ),

            "prediction": prediction,

            "result": result,

            "threshold": threshold,

            "model": model_info["model_name"]

        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Input values must be numeric."
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=False)