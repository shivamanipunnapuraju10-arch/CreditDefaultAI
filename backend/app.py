from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Configure Flask to serve the frontend
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

CORS(app)

# Load trained machine learning model
model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

model_info = joblib.load(
    os.path.join(MODEL_DIR, "model_info.pkl")
)

threshold = model_info["threshold"]
features = model_info["features"]


# Display frontend website
@app.route("/")
def home():
    return app.send_static_file("index.html")


# Backend health check
@app.route("/health")
def health():
    return jsonify({
        "message": "Credit Default Risk Prediction API is running!",
        "model": model_info["model_name"],
        "threshold": threshold
    })


# Credit default prediction
@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_data = request.get_json(silent=True)

        if not isinstance(input_data, dict):
            return jsonify({
                "error": "Invalid JSON input."
            }), 400

        values = []

        for feature in features:

            if feature not in input_data:
                return jsonify({
                    "error": f"Missing feature: {feature}"
                }), 400

            values.append(float(input_data[feature]))

        input_array = np.array(values).reshape(1, -1)

        if not np.isfinite(input_array).all():
            return jsonify({
                "error": "Input values must be finite numbers."
            }), 400

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

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )