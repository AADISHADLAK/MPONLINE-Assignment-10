"""
app.py
------
Task 3: API Development

A Flask REST API that:
    - Loads the trained model (model.pkl), scaler (scaler.pkl) and
      feature order (feature_names.pkl).
    - Accepts patient clinical details as JSON input.
    - Returns the heart disease risk prediction as JSON.

Run locally:
    python app.py

Then test with:
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "age": 58, "sex": 1, "cp": 0, "trestbps": 145,
               "chol": 261, "fbs": 0, "restecg": 0, "thalach": 130,
               "exang": 1, "oldpeak": 2.8, "slope": 1, "ca": 2, "thal": 3
             }'
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ----------------------------------------------------------------------
# Load trained artifacts once, at startup
# ----------------------------------------------------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_names.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_NAMES = joblib.load(FEATURES_PATH)

FEATURE_RANGES_HELP = {
    "age": "Age in years (e.g. 45)",
    "sex": "1 = male, 0 = female",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure in mm Hg",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "Thalassemia (0-3)",
}


@app.route("/", methods=["GET"])
def home():
    """Simple landing page / health check with a basic HTML form."""
    try:
        return render_template("index.html", features=FEATURE_NAMES)
    except Exception:
        return jsonify({
            "message": "Heart Disease Prediction API is running.",
            "usage": "POST /predict with a JSON body of patient details.",
            "required_fields": FEATURE_NAMES,
            "field_meaning": FEATURE_RANGES_HELP,
        })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint used by Render / monitoring tools."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts patient details as JSON and returns the prediction as JSON.

    Expected JSON body (all 13 fields required):
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal
    """
    try:
        data = request.get_json(force=True)

        if data is None:
            return jsonify({"error": "Invalid or missing JSON body."}), 400

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing,
                "required_fields": FEATURE_NAMES,
            }), 400

        # Build feature vector in the exact order the model was trained on
        try:
            input_vector = [float(data[f]) for f in FEATURE_NAMES]
        except (TypeError, ValueError):
            return jsonify({"error": "All fields must be numeric."}), 400

        X = np.array(input_vector).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])
        probability = model.predict_proba(X_scaled)[0][prediction]

        result = {
            "prediction": "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected",
            "prediction_label": prediction,
            "confidence": round(float(probability), 4),
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
