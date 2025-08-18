from flask import Flask, request, jsonify
import pickle
import numpy as np
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# ===== Load Models =====
with open("model.pkl", "rb") as f:  # Diabetes model
    diabetes_model = pickle.load(f)

with open("heart_model.pkl", "rb") as f:  # Heart disease model
    heart_model = pickle.load(f)

with open("liver_model.pkl", "rb") as f:  # Liver model
    liver_model = pickle.load(f)

with open("feature_names.json", "r") as f:  # Liver model feature order
    liver_features = json.load(f)


@app.route("/")
def home():
    return "✅ Health Prediction API is running"


# ===== Diabetes Prediction =====
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json()
        features = [
            data.get('Pregnancies', 0),
            data.get('Glucose', 0),
            data.get('BloodPressure', 0),
            data.get('SkinThickness', 0),
            data.get('Insulin', 0),
            data.get('BMI', 0.0),
            data.get('DiabetesPedigreeFunction', 0.0),
            data.get('Age', 0)
        ]
        prediction = diabetes_model.predict([np.array(features)])[0]
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes"
        return jsonify({"prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ===== Heart Disease Prediction =====
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json()

        # Features expected by your trained model
        features = [
            data.get('age', 0),
            data.get('sex', 0),       # 0 = female, 1 = male
            data.get('cp', 0),        # chest pain type
            data.get('trestbps', 0),  # resting blood pressure
            data.get('chol', 0),      # cholesterol
            data.get('fbs', 0),       # fasting blood sugar
            data.get('restecg', 0),   # resting ECG results
            data.get('thalch', 0),    # maximum heart rate
            data.get('exang', 0),     # exercise-induced angina
            data.get('oldpeak', 0.0), # ST depression
            data.get('slope', 0),     # slope of ST segment
            data.get('ca', 0),        # major vessels
            data.get('thal', 0)       # thalassemia
        ]

        features_array = np.array(features).reshape(1, -1)
        prediction = heart_model.predict(features_array)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return jsonify({"prediction": int(prediction), "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ===== Liver Disease Prediction =====
@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    try:
        data = request.get_json()

        # Extract features in correct order for liver model
        features = [data.get(feat, 0) for feat in liver_features]
        features_array = np.array(features).reshape(1, -1)

        prediction = liver_model.predict(features_array)[0]
        proba = liver_model.predict_proba(features_array)[0][1]

        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(proba), 3),
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
