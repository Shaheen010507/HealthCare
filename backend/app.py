"""from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# Load models
diabetes_model = pickle.load(open("model.pkl", "rb"))
heart_model = pickle.load(open("heart_random_forest_model.pkl", "rb"))

# ---------------- Diabetes Prediction ----------------
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = diabetes_model.predict(features)[0]

        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({"prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- Heart Disease Prediction ----------------
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json()

        # Ensure correct order of features for the model
        feature_order = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        features = np.array([data[feat] for feat in feature_order]).reshape(1, -1)
        prediction = heart_model.predict(features)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return jsonify({"prediction": int(prediction), "result": result})
    except KeyError as e:
        return jsonify({"error": f"Missing feature in request: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- Home ----------------
@app.route("/", methods=["GET"])
def home():
    return "Flask AI Prediction API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

"""

"""from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# Load models
diabetes_model = pickle.load(open("model.pkl", "rb"))  # Diabetes model stays the same
heart_model = joblib.load("heart_random_forest_model.pkl")  # Only heart model changed
liver_model = joblib.load("liver_xgb_model.pkl") 
kidney_model = joblib.load("kidney_xgb_model.pkl") 
# ---------------- Diabetes Prediction ----------------
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = diabetes_model.predict(features)[0]

        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({"prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------- Heart Disease Prediction ----------------
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = heart_model.predict(features)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return jsonify({"prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    # ---------------- Liver Disease Prediction ----------------
@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = liver_model.predict(features)[0]

        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
        return jsonify({"prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    # ---------------- Kidney Disease Prediction ----------------
@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():
    try:
        data = request.get_json()

        # ---------------- Helper Function ----------------
        def safe_float(value, default=0):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        # ---------------- Preprocessing ----------------
        mapping_yes_no = {"Yes": 1, "No": 0}
        mapping_abnormal = {"Abnormal": 1, "Normal": 0}
        mapping_present = {"Present": 1, "Not Present": 0}
        mapping_appetite = {"Good": 1, "Poor": 0}

        # Replace text fields with numeric safely
        data_numeric = {
            "Age": safe_float(data.get("Age")),
            "bp": safe_float(data.get("bp")),
            "sg": safe_float(data.get("sg")),
            "al": safe_float(data.get("al")),
            "su": safe_float(data.get("su")),
            "rbc": mapping_abnormal.get(data.get("rbc"), 0),
            "pc": mapping_abnormal.get(data.get("pc"), 0),
            "pcc": mapping_present.get(data.get("pcc"), 0),
            "ba": mapping_present.get(data.get("ba"), 0),
            "bgr": safe_float(data.get("bgr")),
            "bu": safe_float(data.get("bu")),
            "sc": safe_float(data.get("sc")),
            "sod": safe_float(data.get("sod")),
            "pot": safe_float(data.get("pot")),
            "hemo": safe_float(data.get("hemo")),
            "pcv": safe_float(data.get("pcv")),
            "wbcc": safe_float(data.get("wbcc")),
            "rbcc": safe_float(data.get("rbcc")),
            "htn": mapping_yes_no.get(data.get("htn"), 0),
            "dm": mapping_yes_no.get(data.get("dm"), 0),
            "cad": mapping_yes_no.get(data.get("cad"), 0),
            "appetite": mapping_appetite.get(data.get("appetite"), 1),
            "pe": mapping_yes_no.get(data.get("pe"), 0),
            "ane": mapping_yes_no.get(data.get("ane"), 0)
        }

        # ---------------- Fill remaining features with 0 ----------------
        all_features = kidney_model.feature_names_in_
        features_array = [data_numeric.get(f, 0) for f in all_features]
        features_array = np.array(features_array).reshape(1, -1)

        # ---------------- Prediction ----------------
        prediction = kidney_model.predict(features_array)[0]
        result = "Chronic Kidney Disease Detected" if prediction == 1 else "No CKD"

        return jsonify({"prediction": int(prediction), "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- Home ----------------
@app.route("/", methods=["GET"])
def home():
    return "Flask AI Prediction API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
"""
### a comleted veersion  of app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# ---------------- Load Models ----------------
diabetes_model = pickle.load(open("model.pkl", "rb"))  # Diabetes model
heart_model = joblib.load("heart_random_forest_model.pkl")
liver_model = joblib.load("liver_xgb_model.pkl")
kidney_model = joblib.load("kidney_xgb_model.pkl")

# Load your trained model
model = joblib.load("breast_cancer_model.pkl")


# ---------------- Diabetes Prediction ----------------
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = diabetes_model.predict(features)[0]

        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({"prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- Heart Disease Prediction ----------------
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = heart_model.predict(features)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return jsonify({"prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- Liver Disease Prediction ----------------
@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = liver_model.predict(features)[0]

        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
        return jsonify({"prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- Kidney Disease Prediction ----------------
@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():
    try:
        data = request.get_json()

        def safe_float(value, default=0):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        mapping_yes_no = {"Yes": 1, "No": 0}
        mapping_abnormal = {"Abnormal": 1, "Normal": 0}
        mapping_present = {"Present": 1, "Not Present": 0}
        mapping_appetite = {"Good": 1, "Poor": 0}

        data_numeric = {
            "Age": safe_float(data.get("Age")),
            "bp": safe_float(data.get("bp")),
            "sg": safe_float(data.get("sg")),
            "al": safe_float(data.get("al")),
            "su": safe_float(data.get("su")),
            "rbc": mapping_abnormal.get(data.get("rbc"), 0),
            "pc": mapping_abnormal.get(data.get("pc"), 0),
            "pcc": mapping_present.get(data.get("pcc"), 0),
            "ba": mapping_present.get(data.get("ba"), 0),
            "bgr": safe_float(data.get("bgr")),
            "bu": safe_float(data.get("bu")),
            "sc": safe_float(data.get("sc")),
            "sod": safe_float(data.get("sod")),
            "pot": safe_float(data.get("pot")),
            "hemo": safe_float(data.get("hemo")),
            "pcv": safe_float(data.get("pcv")),
            "wbcc": safe_float(data.get("wbcc")),
            "rbcc": safe_float(data.get("rbcc")),
            "htn": mapping_yes_no.get(data.get("htn"), 0),
            "dm": mapping_yes_no.get(data.get("dm"), 0),
            "cad": mapping_yes_no.get(data.get("cad"), 0),
            "appetite": mapping_appetite.get(data.get("appetite"), 1),
            "pe": mapping_yes_no.get(data.get("pe"), 0),
            "ane": mapping_yes_no.get(data.get("ane"), 0)
        }

        all_features = kidney_model.feature_names_in_
        features_array = [data_numeric.get(f, 0) for f in all_features]
        features_array = np.array(features_array).reshape(1, -1)

        prediction = kidney_model.predict(features_array)[0]
        result = "Chronic Kidney Disease Detected" if prediction == 1 else "No CKD"

        return jsonify({"prediction": int(prediction), "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

##----------------- Breast Cancer Prediction ----------------
@app.route('/predict_breast_cancer', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.json
        features = data.get("features")  # Expecting a list of 9 values

        if not features or len(features) != 9:
            return jsonify({"error": "You must provide 9 features"}), 400

        # Convert to numpy array and reshape for model
        input_data = np.array([features])
        
        # Make prediction
        prediction = model.predict(input_data)[0]

        # Map result
        result = "Benign (2)" if prediction == 2 else "Malignant (4)"

        return jsonify({
            "prediction": int(prediction),
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Home ----------------
@app.route("/", methods=["GET"])
def home():
    return "Flask AI Prediction API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
