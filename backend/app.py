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
### a completed version  of app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np
import pandas as pd

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
    
    #------------diabetivs batch upload-----------
    
@app.route("/predict_diabetes_batch", methods=["POST"])
def predict_diabetes_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Read CSV file with encoding fix
        df = pd.read_csv(file, encoding='latin1')

        features = df.values
        predictions = diabetes_model.predict(features)

        results = []
        for i, row in df.iterrows():
            result = "Diabetic" if predictions[i] == 1 else "Non-Diabetic"
            results.append({
                "patient_data": row.to_dict(),
                "prediction": int(predictions[i]),
                "result": result
            })

        return jsonify({"predictions": results})

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

# ---------------- Heart Disease Batch Prediction (CSV Upload) ----------------
@app.route("/predict_heart_batch", methods=["POST"])
def predict_heart_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Read CSV with encoding fix to avoid errors
        df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')

        # Keep only the columns the model expects
        expected_features = list(heart_model.feature_names_in_)
        df = df[expected_features]

        # Predict
        features = df.values
        predictions = heart_model.predict(features)

        # Prepare results
        results = []
        for i, row in df.iterrows():
            result_text = "Heart Disease Detected" if predictions[i] == 1 else "No Heart Disease"
            results.append({
                "patient_data": row.to_dict(),
                "prediction": int(predictions[i]),
                "result": result_text
            })

        return jsonify({"predictions": results})

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

# ---------------- Liver Batch Prediction (CSV Upload) ----------------


"""@app.route('/predict_liver_batch', methods=['POST'])
def predict_liver_batch():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Dummy prediction logic (replace with ML model later)
        predictions = [{"row": i+1, "result": "No Liver Disease"} for i in range(len(df))]

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 400"""
##a new version
# ---------------- Liver Batch Prediction ----------------
@app.route("/predict_liver_batch", methods=["POST"])
def predict_liver_batch():
    try:
        # Check if file is uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Read CSV
        df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # ---------------- Mapping categorical data ----------------
        mapping_gender = {"male": 1, "female": 0}
        mapping_yes_no = {"yes": 1, "no": 0}

        features_list = []
        all_features = [f.lower() for f in liver_model.feature_names_in_]

        for _, row in df.iterrows():
            # Map categorical features and default numeric values
            data_numeric = {
                "age": float(row.get("age", 0)),
                "gender": mapping_gender.get(str(row.get("gender", "male")).strip().lower(), 0),
                "bilirubin": float(row.get("bilirubin", 0)),
                "alkaline_phosphatase": float(row.get("alkaline_phosphatase", 0)),
                "sgot": float(row.get("sgot", 0)),
                "albumin": float(row.get("albumin", 0)),
                "protime": float(row.get("protime", 0)),
                "fatigue": mapping_yes_no.get(str(row.get("fatigue", "no")).strip().lower(), 0),
                "anorexia": mapping_yes_no.get(str(row.get("anorexia", "no")).strip().lower(), 0),
                "nausea": mapping_yes_no.get(str(row.get("nausea", "no")).strip().lower(), 0)
            }

            # Align features with model
            features_array = [data_numeric.get(f, 0) for f in all_features]
            features_list.append(features_array)

        features_array = np.array(features_list)
        predictions = liver_model.predict(features_array)

        # ---------------- Build results ----------------
        results = []
        for i, row in df.iterrows():
            result_text = "Liver Disease Detected" if predictions[i] == 1 else "No Liver Disease"
            results.append({
                "patient_data": row.to_dict(),
                "prediction": int(predictions[i]),
                "result": result_text
            })

        return jsonify({"predictions": results})

    except Exception as e:
        print("Error in liver batch:", e)
        return jsonify({"error": str(e)}), 400




## suma testing
"""@app.route("/predict_liver_batch", methods=["POST"])
def predict_liver_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')

        # Strip whitespace & lowercase column names
        df.columns = df.columns.str.strip().str.lower()

        # Mapping categorical features
        gender_map = {"male": 1, "female": 0}

        # List of model features
        model_features = [f.lower() for f in liver_model.feature_names_in_]

        features_list = []
        for _, row in df.iterrows():
            # Default numeric values
            row_data = {
                "age": float(row.get("age", 0)),
                "gender": gender_map.get(str(row.get("gender", "male")).strip().lower(), 0),
                "total_bilirubin": float(row.get("total_bilirubin", row.get("bilirubin_total", 0))),
                "direct_bilirubin": float(row.get("direct_bilirubin", row.get("bilirubin_direct", 0))),
                "alkaline_phosphotase": float(row.get("alkaline_phosphotase", row.get("alk_phos", 0))),
                "alamine_aminotransferase": float(row.get("alamine_aminotransferase", row.get("alt", 0))),
                "aspartate_aminotransferase": float(row.get("aspartate_aminotransferase", row.get("ast", 0))),
                "albumin": float(row.get("albumin", 0)),
                "total_proteins": float(row.get("total_proteins", row.get("proteins_total", 0))),
                "albumin_and_globulin_ratio": float(row.get("albumin_and_globulin_ratio", row.get("ag_ratio", 0)))
            }

            # Ensure the order matches model_features
            features_array = [row_data.get(f, 0) for f in model_features]
            features_list.append(features_array)

        features_array = np.array(features_list)
        predictions = liver_model.predict(features_array)

        results = []
        for i, row in df.iterrows():
            result_text = "Liver Disease Detected" if predictions[i] == 1 else "No Liver Disease"
            results.append({
                "patient_data": row.to_dict(),
                "prediction": int(predictions[i]),
                "result": result_text
            })

        return jsonify({"predictions": results})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

"""

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
     
       

##----kideney batch upload----
# ---------------- Kidney Disease Batch Prediction ----------------
"""@app.route("/predict_kidney_batch", methods=["POST"])
def predict_kidney_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')

        # ---------------- Mappings ----------------
        mapping_yes_no = {"Yes": 1, "No": 0}
        mapping_abnormal = {"Abnormal": 1, "Normal": 0}
        mapping_present = {"Present": 1, "Not Present": 0}
        mapping_appetite = {"Good": 1, "Poor": 0}

        # ---------------- Prepare features ----------------
        features_list = []
        all_features = kidney_model.feature_names_in_

        for _, row in df.iterrows():
            # Convert available columns to numeric, missing columns get default 0
            data_numeric = {
                "Age": row.get("Age", 0),
                "bp": row.get("bp", 0),
                "sg": row.get("sg", 0),
                "al": row.get("al", 0),
                "su": row.get("su", 0),
                "rbc": mapping_abnormal.get(row.get("rbc"), 0),
                "pc": mapping_abnormal.get(row.get("pc"), 0),
                "pcc": mapping_present.get(row.get("pcc"), 0),
                "ba": mapping_present.get(row.get("ba"), 0),
                "bgr": row.get("bgr", 0),
                "bu": row.get("bu", 0),
                "sc": row.get("sc", 0),
                "sod": row.get("sod", 0),
                "pot": row.get("pot", 0),
                "hemo": row.get("hemo", 0),
                "pcv": row.get("pcv", 0),
                "wbcc": row.get("wbcc", 0),
                "rbcc": row.get("rbcc", 0),
                "htn": mapping_yes_no.get(row.get("htn"), 0),
                "dm": mapping_yes_no.get(row.get("dm"), 0),
                "cad": mapping_yes_no.get(row.get("cad"), 0),
                "appetite": mapping_appetite.get(row.get("appetite"), 1),
                "pe": mapping_yes_no.get(row.get("pe"), 0),
                "ane": mapping_yes_no.get(row.get("ane"), 0)
            }

            # Match the order of features in the trained model
            features_array = [data_numeric.get(f, 0) for f in all_features]
            features_list.append(features_array)

        features_array = np.array(features_list)
        predictions = kidney_model.predict(features_array)

        # ---------------- Build results ----------------
        results = []
        for i, row in df.iterrows():
            result_text = "Chronic Kidney Disease Detected" if predictions[i] == 1 else "No CKD"
            results.append({
                "patient_data": row.to_dict(),
                "prediction": int(predictions[i]),
                "result": result_text
            })

        return jsonify({"predictions": results})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

"""

#the newv version
"""@app.route("/predict_kidney_batch", methods=["POST"])
def predict_kidney_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')

        # Mapping categorical values to numeric (must match single prediction)
        mapping_yes_no = {"Yes": 1, "No": 0}
        mapping_abnormal = {"Abnormal": 1, "Normal": 0}
        mapping_present = {"Present": 1, "Not Present": 0}
        mapping_appetite = {"Good": 1, "Poor": 0}

        # Prepare features
        features_list = []
        for _, row in df.iterrows():
            data_numeric = {
                "Age": float(row.get("Age", 0)),
                "bp": float(row.get("bp", 0)),
                "sg": float(row.get("sg", 0)),
                "al": float(row.get("al", 0)),
                "su": float(row.get("su", 0)),
                "rbc": mapping_abnormal.get(row.get("rbc", "Normal"), 0),
                "pc": mapping_abnormal.get(row.get("pc", "Normal"), 0),
                "pcc": mapping_present.get(row.get("pcc", "Not Present"), 0),
                "ba": mapping_present.get(row.get("ba", "Not Present"), 0),
                "bgr": float(row.get("bgr", 0)),
                "bu": float(row.get("bu", 0)),
                "sc": float(row.get("sc", 0)),
                "sod": float(row.get("sod", 0)),
                "pot": float(row.get("pot", 0)),
                "hemo": float(row.get("hemo", 0)),
                "pcv": float(row.get("pcv", 0)),
                "wbcc": float(row.get("wbcc", 0)),
                "rbcc": float(row.get("rbcc", 0)),
                "htn": mapping_yes_no.get(row.get("htn", "No"), 0),
                "dm": mapping_yes_no.get(row.get("dm", "No"), 0),
                "cad": mapping_yes_no.get(row.get("cad", "No"), 0),
                "appetite": mapping_appetite.get(row.get("appetite", "Good"), 1),
                "pe": mapping_yes_no.get(row.get("pe", "No"), 0),
                "ane": mapping_yes_no.get(row.get("ane", "No"), 0)
            }

            # Ensure columns match model
            all_features = kidney_model.feature_names_in_
            features_array = [data_numeric.get(f, 0) for f in all_features]
            features_list.append(features_array)

        features_array = np.array(features_list)
        predictions = kidney_model.predict(features_array)

        results = []
        for i, row in df.iterrows():
            result_text = "Chronic Kidney Disease Detected" if predictions[i] == 1 else "No CKD"
            results.append({
                "patient_data": row.to_dict(),
                "prediction": int(predictions[i]),
                "result": result_text
            })

        return jsonify({"predictions": results})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400"""
## the very new version

"""@app.route("/predict_kidney_batch", methods=["POST"])
def predict_kidney_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Mapping categorical values to numeric
        mapping_yes_no = {"Yes": 1, "No": 0}
        mapping_abnormal = {"Abnormal": 1, "Normal": 0}
        mapping_present = {"Present": 1, "Not Present": 0}
        mapping_appetite = {"Good": 1, "Poor": 0}

        features_list = []
        for _, row in df.iterrows():
            # Strip spaces and standardize values
            rbc_val = str(row.get("rbc", "Normal")).strip().capitalize()
            pc_val = str(row.get("pc", "Normal")).strip().capitalize()
            pcc_val = str(row.get("pcc", "Not Present")).strip().capitalize()
            ba_val = str(row.get("ba", "Not Present")).strip().capitalize()
            htn_val = str(row.get("htn", "No")).strip().capitalize()
            dm_val = str(row.get("dm", "No")).strip().capitalize()
            cad_val = str(row.get("cad", "No")).strip().capitalize()
            appetite_val = str(row.get("appetite", "Good")).strip().capitalize()
            pe_val = str(row.get("pe", "No")).strip().capitalize()
            ane_val = str(row.get("ane", "No")).strip().capitalize()

            data_numeric = {
                "Age": float(row.get("Age", 0)),
                "bp": float(row.get("bp", 0)),
                "sg": float(row.get("sg", 0)),
                "al": float(row.get("al", 0)),
                "su": float(row.get("su", 0)),
                "rbc": mapping_abnormal.get(rbc_val, 0),
                "pc": mapping_abnormal.get(pc_val, 0),
                "pcc": mapping_present.get(pcc_val, 0),
                "ba": mapping_present.get(ba_val, 0),
                "bgr": float(row.get("bgr", 0)),
                "bu": float(row.get("bu", 0)),
                "sc": float(row.get("sc", 0)),
                "sod": float(row.get("sod", 0)),
                "pot": float(row.get("pot", 0)),
                "hemo": float(row.get("hemo", 0)),
                "pcv": float(row.get("pcv", 0)),
                "wbcc": float(row.get("wbcc", 0)),
                "rbcc": float(row.get("rbcc", 0)),
                "htn": mapping_yes_no.get(htn_val, 0),
                "dm": mapping_yes_no.get(dm_val, 0),
                "cad": mapping_yes_no.get(cad_val, 0),
                "appetite": mapping_appetite.get(appetite_val, 1),
                "pe": mapping_yes_no.get(pe_val, 0),
                "ane": mapping_yes_no.get(ane_val, 0)
            }

            # Ensure columns match model
            all_features = kidney_model.feature_names_in_
            features_array = [data_numeric.get(f, 0) for f in all_features]
            features_list.append(features_array)

        features_array = np.array(features_list)
        predictions = kidney_model.predict(features_array)

        results = []
        for i, row in df.iterrows():
            result_text = "Chronic Kidney Disease Detected" if predictions[i] == 1 else "No CKD"
            results.append({
                "patient_data": row.to_dict(),
                "prediction": int(predictions[i]),
                "result": result_text
            })

        return jsonify({"predictions": results})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

"""
####last try
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        # Ensure correct columns
        expected_cols = [
            "Age", "Blood_Pressure", "Specific_Gravity", "Albumin", "Sugar",
            "Red_Blood_Cells", "Pus_Cell", "Pus_Cell_Clumps", "Bacteria",
            "Hypertension", "Diabetes_Mellitus", "Coronary_Artery_Disease",
            "Appetite", "Pedal_Edema"
        ]
        df = df[expected_cols]

        # Predict
        df['prediction'] = model.predict(df)
        df['result'] = df['prediction'].apply(lambda x: "CKD Detected" if x==1 else "No CKD")

        # Convert to JSON
        results = df.to_dict(orient="records")
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
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
