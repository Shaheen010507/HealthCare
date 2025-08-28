# train_kidney_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib

# ---------------------------
# Step 1: Load Dataset
# ---------------------------
df = pd.read_csv("chronic_kidney_diseas.csv")  # Replace with your CSV path
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# ---------------------------
# Step 2: Preprocessing
# ---------------------------

# Drop rows with missing target
df = df.dropna(subset=['Diagnosis'])

# Use Diagnosis directly (already numeric: 1 = CKD, 0 = Not CKD)
y = df['Diagnosis']

# Drop non-predictive columns
X = df.drop(["Diagnosis", "PatientID", "DoctorInCharge"], axis=1)

# Handle missing values in features
# Numeric columns: fill with mean
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# Categorical columns: fill with "Unknown" and encode
categorical_cols = X.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    X[col] = X[col].fillna("Unknown").astype("category").cat.codes

# ---------------------------
# Step 3: Split Data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

# ---------------------------
# Step 4: Train XGBoost Model
# ---------------------------
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# ---------------------------
# Step 5: Evaluate Model
# ---------------------------
y_pred = xgb_model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Step 6: Save the Model
# ---------------------------
joblib.dump(xgb_model, "kidney_xgb_model.pkl")
print("\nModel saved as 'kidney_xgb_model.pkl'")

