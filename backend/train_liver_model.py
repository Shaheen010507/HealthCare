
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import json

# ===== 1. Load dataset =====
df = pd.read_csv("liver.csv")

# ===== 2. Preprocess =====
# Convert Gender to numeric (Male=1, Female=0)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Fill missing values with median
for col in df.columns:
    if df[col].dtype != 'object':
        df[col].fillna(df[col].median(), inplace=True)

# Separate features and target
X = df.drop(columns=["Dataset"])   # Features
y = df["Dataset"].apply(lambda x: 1 if x == 1 else 0)  # 1 = disease, 0 = healthy

# ===== 3. Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== 4. Balance with SMOTE =====
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {np.bincount(y_train)}")
print(f"After SMOTE: {np.bincount(y_train_bal)}")

# ===== 5. Model Training =====
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_bal, y_train_bal)

# ===== 6. Evaluation =====
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===== 7. Save Model and Features =====
with open("liver_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_names.json", "w") as f:
    json.dump(list(X.columns), f)

print("\n💾 liver_model.pkl and feature_names.json saved successfully!")
