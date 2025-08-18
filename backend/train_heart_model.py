# train_heart_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load dataset
df = pd.read_csv("heart.csv")

print("Dataset Columns:", df.columns)

# 2. Features (X) and Target (y)
X = df.drop(columns=["id", "dataset", "num"])  # drop id, dataset, label column
y = df["num"].apply(lambda x: 1 if x > 0 else 0)  # convert to binary

# 3. Encode categorical features
for col in X.columns:
    if X[col].dtype == "object":  # if column is categorical
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"Encoded {col}: {list(le.classes_)}")

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model
with open("heart_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)  # save model + feature names

print("💾 heart_model.pkl saved successfully!")
