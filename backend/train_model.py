# train_model.py
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the absolute path to diabetes.csv (in the same folder as this script)
csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")

# Load the CSV
try:
    df = pd.read_csv(csv_path)
except pd.errors.EmptyDataError:
    raise Exception(f"CSV file found at {csv_path}, but it appears empty.")

print("✅ CSV loaded successfully!")
print(df.head())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

# Save model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"💾 Model saved as {model_path}")
