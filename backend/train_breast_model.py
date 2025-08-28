import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("breast_cancer_bd.csv")

# Drop ID column (it's called 'Sample code number')
X = data.drop(columns=["Sample code number", "Class"])
y = data["Class"]

# Replace '?' with NaN in Bare Nuclei
X = X.replace("?", pd.NA)

# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values with median
X = X.fillna(X.median())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "breast_cancer_model.pkl")

print("✅ Model trained and saved as breast_cancer_model.pkl")
