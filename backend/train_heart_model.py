# train_heart_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------------------
# Step 1: Load Dataset
# ---------------------------
df = pd.read_csv('heart_new.csv')  # Replace with your CSV file path
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# ---------------------------
# Step 2: Preprocessing
# ---------------------------
# Features and Target
X = df.drop('target', axis=1)
y = df['target']

# Optional: If needed, you can encode categorical columns (your dataset seems numeric)

# ---------------------------
# Step 3: Split Data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

# ---------------------------
# Step 4: Train Random Forest
# ---------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# ---------------------------
# Step 5: Evaluate Initial Model
# ---------------------------
y_pred = rf_model.predict(X_test)
print("\n--- Initial Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Step 6: Hyperparameter Tuning (Optional)
# ---------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# ---------------------------
# Step 7: Evaluate Best Model
# ---------------------------
y_pred_best = best_model.predict(X_test)
print("\n--- Best Model Evaluation ---")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))

# ---------------------------
# Step 8: Save the Model
# ---------------------------
joblib.dump(best_model, 'heart_random_forest_model.pkl')
print("\nModel saved as 'heart_random_forest_model.pkl'")
