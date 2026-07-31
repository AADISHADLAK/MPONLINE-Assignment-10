"""
train_model.py
---------------
Task 1: Data Understanding and Preprocessing
Task 2: Model Development

This script:
    1. Loads the Heart Disease dataset using Pandas.
    2. Displays the first five records.
    3. Identifies numerical features and the target variable.
    4. Checks for missing values.
    5. Splits the dataset into 80% training / 20% testing.
    6. Trains a Random Forest classifier.
    7. Evaluates the model using Accuracy Score.
    8. Saves the trained model (and the feature scaler) using Joblib.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# ----------------------------------------------------------------------

print("=" * 60)
print("TASK 1: DATA UNDERSTANDING AND PREPROCESSING")
print("=" * 60)

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")
print(f"\nDataset shape: {df.shape}")

# 2. Display the first five records
print("\nFirst five records:")
print(df.head())

# 3. Identify numerical features and the target variable
TARGET = "target"
numerical_features = [col for col in df.columns if col != TARGET]
print(f"\nNumerical features ({len(numerical_features)}):")
print(numerical_features)
print(f"\nTarget variable: '{TARGET}'")

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size:  {X_test.shape[0]} rows")

# Feature scaling (helps the model converge / perform better,
# and is saved alongside the model so the API can reuse it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# Task 2: Model Development
# ----------------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 2: MODEL DEVELOPMENT")
print("=" * 60)

# Using Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# Evaluate using Accuracy Score
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel: Random Forest Classifier")
print(f"Accuracy Score: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (useful for the README / conclusion)
importances = pd.Series(model.feature_importances_, index=numerical_features)
print("\nTop 5 most important features:")
print(importances.sort_values(ascending=False).head(5))

# ----------------------------------------------------------------------
# Save the trained model and scaler using Joblib
# ----------------------------------------------------------------------

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(numerical_features, "feature_names.pkl")

print("\nSaved trained model to 'model.pkl'")
print("Saved fitted scaler to 'scaler.pkl'")
print("Saved feature order to 'feature_names.pkl'")

# Save accuracy to a small text file so it can be referenced in the README
with open("accuracy.txt", "w") as f:
    f.write(f"{accuracy:.4f}")

print("\nTraining complete.")
