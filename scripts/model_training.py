"""
model_training.py

Trains predictive models for the patient outcome prediction project
using preprocessed data. Uses absolute paths to avoid working-directory issues.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# ------------------------------------------------------
# Resolve paths safely
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

DATA_FILE = DATA_DIR / "cleaned_data.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading cleaned data from: {DATA_FILE}")

# ------------------------------------------------------
# Load data
# ------------------------------------------------------
data = pd.read_csv(DATA_FILE)

if "outcome" not in data.columns:
    raise ValueError("Target column 'outcome' not found in cleaned_data.csv")

# ------------------------------------------------------
# Train / test split
# ------------------------------------------------------
X = data.drop(columns=["outcome"])
y = data["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, MODEL_DIR / "lr_model.joblib")

# ------------------------------------------------------
# Decision Tree
# ------------------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, MODEL_DIR / "dt_model.joblib")

# ------------------------------------------------------
# Neural Network
# ------------------------------------------------------
nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)
nn_model.fit(X_train, y_train)
joblib.dump(nn_model, MODEL_DIR / "nn_model.joblib")

# ------------------------------------------------------
# Random Forest
# ------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, MODEL_DIR / "rf_model.joblib")

print("All models trained and saved successfully.")
print(f"Models saved to: {MODEL_DIR}")
