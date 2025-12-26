"""
model_evaluation.py

Evaluates predictive models for the patient outcome prediction project
using accuracy, precision, recall, and F1 score.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------
# Load preprocessed data (PATH-SAFE)
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "cleaned_data.csv"

print(f"Loading data from: {DATA_FILE}")

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
lr_preds = lr_model.predict(X_test)

print("\nLogistic Regression Evaluation")
print("Accuracy :", accuracy_score(y_test, lr_preds))
print("Precision:", precision_score(y_test, lr_preds))
print("Recall   :", recall_score(y_test, lr_preds))
print("F1 Score :", f1_score(y_test, lr_preds))

# ------------------------------------------------------
# Decision Tree
# ------------------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print("\nDecision Tree Evaluation")
print("Accuracy :", accuracy_score(y_test, dt_preds))
print("Precision:", precision_score(y_test, dt_preds))
print("Recall   :", recall_score(y_test, dt_preds))
print("F1 Score :", f1_score(y_test, dt_preds))

# ------------------------------------------------------
# Neural Network
# ------------------------------------------------------
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)
nn_preds = nn_model.predict(X_test)

print("\nNeural Network Evaluation")
print("Accuracy :", accuracy_score(y_test, nn_preds))
print("Precision:", precision_score(y_test, nn_preds))
print("Recall   :", recall_score(y_test, nn_preds))
print("F1 Score :", f1_score(y_test, nn_preds))

# ------------------------------------------------------
# Random Forest
# ------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\nRandom Forest Evaluation")
print("Accuracy :", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall   :", recall_score(y_test, rf_preds))
print("F1 Score :", f1_score(y_test, rf_preds))
