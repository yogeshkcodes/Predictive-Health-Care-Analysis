"""
model_refinement.py

Performs model refinement for the patient outcome prediction project,
including hyperparameter tuning and feature importance analysis.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# ------------------------------------------------------
# Resolve paths safely
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from: {DATA_FILE}")

# ------------------------------------------------------
# Load data
# ------------------------------------------------------
data = pd.read_csv(DATA_FILE)

if "outcome" not in data.columns:
    raise ValueError("Target column 'outcome' not found in cleaned_data.csv")

X = data.drop(columns=["outcome"])
y = data["outcome"]

# ------------------------------------------------------
# Hyperparameter tuning (Random Forest)
# ------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20],
    "min_samples_leaf": [1, 2, 5]
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X, y)

best_params = grid_search.best_params_
print("\nBest hyperparameters for Random Forest:")
print(best_params)

# ------------------------------------------------------
# Train refined model
# ------------------------------------------------------
best_model = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X, y)

# Save refined model
MODEL_PATH = MODEL_DIR / "rf_model_refined.joblib"
joblib.dump(best_model, MODEL_PATH)
print(f"Refined model saved to: {MODEL_PATH}")

# ------------------------------------------------------
# Feature importance analysis
# ------------------------------------------------------
importances = best_model.feature_importances_

feature_importances = (
    pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\nTop feature importances:")
print(feature_importances.head(15))

# Save feature importance report
FEATURE_REPORT_PATH = REPORT_DIR / "feature_importances.csv"
feature_importances.to_csv(FEATURE_REPORT_PATH, index=False)

print(f"Feature importance report saved to: {FEATURE_REPORT_PATH}")
