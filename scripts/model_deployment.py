"""
model_deployment.py

Loads a trained machine learning model and performs inference on new patient data.
Predictions are saved to a CSV file for downstream use.
"""

import pandas as pd
import joblib
from pathlib import Path

# ------------------------------------------------------
# Resolve project paths safely
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
MODEL_PATH = MODEL_DIR / "rf_model.joblib"   # Change if needed
INPUT_DATA_PATH = DATA_DIR / "processed" / "cleaned_data.csv"
OUTPUT_FILE = OUTPUT_DIR / "model_predictions.csv"

# ------------------------------------------------------
# Load model
# ------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"Loaded model from: {MODEL_PATH}")

# ------------------------------------------------------
# Load input data
# ------------------------------------------------------
if not INPUT_DATA_PATH.exists():
    raise FileNotFoundError(f"Input data not found at {INPUT_DATA_PATH}")

data = pd.read_csv(INPUT_DATA_PATH)

if "outcome" in data.columns:
    X = data.drop(columns=["outcome"])
else:
    X = data

print(f"Loaded input data with shape: {X.shape}")

# ------------------------------------------------------
# Run predictions
# ------------------------------------------------------
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
else:
    predictions = model.predict(X)
    probabilities = None

# ------------------------------------------------------
# Save results
# ------------------------------------------------------
output_df = X.copy()
output_df["predicted_outcome"] = predictions

if probabilities is not None:
    output_df["predicted_risk_score"] = probabilities

output_df.to_csv(OUTPUT_FILE, index=False)

print(f"Predictions saved to: {OUTPUT_FILE}")
