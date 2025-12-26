"""
data_preprocessing.py

Preprocesses raw patient data into a clean, model-ready dataset.
This version uses absolute paths derived from the script location
to avoid working-directory issues.
"""

import pandas as pd
from pathlib import Path

# ------------------------------------------------------
# Resolve project paths (bulletproof)
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "patient_data.csv"
OUT_FILE = PROCESSED_DIR / "cleaned_data.csv"

print(f"Loading raw data from: {RAW_FILE}")

# ------------------------------------------------------
# Load raw data
# ------------------------------------------------------
data = pd.read_csv(RAW_FILE)

# ------------------------------------------------------
# Drop unnecessary columns
# ------------------------------------------------------
drop_cols = [col for col in ["patient_id", "date_of_birth"] if col in data.columns]
data.drop(columns=drop_cols, inplace=True)

# ------------------------------------------------------
# Rename columns for consistency
# ------------------------------------------------------
rename_map = {
    "gender": "sex",
    "disease_progression": "outcome"
}
data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns}, inplace=True)

# ------------------------------------------------------
# Categorical encoding
# ------------------------------------------------------
categorical_cols = [
    col for col in [
        "sex",
        "race",
        "smoker",
        "diabetes",
        "hypertension",
        "heart_disease",
        "stroke"
    ]
    if col in data.columns
]

data = pd.get_dummies(data, columns=categorical_cols)

# ------------------------------------------------------
# Handle missing values
# ------------------------------------------------------
numerical_cols = [
    col for col in [
        "age",
        "weight",
        "height",
        "bmi",
        "comorbidity_score"
    ]
    if col in data.columns
]

for col in numerical_cols:
    if data[col].isna().any():
        if col == "comorbidity_score":
            data[col] = data[col].fillna(0)
        else:
            data[col] = data[col].fillna(data[col].median())

# ------------------------------------------------------
# Save processed data
# ------------------------------------------------------
data.to_csv(OUT_FILE, index=False)

print(f"Preprocessing complete.")
print(f"Cleaned data saved to: {OUT_FILE}")
print(f"Final shape: {data.shape}")
