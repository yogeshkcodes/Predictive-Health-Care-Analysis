import pandas as pd
from pathlib import Path

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# ------------------------------------------------------
# Load synthetic tables
# ------------------------------------------------------
demographic = pd.read_csv(RAW / "demographic.csv")
comorbidities = pd.read_csv(RAW / "comorbidities.csv")
treatments = pd.read_csv(RAW / "treatments.csv")
outcomes = pd.read_csv(RAW / "outcomes.csv")

# ------------------------------------------------------
# Merge into single patient-level table
# ------------------------------------------------------
df = demographic.merge(comorbidities, on="patient_id", how="left")
df = df.merge(treatments, on="patient_id", how="left")
df = df.merge(outcomes, on="patient_id", how="left")

# ------------------------------------------------------
# Create columns EXPECTED by data_preprocessing.py
# ------------------------------------------------------
df["date_of_birth"] = 2024 - df["age"]  # dummy but valid
df["race"] = "Unknown"
df["smoker"] = 0
df["stroke"] = 0
df["weight"] = 70
df["height"] = 170
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["comorbidity_score"] = (
    df["diabetes"]
    + df["hypertension"]
    + df["heart_disease"]
    + df["stroke"]
)

# Rename to match preprocessing logic
df.rename(columns={
    "gender": "gender",
    "readmitted": "disease_progression"
}, inplace=True)

# ------------------------------------------------------
# Save EXACT file expected by repo
# ------------------------------------------------------
OUT = RAW / "patient_data.csv"
df.to_csv(OUT, index=False)

print("patient_data.csv created successfully in data/raw/")
