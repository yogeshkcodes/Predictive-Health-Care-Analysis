import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------
# Project paths (auto-detect repo root)
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 1000

# ------------------------------------------------------
# demographic.csv
# ------------------------------------------------------
demographic = pd.DataFrame({
    "patient_id": range(1, n + 1),
    "age": np.random.randint(18, 90, n),
    "gender": np.random.choice(["Male", "Female"], n)
})

# ------------------------------------------------------
# comorbidities.csv
# ------------------------------------------------------
comorbidities = pd.DataFrame({
    "patient_id": demographic["patient_id"],
    "diabetes": np.random.binomial(1, 0.30, n),
    "hypertension": np.random.binomial(1, 0.40, n),
    "heart_disease": np.random.binomial(1, 0.20, n)
})

# ------------------------------------------------------
# treatments.csv
# ------------------------------------------------------
treatments = pd.DataFrame({
    "patient_id": demographic["patient_id"],
    "medication_count": np.random.poisson(3, n),
    "procedure_count": np.random.poisson(1, n)
})

# ------------------------------------------------------
# outcomes.csv
# ------------------------------------------------------
outcomes = pd.DataFrame({
    "patient_id": demographic["patient_id"],
    "readmitted": np.random.binomial(1, 0.25, n)
})

# ------------------------------------------------------
# Save files EXACTLY where repo expects them
# ------------------------------------------------------
demographic.to_csv(RAW_DIR / "demographic.csv", index=False)
comorbidities.to_csv(RAW_DIR / "comorbidities.csv", index=False)
treatments.to_csv(RAW_DIR / "treatments.csv", index=False)
outcomes.to_csv(RAW_DIR / "outcomes.csv", index=False)

print("Synthetic raw data written to data/raw/")
