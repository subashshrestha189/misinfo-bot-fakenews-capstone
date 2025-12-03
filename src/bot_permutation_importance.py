# src/bot_permutation_importance.py
import joblib, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from twibot_features import build_features

MODEL_DIR = Path("models/bot_tuned")
model = joblib.load(MODEL_DIR / "twibot_rf_calibrated.joblib")

df = pd.read_csv("data/twibot/users.csv")
y = df["label"].astype(int).values
X = build_features(df)

# NOTE: calibrated pipeline accepts raw features (preprocessor inside)
r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std}) \
       .sort_values("importance_mean", ascending=False)

out_csv = MODEL_DIR / "permutation_importance.csv"
imp.to_csv(out_csv, index=False)
print(f"Saved permutation importance → {out_csv}")
print(imp.head(10))
