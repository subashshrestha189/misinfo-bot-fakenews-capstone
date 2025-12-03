import pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from utils_io import ensure_dir

DATA = Path("data/fake_news/fake_news_clean.csv")
MODEL_DIR = ensure_dir("models/fake_news_baseline")

df = pd.read_csv(DATA)
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["subject"].astype(str))

# Save encoder mapping
mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
print("✅ Label encoder saved:", mapping)

# Export encoded dataset for training
df[["content","label"]].to_csv("data/fake_news/fake_news_encoded.csv", index=False)
print("✅ Encoded dataset saved.")
