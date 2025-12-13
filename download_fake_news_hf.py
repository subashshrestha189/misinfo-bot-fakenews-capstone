from datasets import load_dataset
import pandas as pd
from pathlib import Path

print("⏬ Downloading dataset from Hugging Face ...")

# 1. load it
ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

# 2. convert each split to pandas
df_train = ds["train"].to_pandas()
df_test = ds["test"].to_pandas() if "test" in ds else pd.DataFrame()
df_valid = ds["validation"].to_pandas() if "validation" in ds else pd.DataFrame()

# 3. merge
df = pd.concat([df_train, df_valid, df_test], ignore_index=True)

# 4. keep useful columns (as in screenshot)
df = df[["title", "text", "subject"]].dropna().reset_index(drop=True)

# 5. save
out_dir = Path("data/fake_news")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "fake_news.csv"
df.to_csv(out_file, index=False)

print(f"✅ Saved {len(df)} rows to {out_file}")
print("Columns:", list(df.columns))
