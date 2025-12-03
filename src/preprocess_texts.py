import re
import pandas as pd
from pathlib import Path

DATA_IN = Path("data/fake_news/fake_news.csv")
DATA_OUT = Path("data/fake_news/fake_news_clean.csv")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv(DATA_IN)
df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
df = df[["content","subject"]]

df.to_csv(DATA_OUT, index=False)
print(f"✅ Cleaned data saved to {DATA_OUT}  ({len(df)} rows)")
