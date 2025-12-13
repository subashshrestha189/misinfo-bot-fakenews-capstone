import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA = Path("data/fake_news/fake_news.csv")
df = pd.read_csv(DATA)

print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head(3))

# Distribution of subjects
print("\n--- Subject counts ---")
print(df['subject'].value_counts())

plt.figure(figsize=(8,4))
sns.countplot(y=df['subject'], order=df['subject'].value_counts().index)
plt.title("Class Distribution - Subject")
plt.tight_layout()
plt.show()

# Text length histogram
df["text_len"] = df["text"].astype(str).apply(len)
plt.figure(figsize=(8,4))
sns.histplot(df["text_len"], bins=50, color="purple")
plt.title("Distribution of Text Lengths")
plt.xlabel("Characters per article")
plt.tight_layout()
plt.show()

print("\nAverage length:", df['text_len'].mean())
