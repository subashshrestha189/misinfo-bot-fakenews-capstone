import pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from utils_io import ensure_dir

DATA = Path("data/fake_news/fake_news.csv")
MODEL_DIR = ensure_dir("models/fake_news_baseline")

# 1. load data
df = pd.read_csv(DATA)
print("✅ Data loaded successfully!")

# --- NEW: Check structure of the dataset ---
print("\nDataset Info:")
print(df.info())

# Remove subjects (classes) with fewer than 2 samples
df = df.groupby('subject').filter(lambda x: len(x) > 1)
print("\n✅ Filtered dataset. Remaining classes:")

print("\nSubject value counts:")
print(df['subject'].value_counts())

# 2. combine title + text
df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

X = df["content"]
y = df["subject"].astype(str)

# 3. split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1,2),
        lowercase=True
    )),
    ("clf", LogisticRegression(
        max_iter=300,
        n_jobs=-1
    ))
])

# 5. train
pipe.fit(X_train, y_train)

# 6. evaluate
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

# 7. save
joblib.dump(pipe, MODEL_DIR / "fake_news_tfidf_logreg.joblib")
print(f"✅ Saved model to {MODEL_DIR / 'fake_news_tfidf_logreg.joblib'}")
