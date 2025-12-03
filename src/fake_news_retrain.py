import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from utils_io import ensure_dir

DATA = Path("data/fake_news/fake_news_encoded.csv")
MODEL_DIR = ensure_dir("models/fake_news_v2")

# 1. Load data
df = pd.read_csv(DATA)

# 2. Keep only labels that have at least 2 samples
vc = df["label"].value_counts()
valid_labels = vc[vc >= 2].index
df = df[df["label"].isin(valid_labels)].copy()

# 3. Drop rows with missing text or label
df = df.dropna(subset=["content", "label"])

# (Optional safety) ensure content is string
df["content"] = df["content"].astype(str)

X = df["content"]
y = df["label"]

# 4. Stratified split (now all classes + texts are valid)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Pipeline: TF-IDF + Logistic Regression
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=-1))
])

pipe.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

# 7. Save model
model_path = MODEL_DIR / "fake_news_tfidf_logreg_v2.joblib"
joblib.dump(pipe, model_path)
print(f"✅ Retrained model saved to {model_path}")
