from pathlib import Path
import datetime, json

summary = {
    "date": datetime.datetime.now().isoformat(),
    "fake_news_model": str(Path("models/fake_news_v2/fake_news_tfidf_logreg_v2.joblib")),
    "encoder": str(Path("models/fake_news_baseline/label_encoder.joblib")),
    "rows": 44000,
    "remarks": "Cleaned, label-encoded, balanced TF-IDF LogisticRegression baseline."
}

out = Path("models/fake_news_v2/training_summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
print(f"✅ Summary saved to {out}")
