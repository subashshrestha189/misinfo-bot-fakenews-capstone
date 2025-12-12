# src/app.py
from src.twibot_features import build_features
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from pathlib import Path
import sys 
import pandas as pd
from src.ensemble import compute_heuristic_score, combine_scores



# ------------- CONFIG PATHS -----------------
BASE_DIR = Path(__file__).resolve().parent.parent 
sys.path.append(str(BASE_DIR / "src"))  # ensure src is on sys.path
from twibot_features import build_features  # import AFTER sys.path tweak
BOT_MODEL_DIR = BASE_DIR / "models" / "bot_tuned"
FAKENEWS_MODEL_DIR = BASE_DIR / "models" / "fake_news_bert_fast"
ENCODER_PATH = BASE_DIR / "models" / "fake_news_baseline" / "label_encoder.joblib"

# ---- BOT THRESHOLDS (tuned) ----
BOT_THRESHOLD = 0.30      # above this, we call the user a bot
HIGH_RISK = 0.60
MEDIUM_RISK = 0.30

# ------------- FASTAPI APP ------------------
app = FastAPI(title="Misinformation Detection API",
              description="Detects bots and fake news using ML models",
              version="1.0")

# ------------- LOAD MODELS ------------------
print("Loading models...")

# --- Fake-news model ---
tokenizer = DistilBertTokenizerFast.from_pretrained(FAKENEWS_MODEL_DIR)
fake_model = DistilBertForSequenceClassification.from_pretrained(FAKENEWS_MODEL_DIR)
label_encoder = joblib.load(ENCODER_PATH)
fake_model.eval()

# --- Bot model ---
bot_model = joblib.load(BOT_MODEL_DIR / "twibot_rf_calibrated.joblib")
bot_schema = joblib.load(BOT_MODEL_DIR / "feature_schema.joblib")
# from twibot_features import build_features


# ------------- SCHEMAS ------------------
class ArticleInput(BaseModel):
    text: str

class UserInput(BaseModel):
    followers_count: float
    following_count: float
    tweet_count: float
    listed_count: float
    account_age_days: float
    has_profile_image: int
    has_description: int
    verified: int
    has_location: int
    has_url: int

class FullInput(BaseModel):
    text: str
    followers_count: float
    following_count: float
    tweet_count: float
    listed_count: float
    account_age_days: float
    has_profile_image: int
    has_description: int
    verified: int
    has_location: int
    has_url: int


# ------------- INTERNAL HELPERS ------------------

def analyze_article_internal(text: str) -> dict:
    """
    Run fake-news model on a single text and return a structured dict.
    """
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        outputs = fake_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
    pred_id = probs.argmax()
    label = label_encoder.inverse_transform([pred_id])[0]
    confidence = float(probs[pred_id])
    probabilities = {label_encoder.classes_[i]: float(p) for i, p in enumerate(probs)}

    return {
        "predicted_label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }

def analyze_user_internal(user_dict: dict) -> dict:
    """
    Run bot model on a single user metadata dict and return a structured dict.
    """
    df = pd.DataFrame([user_dict])
    X = build_features(df)
    X = X.reindex(columns=bot_schema["feature_list"], fill_value=0)

    prob = float(bot_model.predict_proba(X)[:, 1][0])  # P(bot | features)
    is_bot = bool(prob >= BOT_THRESHOLD)

    if prob >= HIGH_RISK:
        risk_level = "High"
    elif prob >= MEDIUM_RISK:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "is_bot": is_bot,
        "bot_probability": round(prob, 3),
        "risk_level": risk_level,
        "threshold_used": BOT_THRESHOLD,
    }




# ------------- ROUTES ------------------

@app.get("/")
def home():
    return {"message": "Misinformation Detection API is running!"}


@app.post("/analyze/article")
def analyze_article(data: ArticleInput):
    """Fake-news analysis endpoint"""
    return analyze_article_internal(data.text)


@app.post("/analyze/user")
def analyze_user(data: UserInput):
    """Bot detection endpoint"""
    # basic sanity checks on numeric fields
    if data.followers_count < 0 or data.following_count < 0 or data.tweet_count < 0:
        raise HTTPException(status_code=400, detail="Counts (followers, following, tweets) cannot be negative.")
    if data.account_age_days <= 0:
        raise HTTPException(status_code=400, detail="Account age must be a positive number of days.")

    return analyze_user_internal(data.dict())

@app.post("/analyze/full")
def analyze_full(data: FullInput):
    """
    Combined endpoint:
    - analyzes article text with BERT
    - analyzes user with RF bot model
    - computes overall trust_score using ensemble weights + heuristics
    """

    # Reuse individual analyzers
    article_result = analyze_article_internal(data.text)

    user_payload = {
        "followers_count": data.followers_count,
        "following_count": data.following_count,
        "tweet_count": data.tweet_count,
        "listed_count": data.listed_count,
        "account_age_days": data.account_age_days,
        "has_profile_image": data.has_profile_image,
        "has_description": data.has_description,
        "verified": data.verified,
        "has_location": data.has_location,
        "has_url": data.has_url,
    }

    # basic sanity checks (same as /analyze/user)
    if user_payload["followers_count"] < 0 or user_payload["following_count"] < 0 or user_payload["tweet_count"] < 0:
        raise HTTPException(status_code=400, detail="Counts (followers, following, tweets) cannot be negative.")
    if user_payload["account_age_days"] <= 0:
        raise HTTPException(status_code=400, detail="Account age must be a positive number of days.")

    user_result = analyze_user_internal(user_payload)

    # Compute heuristics
    heur_score = compute_heuristic_score(
        user=user_payload,
        article_label=article_result.get("predicted_label"),
        article_confidence=article_result.get("confidence"),
    )

    # Combine into trust score
    ensemble = combine_scores(
        content_confidence=article_result.get("confidence", 0.0),
        bot_probability=user_result.get("bot_probability", 0.0),
        heuristic_score=heur_score,
    )

    return {
        "article": article_result,
        "user": user_result,
        "heuristics": {
            "heuristic_score": heur_score
        },
        "ensemble": ensemble
    }

