# src/ensemble.py
from __future__ import annotations
from typing import Dict, Any

def compute_heuristic_score(user: Dict[str, Any], article_label: str | None = None, article_confidence: float | None = None) -> float:
    """
    Simple heuristic score in [0,1].
    Higher = more trustworthy / less risky.
    Uses basic profile signals and optional article info.
    """
    # Start from a neutral base
    score = 0.5

    followers = user.get("followers_count", 0) or 0
    account_age_days = user.get("account_age_days", 0) or 0
    has_profile_image = bool(user.get("has_profile_image", 0))
    has_description = bool(user.get("has_description", 0))
    verified = bool(user.get("verified", 0))

    # Older accounts are slightly more trusted
    if account_age_days > 365:
        score += 0.10
    if account_age_days > 730:
        score += 0.05  # > 2 years

    # Basic completeness
    if has_profile_image:
        score += 0.05
    if has_description:
        score += 0.05

    # Verified badge gives a bigger bump
    if verified:
        score += 0.15

    # Very new + very low follower accounts → more suspicious
    if account_age_days < 30 and followers < 20:
        score -= 0.15

    # Optional: adjust slightly based on article topic / confidence
    if article_label:
        label_lower = article_label.lower()
        # Political or breaking news topics tend to carry more risk
        if any(key in label_lower for key in ["politic", "news", "world"]):
            score -= 0.05
    if article_confidence is not None:
        # If the model is very uncertain (< 0.4), reduce trust a bit
        if article_confidence < 0.4:
            score -= 0.05

    # Clip to [0,1]
    score = max(0.0, min(1.0, score))
    return float(score)

def combine_scores(content_confidence: float, bot_probability: float, heuristic_score: float) -> Dict[str, Any]:
    """
    Combine BERT content confidence, bot probability, and heuristics into a single trust score.
    content_confidence: [0,1] – how confident the model is in its classification of the article
    bot_probability:    [0,1] – probability that account is a bot
    heuristic_score:    [0,1] – hand-crafted trust score

    Returns a dict with trust_score [0,1] and trust_level string.
    """
    # We interpret:
    # - higher content_confidence = better (model is more sure about classification)
    # - higher bot_probability = worse (less trustworthy)
    # - higher heuristic_score = better (profile looks more legitimate)
    w_content = 0.5
    w_bot = 0.3
    w_heur = 0.2

    bot_trust_component = 1.0 - bot_probability  # invert: 1 = human-like, 0 = strong bot

    trust_score = (
        w_content * float(content_confidence) +
        w_bot * float(bot_trust_component) +
        w_heur * float(heuristic_score)
    )

    # Clip to [0,1]
    trust_score = max(0.0, min(1.0, trust_score))

    # Map to trust level
    if trust_score >= 0.75:
        trust_level = "High Trust"
    elif trust_score >= 0.5:
        trust_level = "Moderate Trust"
    else:
        trust_level = "Low Trust"

    return {
        "trust_score": float(trust_score),
        "trust_level": trust_level,
        "weights": {
            "content": w_content,
            "bot": w_bot,
            "heuristics": w_heur,
        }
    }
