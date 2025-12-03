# tests/test_api_user.py
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def valid_user_payload():
    return {
        "followers_count": 120,
        "following_count": 200,
        "tweet_count": 1500,
        "listed_count": 10,
        "account_age_days": 800,
        "has_profile_image": 1,
        "has_description": 1,
        "verified": 0,
        "has_location": 1,
        "has_url": 0
    }

def test_analyze_user_valid():
    resp = client.post("/analyze/user", json=valid_user_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert "is_bot" in data
    assert "bot_probability" in data
    assert 0.0 <= data["bot_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]

def test_analyze_user_negative_counts():
    bad_payload = valid_user_payload()
    bad_payload["followers_count"] = -10
    resp = client.post("/analyze/user", json=bad_payload)
    assert resp.status_code == 400
    data = resp.json()
    assert "detail" in data
    assert "cannot be negative" in data["detail"].lower()

def test_analyze_user_zero_age():
    bad_payload = valid_user_payload()
    bad_payload["account_age_days"] = 0
    resp = client.post("/analyze/user", json=bad_payload)
    assert resp.status_code == 400
    data = resp.json()
    assert "account age" in data["detail"].lower()
