# tests/test_api_basic.py
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_home_ok():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert "API" in data["message"] or "running" in data["message"]

def test_analyze_article_valid():
    payload = {"text": "Government announces new climate policy for renewable energy."}
    resp = client.post("/analyze/article", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_label" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_analyze_article_empty():
    payload = {"text": "   "}
    resp = client.post("/analyze/article", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert "detail" in data
    assert "cannot be empty" in data["detail"].lower()
