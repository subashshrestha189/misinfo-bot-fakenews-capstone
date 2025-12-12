import json
import time
from typing import Dict, Any

import requests
import pandas as pd
import streamlit as st

from config import API_BASE

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Misinformation Detector",
    page_icon="🛡️",
    layout="wide")
st.write("DEBUG: dashboard script started")


# ---------- SIDEBAR / SETTINGS ----------
with st.sidebar:
    st.title("⚙️ Settings")
    api_base = st.text_input("API Base URL", value=API_BASE, help="Your FastAPI base URL")
    if st.button("Test API Connection"):
        try:
            r = requests.get(f"{api_base}/")
            if r.ok:
                st.success(f"Connected: {r.json().get('message','OK')}")
            else:
                st.error(f"Status {r.status_code}: {r.text[:200]}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    st.markdown("---")
    st.caption("Tip: Keep your FastAPI running in another terminal.")

st.title("🛡️ Unified Misinformation Detection")
st.write("Analyze **articles** for credibility and **users** for bot-likelihood")

tabs = st.tabs(["📰 Article Analyzer", "👤 User Analyzer", "📈 Diagnostics"])

# ---------- HELPERS ----------
def call_api_json(method: str, url: str, payload: Dict[str, Any] | None = None, timeout: int = 30):
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)

def risk_badge(prob: float) -> str:
    if prob > 0.60:  return "🔴 High"
    if prob > 0.30:  return "🟠 Medium"
    return "🟢 Low"

# ---------- TAB 1: ARTICLE ANALYZER ----------
with tabs[0]:
    st.subheader("📰 Analyze Article Text")
    with st.form("article_form"):
        text = st.text_area(
            "Paste article text (or a few paragraphs)",
            height=200,
            placeholder="e.g., Government announces new renewable energy policy for 2026..."
        )
        colA, colB = st.columns([1,1])
        with colA:
            maxlen = st.slider("Token limit (client-side truncate)", 64, 256, 128, step=32)
        with colB:
            submit_article = st.form_submit_button("Analyze Article")

    if submit_article:
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Analyzing with BERT…"):
                payload = {"text": text[:10000]}  # guard against huge pastes
                data, err = call_api_json("POST", f"{api_base}/analyze/article", payload)
                time.sleep(0.1)
            if err:
                st.error(f"API error: {err}")
            elif not data:
                st.error("No response from API.")
            else:
                st.success("Analysis complete.")
                col1, col2 = st.columns([1,1])
                with col1:
                    st.metric("Predicted Label", data.get("predicted_label","—"))
                    st.metric("Confidence", f"{data.get('confidence',0)*100:.1f}%")
                with col2:
                    probs = data.get("probabilities", {})
                    if probs:
                        df = pd.DataFrame(
                            [{"Class": k, "Probability": float(v)} for k,v in probs.items()]
                        ).sort_values("Probability", ascending=False)
                        st.write("Class Probabilities")
                        st.bar_chart(df.set_index("Class"))
                        st.dataframe(df, use_container_width=True)
                st.caption("Model: DistilBERT fine-tuned (fast subset).")

# ---------- TAB 2: USER ANALYZER ----------
with tabs[1]:
    st.subheader("👤 Analyze User Metadata (Bot Detection)")
    st.caption("Enter basic profile/activity stats")

    with st.form("user_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            followers_count   = st.number_input("Followers", min_value=0, value=120, step=1)
            listed_count      = st.number_input("Listed Count", min_value=0, value=10, step=1)
        with c2:
            following_count   = st.number_input("Following", min_value=0, value=200, step=1)
            account_age_days  = st.number_input("Account Age (days)", min_value=1, value=800, step=1)
        with c3:
            tweet_count       = st.number_input("Tweet Count", min_value=0, value=1500, step=1)

        st.markdown("#### Profile Flags")
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1: has_profile_image = st.checkbox("Has Profile Image", value=True)
        with d2: has_description   = st.checkbox("Has Description", value=True)
        with d3: verified          = st.checkbox("Verified", value=False)
        with d4: has_location      = st.checkbox("Has Location", value=True)
        with d5: has_url           = st.checkbox("Has URL", value=False)

        submit_user = st.form_submit_button("Analyze User")

    if submit_user:
        payload = {
            "followers_count": followers_count,
            "following_count": following_count,
            "tweet_count": tweet_count,
            "listed_count": listed_count,
            "account_age_days": account_age_days,
            "has_profile_image": int(has_profile_image),
            "has_description": int(has_description),
            "verified": int(verified),
            "has_location": int(has_location),
            "has_url": int(has_url),
        }
        with st.spinner("Scoring bot probability…"):
            data, err = call_api_json("POST", f"{api_base}/analyze/user", payload)
            time.sleep(0.1)
        if err:
            st.error(f"API error: {err}")
        elif not data:
            st.error("No response from API.")
        else:
            prob = float(data.get("bot_probability", 0.0))
            badge = risk_badge(prob)
            st.success("Analysis complete.")
            st.metric("Bot Probability", f"{prob*100:.1f}%")
            st.write(f"**Risk Level:** {badge}")
            # Visual meter
            st.progress(min(max(prob, 0.0), 1.0))
            st.json(data)

    st.caption("Model: Random Forest with calibrated probabilities.")

# ---------- TAB 3: DIAGNOSTICS ----------
with tabs[2]:
    st.subheader("📈 Diagnostics & Health")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**API Ping**")
        ping_data, ping_err = call_api_json("GET", f"{api_base}/")
        if ping_err:
            st.error(f"Ping failed: {ping_err}")
        else:
            st.success(ping_data)
    with c2:
        st.write("**Config**")
        st.code(json.dumps({"API_BASE": api_base}, indent=2))
    st.caption("If API ping fails, verify FastAPI is running and the base URL is correct.")
