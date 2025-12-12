## Project Overview

This project builds a dual-module system to detect misinformation on social media:

Fake News Classifier – Uses NLP (DistilBERT) to classify text as real or fake.

Bot Detection Model – Uses behavioral metadata and a Random Forest model to estimate bot likelihood.

A FastAPI backend serves predictions, and a Streamlit dashboard allows real-time testing.

## My Role (Data Engineer)

Processed Fake News and TwiBot-20 datasets

Converted raw JSON → clean CSV structures

Built preprocessing pipelines and engineered features

Performed EDA and prepared training-ready datasets

Managed GitHub code, version control, and documentation

## Tech Stack

Python, scikit-learn, Transformers (DistilBERT), pandas, numpy, FastAPI, Streamlit

How to Run

Install dependencies:

pip install -r requirements.txt


## Start API:

uvicorn src.app:app --reload


Start Dashboard:

streamlit run src/dashboard.py

Performance Summary

Fake News Model (DistilBERT)

Accuracy: 76.5%

F1 Score: 48.7%

Bot Detection Model (Random Forest)

Accuracy: 92.2%

F1 Score: 88.5%
