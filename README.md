**Social Media Bot and Fake News Detection**

This repository contains a prototype system that detects **fake news articles** and estimates the likelihood that a **social media account is a bot**.

We focus on two main components:

1. **Fake News Analyzer**: fine-tuned DistilBERT model categorizing news articles.
2. **Bot Detection**: Random Forest model trained on TwiBot-20 user metadata.

Both models are provided through a **FastAPI** backend and an interactive **Streamlit** dashboard.

This work was completed as part of the **AIDI1003 Capstone Project** at **Durham College**.

---

## Project Objectives

- Build reliable ML pipelines for detecting misinformation in news text.
- Analyze user activity and information to identify automated (bot) behavior.
- Integrate multiple models into a single functional API.
- Demonstrate dataset handling, preprocessing, training, evaluation, and deployment readiness.

 ---

 ## System Overview

 The system contains two independent ML workflows:

## Fake News Detection
- Dataset: English Fake News dataset from HuggingFace.
- Preprocessing: Text cleaning & tokenization.
- Models:
  - Baseline: **TF-IDF + Logistic Regression**
  - Advanced: **DistilBERT fine-tuning**
- Metrics:
  - **Accuracy:** ~76.5% (validation)
  - **F1-score:** ~0.49 (fast demo fine-tune)
 
## Bot Detection
- Dataset: Official TwiBot-20 user metadata & labels (`user.json`, `label.csv`)
- Feature Engineering:
  - Activity rate
  - Account age
  - Bot interaction metrics
  - Hashtag & tweet statistics
- Model:
  - **Random Forest Classifier**
  - Cross-validated + probability calibration
- Metrics:
  - **Accuracy:** 92.2%
  - **ROC-AUC:** ~0.77

---

## Web Interface

Our Streamlit dashboard contains the following tabs:

## Article Analyzer
Paste news article text → get:
- Fake/Real prediction
- Confidence score

## User Analyzer
Enter user profile metadata → get:
- Bot probability score
- Risk classification

## Diagnostics
System logs and internal test runs.

---

## Project Structure

misinfo-mvp/
  src/
     app.py - FastAPI application endpoints
     dashboard.py - Streamlit dashboard launcher
     Fake News Pipeline:
         fake_news_eda.py
         preprocess_texts.py
         fake_news_baseline.py
         fake_news_bert_train.py
         fake_news_retrain.py
         test_fake_news_bert.py
     Bot Detection Pipeline:
         twibot_extract.py
         twibot_features.py
         bot_baseline.py
         bot_train_twibot.py
         bot_traina_cv_calibrated.py
         bot_infer.py
         bot_permutation_importance.py
     Data Utilities:
         encode_labels.py
         utils_io.py
         save_summary.py
     api helpers:
         mini_app.py
         config.py
     requirements.txt
     .gitignore
     README.md

 ---

 ## Data and Model Storage 

 **Large datasets and trained models are not committed** to GitHub due to platform file size limits.

 Files excluded:
 - Raw CSV datasets (100MB+)
- `.safetensors' transformer weights (250MB+)
- TwiBot JSON datasets (multi-GB size)
- Virtual environment ('.venv/')

All training can be fully reproduced by users following the instructions below.

---

## Datasets

## Fake News Dataset
Source:
- HuggingFace dataset portal  
- Example:  
  https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English

Downloaded using:
python src/download_fake_news_hf.py

## TwiBot-20 Dataset
Source:
- https://github.com/BunsenFeng/TwiBot-20

Files used:
- user.json
- label.csv

Feature extraction:
python src/twibot_extract.py

## Setup and Installation

1. Clone the Repo

git clone https://github.com/subashshrestha189/misinfo-bot-fakenews-capstone.git
cd misinfo-bot-fakenews-capstone

2. Create Virtual Environment

python -m venv .venv
.\.venv\Scripts\activate

3. Install Requirements

pip install -r requirements.txt

## Model Training

Fake News Baseline:
python src/fake_news_baseline.py

DistilBERT Fine-Tune:
python src/fake_news_bert_train.py

Bot Detection Training:
python src/twibot_extract.py
python src/bot_train_twibot.py
python src/bot_train_cv_calibrated.py

API Launch (FastAPI)
uvicorn src.app:app --reload

## Endpoints:
- POST /analyze/article
- POST /analyze/user

Interactive documentation:
http://127.0.0.1:8000/docs

## Dashboard Launch (Streamlit)
streamlit run src/dashboard.py

---

## Performance Results

## Bot Detection
- Samples: 700,000 accounts
- Accuracy: 92.2%
- ROC-AUC: 0.77
- Feature count: 20

## Fake News Detection
- Samples: ~5,000 articles
- Base model: distilbert
- Epochs: 1 (lightweight)
- Accuracy: 76.5%
- F1 score: 0.49

---

## Team

Subash Shrestha: Model integration, API deployment, GitHub management & documentation.

Baburam Panta: Bot feature engineering and dataset preprocessing.

Aman Bansal: Transformer fine-tuning.

Laxman Neupane: Backend architecture.

Sisam Kafle: Streamlit UX design.

Aditya Sharma: Fake-news baseline modeling & evaluation.

---

## Reproducibility

This project is completely reproducible:

- Install Python environment.
- Download datasets from official sources.
- Execute scripts for preprocessing.
- Train the models.
- Open the dashboard and API.

Due to dataset license and size limits, raw data and weights are collected externally but pipelines are fully automated using this repository.

---

## License

Academic use only - Durham College Capstone Project
