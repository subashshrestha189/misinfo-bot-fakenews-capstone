Fake News Detection System
A machine learning project implementing two approaches for automated fake news detection: a baseline TF-IDF + Logistic Regression model and an advanced DistilBERT transformer model.
Project Overview
This system classifies news articles into different subject categories to identify potential misinformation. The project demonstrates both traditional NLP techniques and modern deep learning approaches.
Features

Dual Model Architecture: Traditional ML and transformer-based models
Fast Training: Optimized for quick fine-tuning on CPU/GPU
Multi-class Classification: Supports multiple news subject categories
Production-Ready: Includes inference utilities and model persistence

Requirements
pandas
numpy
scikit-learn
joblib
torch
transformers
Install dependencies:
bashpip install pandas numpy scikit-learn joblib torch transformers
Project Structure
.
├── data/
│   └── fake_news/
│       ├── fake_news_clean.csv
│       └── fake_news_encoded.csv
├── models/
│   ├── fake_news_baseline/
│   │   └── label_encoder.joblib
│   ├── fake_news_v2/
│   │   └── fake_news_tfidf_logreg_v2.joblib
│   └── fake_news_bert_fast/
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer files
├── fake_news_retrain.py
├── fake_news_bert_train.py
└── test_fake_news_bert.py
Usage
1. Train Baseline Model (TF-IDF + Logistic Regression)
bashpython fake_news_retrain.py
This script:

Loads and preprocesses the dataset
Filters out classes with insufficient samples
Trains a TF-IDF vectorizer with Logistic Regression
Evaluates on test set and saves the model

Key Parameters:

max_features=40000: TF-IDF vocabulary size
ngram_range=(1, 2): Unigrams and bigrams
max_iter=400: Logistic Regression iterations
class_weight="balanced": Handles class imbalance

2. Train BERT Model (DistilBERT)
bashpython fake_news_bert_train.py
This script:

Loads a subset of data for fast training (5000 samples)
Fine-tunes DistilBERT for sequence classification
Saves model, tokenizer, and training metrics

Key Parameters:

MAX_LENGTH=128: Token sequence length
BATCH_SIZE=8: Training batch size
EPOCHS=1: Quick demonstration training
SUBSET_SIZE=5000: Samples used for training

Note: Adjust SUBSET_SIZE and EPOCHS for full training on complete dataset.
3. Test BERT Model
bashpython test_fake_news_bert.py
Example inference:
pythonfrom test_fake_news_bert import predict

text = "Government announces new healthcare reform plan."
subject, confidence = predict(text)
print(f"Subject: {subject} (Confidence: {confidence:.3f})")
Model Performance
Baseline Model (TF-IDF + Logistic Regression)

Fast training and inference
Interpretable features
Good baseline performance

BERT Model (DistilBERT)

State-of-the-art accuracy
Contextual understanding
Requires more computational resources

Data Requirements
The dataset should contain:

content: News article text
subject or label: Category labels

Supported formats: CSV files with proper encoding.
Configuration
Baseline Model
Edit fake_news_retrain.py:
pythonDATA = Path("data/fake_news/fake_news_encoded.csv")
MODEL_DIR = ensure_dir("models/fake_news_v2")
BERT Model
Edit fake_news_bert_train.py:
pythonBASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "fake_news" / "fake_news_clean.csv"
ENCODER_PATH = BASE_DIR / "models" / "fake_news_baseline" / "label_encoder.joblib"
OUT_DIR = BASE_DIR / "models" / "fake_news_bert_fast"
Performance Optimization
For CPU Training:

Use smaller SUBSET_SIZE (2000-5000)
Reduce MAX_LENGTH to 64
Keep BATCH_SIZE at 8

For GPU Training:

Increase BATCH_SIZE to 16-32
Use full dataset
Increase EPOCHS to 3-5

Troubleshooting
Issue: "Missing dataset" error

Ensure CSV files are in correct directories
Check file paths in configuration

Issue: Out of memory

Reduce BATCH_SIZE
Decrease MAX_LENGTH
Use smaller SUBSET_SIZE

Issue: Stratification error

The code automatically filters classes with <2 samples
Ensure dataset has balanced classes

Future Improvements

Implement cross-validation
Add ensemble methods
Expand to multi-label classification
Deploy as REST API
Add data augmentation techniques

License
This project is for educational purposes.
Acknowledgments

Uses Hugging Face Transformers library
Built with scikit-learn and PyTorch
DistilBERT model by Hugging Face

Authors
Aman - Student Project
