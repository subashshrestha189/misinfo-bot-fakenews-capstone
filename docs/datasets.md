# Datasets Used in the Project

This project works on two datasets: one for **fake news detection** and one for **bot detection**.

---

# Fake News Dataset

**Source:** HuggingFace dataset portal
**Example dataset:** `ErfanMoosaviMonazzah/fake-news-detection-dataset-English'
**Content:** News article titles and full text with labels indicating whether the content is real or fake.
**Usage in this project:**
  - Loaded and preprocessed with the help of 'preprocess_texts.py' and 'download_fake_news_hf.py'.
  - In `fake_news_baseline.py', a baseline model (TF-IDF + Logistic Regression) is trained.
  - DistilBERT fine-tuning is conducted in `fake_news_bert_train.py'.

The whole CSV files are **not** committed to this repository due to GitHub file size restrictions and licensing.  
Before executing the training scripts, users should get the dataset straight from HuggingFace.

---

## TwiBot-20 Dataset (Bot Detection)

**Source:** Official TwiBot-20 GitHub repository.
**Key files used:**
  - `user.json' – user profile metadata.
  - `label.csv' – labels indicating whether a user is a bot or a human.
**Usage in this project:**
  - Using `twibot_extract.py', raw files are transformed into a tabular format.
  - Engineered features (e.g., follower/following counts, account age, activity rates) are created in `twibot_features.py'.
  -  Models are calibrated using 'bot_train_cv_calibrated.py' and trained using 'bot_train_twibot.py`.

As with the fake news dataset, TwiBot-20 raw data is **not stored** in this repository.  
After obtaining the dataset from its official source, users must execute the extraction scripts that are included.
