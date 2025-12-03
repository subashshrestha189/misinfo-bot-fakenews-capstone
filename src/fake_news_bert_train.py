import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "fake_news" / "fake_news_clean.csv"
ENCODER_PATH = BASE_DIR / "models" / "fake_news_baseline" / "label_encoder.joblib"
OUT_DIR = BASE_DIR / "models" / "fake_news_bert_fast"

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128           # shorter sequences = faster
BATCH_SIZE = 8             # safe on CPU
EPOCHS = 1                 # 1 epoch for demo
LR = 5e-5
SUBSET_SIZE = 5000         # use only first 5k samples

os.makedirs(OUT_DIR, exist_ok=True)

assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"
assert ENCODER_PATH.exists(), f"Missing label encoder: {ENCODER_PATH}"

# ---------- LOAD & SAMPLE DATA ----------
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)

if "content" not in df.columns:
    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else None
    if title_col and text_col:
        df["content"] = (df[title_col].fillna("") + " " + df[text_col].fillna("")).str.lower()
    else:
        raise ValueError("No 'content' column and no (title, text) to rebuild it.")

df = df[["content", "subject"]].dropna().reset_index(drop=True)

# Take subset for fast training
if len(df) > SUBSET_SIZE:
    df = df.sample(SUBSET_SIZE, random_state=42).reset_index(drop=True)
print(f"Using {len(df)} samples for fast fine-tuning.")

# ---------- ENCODE LABELS ----------
print(f"Loading label encoder from {ENCODER_PATH} ...")
label_encoder = joblib.load(ENCODER_PATH)

valid_mask = df["subject"].isin(label_encoder.classes_)
if not valid_mask.all():
    dropped = (~valid_mask).sum()
    print(f"Dropping {dropped} rows with unseen labels.")
    df = df[valid_mask].reset_index(drop=True)

# Drop subjects that have fewer than 2 samples (needed for stratify)
vc = df["subject"].value_counts()
rare = vc[vc < 2]

if not rare.empty:
    print("Dropping subjects with too few samples for stratified split:")
    print(rare)
    df = df[~df["subject"].isin(rare.index)].reset_index(drop=True)

labels = label_encoder.transform(df["subject"].astype(str))
num_labels = len(label_encoder.classes_)
print(f"Number of classes: {num_labels}")
print("Classes:", list(label_encoder.classes_))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["content"].tolist(),
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels if len(set(labels)) > 1 else None
)

# ---------- TOKENIZER ----------
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

train_dataset = FakeNewsDataset(train_texts, train_labels)
val_dataset = FakeNewsDataset(val_texts, val_labels)

# ---------- MODEL ----------
print(f"Loading model: {MODEL_NAME}")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={i: c for i, c in enumerate(label_encoder.classes_)},
    label2id={c: i for i, c in enumerate(label_encoder.classes_)},
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# ---------- METRICS ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

# ---------- TRAINING ARGS ----------
training_args = TrainingArguments(
    output_dir=str(OUT_DIR / "checkpoints"),
    evaluation_strategy="epoch",
    save_strategy="no",            # don't save every epoch - faster
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    report_to=[],                  # no wandb/etc
    load_best_model_at_end=False,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ---------- TRAIN ----------
print("Starting fast fine-tuning ...")
trainer.train()

print("Evaluating on validation set ...")
metrics = trainer.evaluate()
print(metrics)

# ---------- SAVE ----------
print(f"Saving model + tokenizer to {OUT_DIR} ...")
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

summary = {
    "samples_used": int(len(df)),
    "metrics": {k: float(v) for k, v in metrics.items()},
    "base_model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "device": device,
    "note": "Fast subset fine-tune for capstone demo.",
}
with open(OUT_DIR / "bert_training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Fast DistilBERT training complete.")
