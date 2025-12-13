import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Use the SAME directory name as in your training script
MODEL_DIR = BASE_DIR / "models" / "fake_news_bert_fast"
ENCODER_PATH = BASE_DIR / "models" / "fake_news_baseline" / "label_encoder.joblib"

assert MODEL_DIR.exists(), f"Model dir not found: {MODEL_DIR}"
assert ENCODER_PATH.exists(), f"Label encoder not found: {ENCODER_PATH}"

tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
label_encoder = joblib.load(ENCODER_PATH)

model.eval()

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_id = int(probs.argmax())
    subject = label_encoder.inverse_transform([pred_id])[0]
    confidence = float(probs[pred_id])
    return subject, confidence

if __name__ == "__main__":
    sample = "Government announces new healthcare reform plan for low-income families."
    label, conf = predict(sample)
    print(f"Predicted subject: {label} (confidence={conf:.3f})")
