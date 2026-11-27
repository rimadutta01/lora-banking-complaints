"""Gradio demo: Banking complaint classifier using the LoRA adapter."""
import json, torch, gradio as gr
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ADAPTER = "models/lora_r8_adapter"
BASE = "distilbert-base-uncased"
labels = json.load(open(f"{ADAPTER}/labels.json"))
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(labels))
model = PeftModel.from_pretrained(base, ADAPTER).eval()

def predict(text):
    x = tok(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        probs = model(**x).logits.softmax(-1)[0]
    top = probs.topk(3)
    return {labels[i].replace("_", " "): float(p) for p, i in zip(top.values, top.indices)}

gr.Interface(fn=predict, inputs=gr.Textbox(lines=3, label="Customer complaint"),
             outputs=gr.Label(num_top_classes=3, label="Predicted category"),
             title="Banking Complaint Classifier (DistilBERT + LoRA)",
             examples=["My UPI payment failed but money was deducted",
                       "I lost my debit card, please block it",
                       "My NEFT transfer is still pending"]).launch()
